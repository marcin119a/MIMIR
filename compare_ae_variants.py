"""
compare_ae_variants.py
======================
Pipeline end-to-end dla 3 wariantów Phase 1 → Phase 2.

Warianty:
  old       Adam, stała LR, brak LN, płytka arch
  new_flat  AdamW+cosine+LN+early-stop, ta sama głębokość co old
  new_deep  AdamW+cosine+LN+early-stop, głębsza architektura

Dla każdego wariantu:
  1. Trenuje autoencodery Phase 1 → zapisuje do ae_variants/<variant>/
  2. Trenuje model dzielonej przestrzeni Phase 2
  3. Porównuje wyniki Phase 2 (best_val_total, krzywe strat)

Wynik: tabela w konsoli + compare_ae_variants.png

Usage:
    python compare_ae_variants.py --device cuda
    python compare_ae_variants.py --p1_epochs 80 --p2_epochs 150
"""

import argparse
import json
import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, AdamW

from src.data_utils import (
    SingleModalityDatasetAligned,
    compute_shared_splits,
    get_dataloader,
    load_shared_splits_from_json,
)
from src.mae_masked import (
    build_pretrain_ae_for_modality,
    eval_modality_epoch_masked,
    pretrain_modality_epoch,
    save_modality_with_config,
)
from src.shared_finetune import run_shared_finetune, save_shared_model


# ─── Phase 1 configs ──────────────────────────────────────────────────────────

_SHARED_P1 = dict(mask_value=0.0, activation_dropout=0.05, mask_p=0.3,
                  l1_alpha=1e-4, alpha_mask=0.5)

P1_VARIANTS = {
    "old": {
        "rna": dict(**_SHARED_P1, hidden_layers=[512],       lr=1e-3, weight_decay=1e-5,
                    use_batchnorm=False, grad_clip=0.0, patience=None),
        "mth": dict(**_SHARED_P1, hidden_layers=[256],       lr=1e-3, weight_decay=1e-5,
                    use_batchnorm=False, grad_clip=0.0, patience=None),
    },
    "new_flat": {
        "rna": dict(**_SHARED_P1, hidden_layers=[512],       lr=1e-3, weight_decay=1e-4,
                    use_batchnorm=True,  grad_clip=1.0, patience=15),
        "mth": dict(**_SHARED_P1, hidden_layers=[256],       lr=1e-3, weight_decay=1e-4,
                    use_batchnorm=True,  grad_clip=1.0, patience=15),
    },
    "new_deep": {
        "rna": dict(**_SHARED_P1, hidden_layers=[1024, 512], lr=1e-3, weight_decay=1e-4,
                    use_batchnorm=True,  grad_clip=1.0, patience=15),
        "mth": dict(**_SHARED_P1, hidden_layers=[512, 256],  lr=1e-3, weight_decay=1e-4,
                    use_batchnorm=True,  grad_clip=1.0, patience=15),
    },
}

MODALITY_KEY_MAP = {"rna": "rna", "mth": "methylation"}  # short → data key
AE_NAME_MAP      = {"rna": "rna", "mth": "mth"}          # short → checkpoint prefix

VARIANT_ORDER  = ["old", "new_flat", "new_deep"]
VARIANT_COLORS = {"old": "#e07b54", "new_flat": "#4caf7d", "new_deep": "#4c8bb5"}


# ─── Phase 1 training ─────────────────────────────────────────────────────────

def train_p1_variant(
    variant_name, cfg_by_mod, multi_omic_data,
    common_samples, train_idx, val_idx,
    n_epochs, device, out_dir, batch_size=128,
):
    os.makedirs(out_dir, exist_ok=True)
    p1_summary = {}

    for short_name, data_key in MODALITY_KEY_MAP.items():
        if data_key not in multi_omic_data:
            print(f"  [SKIP] '{data_key}' not in data")
            continue

        cfg = cfg_by_mod[short_name]
        data_df = multi_omic_data[data_key]
        use_new = variant_name != "old"

        print(f"  [{variant_name}] {short_name.upper()}"
              f"  hidden={cfg['hidden_layers']}"
              f"  LN={cfg['use_batchnorm']}"
              f"  clip={cfg['grad_clip']}"
              f"  patience={cfg['patience']}")

        ds = SingleModalityDatasetAligned(data_df, common_samples)
        train_loader = get_dataloader(ds, batch_size=batch_size, shuffle=True,  split_idx=train_idx)
        val_loader   = get_dataloader(ds, batch_size=batch_size, shuffle=False, split_idx=val_idx)

        ae, _ = build_pretrain_ae_for_modality(
            input_dim=data_df.shape[1],
            hidden_layers=cfg["hidden_layers"],
            activation_dropout=cfg["activation_dropout"],
            denoising=True, mask_p=cfg["mask_p"], tied=False,
            mask_value=cfg["mask_value"], loss_on_masked=True,
            use_batchnorm=cfg["use_batchnorm"],
        )
        ae = ae.to(device)

        if use_new:
            opt = AdamW(ae.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_epochs, eta_min=cfg["lr"] * 0.01
            )
        else:
            opt = Adam(ae.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            scheduler = None

        best_val_masked = float("inf")
        best_state      = None
        no_improve      = 0
        patience        = cfg.get("patience")
        epochs_run      = 0

        t0 = time.time()
        for ep in range(1, n_epochs + 1):
            _, tr_overall, tr_masked = pretrain_modality_epoch(
                ae, train_loader, opt, device,
                l1_alpha=cfg["l1_alpha"], alpha_mask=cfg["alpha_mask"],
                grad_clip=cfg["grad_clip"],
            )
            va_overall, va_masked = eval_modality_epoch_masked(ae, val_loader, device)
            if scheduler:
                scheduler.step()

            epochs_run = ep
            if va_masked < best_val_masked:
                best_val_masked = va_masked
                best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if patience and no_improve >= patience:
                print(f"    early stop ep={ep}")
                break

        if best_state:
            ae.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        ckpt_prefix = os.path.join(out_dir, f"{AE_NAME_MAP[short_name]}_ae")
        ae_config = dict(
            input_dim=data_df.shape[1], hidden_layers=cfg["hidden_layers"],
            activation_dropout=cfg["activation_dropout"], denoising=True,
            mask_p=cfg["mask_p"], tied=False, mask_value=cfg["mask_value"],
            loss_on_masked=True, use_batchnorm=cfg["use_batchnorm"],
        )
        save_modality_with_config(ae, ae_config, ckpt_prefix)

        elapsed = time.time() - t0
        p1_summary[short_name] = dict(
            best_val_masked=best_val_masked, epochs=epochs_run, time_s=elapsed
        )
        print(f"    → best_val_masked={best_val_masked:.4f}  "
              f"epochs={epochs_run}  time={elapsed:.1f}s  → {ckpt_prefix}.pt")

    return p1_summary


# ─── Phase 2 training ─────────────────────────────────────────────────────────

def train_p2_variant(
    variant_name, ae_dir, multi_omic_data,
    common_samples, train_idx, val_idx, test_idx,
    p2_cfg, device,
):
    name_map   = {"rna": "rna", "methylation": "mth"}
    model_paths = {mod: os.path.join(ae_dir, f"{name_map[mod]}_ae.pt")
                   for mod in multi_omic_data}

    missing = [p for p in model_paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing AE checkpoints: {missing}")

    t0 = time.time()
    model, train_hist, val_hist, *_ = run_shared_finetune(
        multi_omic_data=multi_omic_data,
        common_samples=common_samples,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_paths=model_paths,
        device=device,
        verbose=False,
        **p2_cfg,
    )
    elapsed = time.time() - t0

    best_val = min(val_hist["total"])
    best_ep  = val_hist["total"].index(best_val) + 1
    print(f"  [{variant_name}] P2: best_val_total={best_val:.4f}"
          f"  ep={best_ep}/{len(val_hist['total'])}"
          f"  time={elapsed:.0f}s")

    return {
        "train_hist": train_hist,
        "val_hist":   val_hist,
        "best_val":   best_val,
        "best_ep":    best_ep,
        "total_ep":   len(val_hist["total"]),
        "time_s":     elapsed,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_p2_comparison(p2_results: dict, save_path: str):
    loss_keys  = ["total", "recon", "contrast", "impute"]
    loss_titles = ["Total Loss", "Recon Loss", "Contrastive Loss", "Impute Loss"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    for col, (lk, lt) in enumerate(zip(loss_keys, loss_titles)):
        for row, split in enumerate(["train_hist", "val_hist"]):
            ax = axes[row][col]
            split_label = "Train" if split == "train_hist" else "Val"
            for v in VARIANT_ORDER:
                if v not in p2_results:
                    continue
                y = p2_results[v][split][lk]
                best = min(y)
                ax.plot(range(1, len(y) + 1), y,
                        color=VARIANT_COLORS[v], linewidth=1.5,
                        label=f"{v} (best={best:.4f})")
            ax.set_title(f"{split_label} {lt}", fontsize=9)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("Loss", fontsize=8)
            ax.legend(fontsize=7)

    plt.suptitle("Phase 2 comparison: OLD vs NEW_FLAT vs NEW_DEEP", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {save_path}")


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/tcga_redo_mlomicZ.pkl")
    p.add_argument("--splits",     default="data/splits.json")
    p.add_argument("--device",     default=None)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--p1_epochs",  type=int,   default=100,
                   help="Max epochs for Phase 1 (early-stop może skrócić)")
    p.add_argument("--p2_epochs",  type=int,   default=150,
                   help="Max epochs dla Phase 2")
    p.add_argument("--ae_root",    default="ae_variants",
                   help="Katalog bazowy dla checkpointów Phase 1")
    p.add_argument("--out",        default="compare_ae_variants.png")
    p.add_argument("--variants",   nargs="+", default=VARIANT_ORDER,
                   choices=VARIANT_ORDER,
                   help="Które warianty uruchomić")
    p.add_argument("--skip_p1",    action="store_true",
                   help="Pomiń Phase 1 (zakłada że checkpointy już istnieją)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    with open(args.data, "rb") as f:
        multi_omic_data = pickle.load(f)

    ACTIVE = ["rna", "methylation"]
    multi_omic_data = {k: v for k, v in multi_omic_data.items() if k in ACTIVE}

    if os.path.exists(args.splits):
        common_samples, train_idx, val_idx, test_idx = load_shared_splits_from_json(
            multi_omic_data, args.splits
        )
    else:
        common_samples, train_idx, val_idx, test_idx = compute_shared_splits(
            multi_omic_data, val_size=0.1, test_size=0.2, seed=42
        )
    print(f"Splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}\n")

    # Phase 2 hyperparameters — identyczne dla wszystkich wariantów (fair comparison)
    P2_CFG = dict(
        shared_dim=128, proj_depth=1, proj_activation_dropout=0.1,
        batch_size=64, lr=3e-4, weight_decay=1e-4,
        epochs=args.p2_epochs,
        lambda_contrast=1.0, lambda_impute=1.0,
        modality_dropout_prob=0.3, feature_mask_p_train=0.2, feature_mask_p_val=0.2,
        alpha_mask_recon=0.5, two_path_clean_for_contrast=False,
        freeze_encoders_decoders=False,
        grad_clip=1.0, early_stopping_patience=15,
        lr_scheduler_patience=7, lr_scheduler_factor=0.5,
    )

    p1_summaries = {}
    p2_results   = {}

    for variant in args.variants:
        ae_dir = os.path.join(args.ae_root, variant)
        print(f"\n{'='*60}")
        print(f"  VARIANT: {variant.upper()}")
        print(f"{'='*60}")

        # ── Phase 1 ──────────────────────────────────────────────
        if not args.skip_p1:
            print(f"\n--- Phase 1 ---")
            p1_summaries[variant] = train_p1_variant(
                variant_name=variant,
                cfg_by_mod=P1_VARIANTS[variant],
                multi_omic_data=multi_omic_data,
                common_samples=common_samples,
                train_idx=train_idx,
                val_idx=val_idx,
                n_epochs=args.p1_epochs,
                device=device,
                out_dir=ae_dir,
                batch_size=args.batch_size,
            )
        else:
            print(f"  [skip P1] using existing checkpoints from {ae_dir}")

        # ── Phase 2 ──────────────────────────────────────────────
        print(f"\n--- Phase 2 ---")
        p2_results[variant] = train_p2_variant(
            variant_name=variant,
            ae_dir=ae_dir,
            multi_omic_data=multi_omic_data,
            common_samples=common_samples,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            p2_cfg=P2_CFG,
            device=device,
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  PHASE 1 summary")
    print(f"{'='*78}")
    if p1_summaries:
        print(f"{'Variant':<12} {'Modality':<8} {'best_val_masked':>16} {'epochs':>7} {'time(s)':>8}")
        print("─" * 55)
        for v, mods in p1_summaries.items():
            for m, s in mods.items():
                print(f"{v:<12} {m:<8} {s['best_val_masked']:>16.4f}"
                      f" {s['epochs']:>7} {s['time_s']:>8.1f}")
            print()

    print(f"\n{'='*78}")
    print(f"  PHASE 2 summary  (identyczne hiperparametry P2 dla wszystkich)")
    print(f"{'='*78}")
    ref_variant = args.variants[0]
    ref_val = p2_results[ref_variant]["best_val"]
    print(f"{'Variant':<12} {'best_val_total':>15} {'vs ' + ref_variant:>12}"
          f" {'best_ep':>8} {'total_ep':>9} {'time(s)':>8}")
    print("─" * 68)
    for v in args.variants:
        r    = p2_results[v]
        diff = r["best_val"] - ref_val
        pct  = 100 * diff / (abs(ref_val) + 1e-9)
        if v == ref_variant:
            delta = "  (ref)"
        else:
            arrow = "↓" if diff < 0 else "↑"
            delta = f"  {arrow}{abs(pct):.1f}%"
        print(f"{v:<12} {r['best_val']:>15.4f} {delta:>12}"
              f" {r['best_ep']:>8} {r['total_ep']:>9} {r['time_s']:>8.0f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_p2_comparison(p2_results, args.out)

    # ── Save JSON summary ─────────────────────────────────────────────────────
    json_path = args.out.replace(".png", ".json")
    summary = {
        v: {"p2_best_val": r["best_val"], "p2_best_ep": r["best_ep"],
            "p2_total_ep": r["total_ep"], "p2_time_s": r["time_s"]}
        for v, r in p2_results.items()
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary → {json_path}")


if __name__ == "__main__":
    main()
