"""
compare_phase1.py — porównuje stary vs nowy trening autoencoderów Phase 1.

Trenuje 3 warianty dla każdej modalności na tych samych danych/splitach:
  OLD      : Adam, stała LR, brak LN, płytka arch [512]/[256]
  NEW_FLAT : AdamW + cosine + LayerNorm + early-stop, ta sama głębokość co OLD
  NEW_DEEP : jak NEW_FLAT ale głębsza architektura [1024,512]/[512,256]

Pozwala oddzielić efekt optymalizatora/schedulera od efektu architektury.
Wynik: tabela w konsoli + compare_phase1_results.png

Usage:
    python compare_phase1.py
    python compare_phase1.py --data data/tcga_redo_mlomicZ.pkl --device cuda
"""

import argparse
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
)


# ─── Config triplets ───────────────────────────────────────────────────────────

_SHARED = dict(mask_value=0.0, activation_dropout=0.05, mask_p=0.3,
               l1_alpha=1e-4, alpha_mask=0.5)

MODALITY_TRIPLETS = [
    {
        "name": "rna",
        "key":  "rna",
        "old":      dict(**_SHARED, hidden_layers=[512],       n_epochs=100,
                         lr=1e-3, weight_decay=1e-5,
                         use_batchnorm=False, grad_clip=0.0, patience=None),
        "new_flat": dict(**_SHARED, hidden_layers=[512],       n_epochs=100,
                         lr=1e-3, weight_decay=1e-4,
                         use_batchnorm=True,  grad_clip=1.0, patience=15),
        "new_deep": dict(**_SHARED, hidden_layers=[1024, 512], n_epochs=100,
                         lr=1e-3, weight_decay=1e-4,
                         use_batchnorm=True,  grad_clip=1.0, patience=15),
    },
    {
        "name": "mth",
        "key":  "methylation",
        "old":      dict(**_SHARED, hidden_layers=[256],      n_epochs=100,
                         lr=1e-3, weight_decay=1e-5,
                         use_batchnorm=False, grad_clip=0.0, patience=None),
        "new_flat": dict(**_SHARED, hidden_layers=[256],      n_epochs=100,
                         lr=1e-3, weight_decay=1e-4,
                         use_batchnorm=True,  grad_clip=1.0, patience=15),
        "new_deep": dict(**_SHARED, hidden_layers=[512, 256], n_epochs=100,
                         lr=1e-3, weight_decay=1e-4,
                         use_batchnorm=True,  grad_clip=1.0, patience=15),
    },
]

VARIANT_ORDER  = ["old", "new_flat", "new_deep"]
VARIANT_COLORS = {"old": "#e07b54", "new_flat": "#4caf7d", "new_deep": "#4c8bb5"}
VARIANT_LABELS = {
    "old":      "OLD  (Adam, flat LR, no LN, shallow)",
    "new_flat": "NEW_FLAT (AdamW+cosine+LN, same depth)",
    "new_deep": "NEW_DEEP (AdamW+cosine+LN, deeper arch)",
}


# ─── Single training run ───────────────────────────────────────────────────────

def run_config(cfg, use_new_opt, data_df, common_samples, train_idx, val_idx,
               device, batch_size=128):
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

    if use_new_opt:
        opt = AdamW(ae.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["n_epochs"], eta_min=cfg["lr"] * 0.01
        )
    else:
        opt = Adam(ae.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = None

    history = {"train_overall": [], "train_masked": [], "val_overall": [], "val_masked": []}
    best_val_masked = float("inf")
    epochs_no_improve = 0
    patience = cfg.get("patience")

    t0 = time.time()
    for ep in range(1, cfg["n_epochs"] + 1):
        _, tr_overall, tr_masked = pretrain_modality_epoch(
            ae, train_loader, opt, device,
            l1_alpha=cfg["l1_alpha"], alpha_mask=cfg["alpha_mask"],
            grad_clip=cfg["grad_clip"],
        )
        va_overall, va_masked = eval_modality_epoch_masked(ae, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        history["train_overall"].append(tr_overall)
        history["train_masked"].append(tr_masked)
        history["val_overall"].append(va_overall)
        history["val_masked"].append(va_masked)

        if va_masked < best_val_masked:
            best_val_masked = va_masked
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if patience is not None and epochs_no_improve >= patience:
            break

    return {
        "history":          history,
        "best_val_masked":  best_val_masked,
        "best_val_overall": min(history["val_overall"]),
        "epochs_run":       ep,
        "elapsed_s":        time.time() - t0,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(results: list, save_path: str):
    modalities = list(dict.fromkeys(r["modality"] for r in results))
    # 2 metrics × 2 (train + val) per modality  →  rows=modalities, cols=4
    fig, axes = plt.subplots(len(modalities), 4,
                             figsize=(20, 4.5 * len(modalities)))
    if len(modalities) == 1:
        axes = [axes]

    col_info = [
        ("train_overall", "Train Overall MSE"),
        ("val_overall",   "Val Overall MSE"),
        ("train_masked",  "Train Masked MSE"),
        ("val_masked",    "Val Masked MSE"),
    ]

    for row, mod in enumerate(modalities):
        mod_results = {r["variant"]: r for r in results if r["modality"] == mod}
        for col, (metric_key, metric_title) in enumerate(col_info):
            ax = axes[row][col]
            for v in VARIANT_ORDER:
                if v not in mod_results:
                    continue
                r = mod_results[v]
                y = r["history"][metric_key]
                best = min(y)
                ax.plot(range(1, len(y) + 1), y,
                        color=VARIANT_COLORS[v],
                        label=f"{VARIANT_LABELS[v]}\nbest={best:.4f}",
                        linewidth=1.5)
            ax.set_title(f"{mod.upper()} — {metric_title}", fontsize=9)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("MSE", fontsize=8)
            ax.legend(fontsize=6)

    plt.suptitle("Phase 1 AE: OLD vs NEW_FLAT vs NEW_DEEP", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/tcga_redo_mlomicZ.pkl")
    p.add_argument("--splits",     default="data/splits.json")
    p.add_argument("--device",     default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out",        default="compare_phase1_results.png")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}\n")

    with open(args.data, "rb") as f:
        multi_omic_data = pickle.load(f)

    if os.path.exists(args.splits):
        common_samples, train_idx, val_idx, _ = load_shared_splits_from_json(
            multi_omic_data, args.splits
        )
    else:
        common_samples, train_idx, val_idx, _ = compute_shared_splits(
            multi_omic_data, val_size=0.1, test_size=0.2, seed=42
        )
    print(f"Splits: train={len(train_idx)} val={len(val_idx)}\n")

    all_results = []

    for triplet in MODALITY_TRIPLETS:
        mod_name = triplet["name"]
        mod_key  = triplet["key"]
        if mod_key not in multi_omic_data:
            print(f"[SKIP] '{mod_key}' not in data")
            continue

        data_df = multi_omic_data[mod_key]

        for variant in VARIANT_ORDER:
            cfg      = triplet[variant]
            use_new  = variant != "old"
            tag      = f"{mod_name.upper()} {variant.upper()}"
            print(f"{'─'*58}")
            print(f"  {tag:20s}  hidden={str(cfg['hidden_layers']):<14}"
                  f"  LN={cfg['use_batchnorm']}  clip={cfg['grad_clip']}"
                  f"  patience={cfg['patience']}")

            res = run_config(cfg, use_new, data_df, common_samples,
                             train_idx, val_idx, device, args.batch_size)
            res["modality"] = mod_name
            res["variant"]  = variant
            all_results.append(res)

            print(f"  → best_val_masked={res['best_val_masked']:.4f}"
                  f"  best_val_overall={res['best_val_overall']:.4f}"
                  f"  epochs={res['epochs_run']}"
                  f"  time={res['elapsed_s']:.1f}s")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"{'Modality':<6} {'Variant':<12} {'best_val_masked':>16}"
          f" {'best_val_overall':>17} {'vs OLD masked':>14} {'epochs':>7} {'time(s)':>8}")
    print("─" * 78)

    for mod_name in dict.fromkeys(r["modality"] for r in all_results):
        old_res = next(r for r in all_results
                       if r["modality"] == mod_name and r["variant"] == "old")
        for v in VARIANT_ORDER:
            r = next((x for x in all_results
                      if x["modality"] == mod_name and x["variant"] == v), None)
            if r is None:
                continue
            if v == "old":
                delta_str = "  (baseline)"
            else:
                diff = r["best_val_masked"] - old_res["best_val_masked"]
                pct  = 100 * diff / (old_res["best_val_masked"] + 1e-9)
                arrow = "↓" if diff < 0 else "↑"
                delta_str = f"  {arrow}{abs(pct):.1f}%"

            print(f"{mod_name.upper():<6} {v:<12}"
                  f" {r['best_val_masked']:>16.4f}"
                  f" {r['best_val_overall']:>17.4f}"
                  f" {delta_str:>14}"
                  f" {r['epochs_run']:>7}"
                  f" {r['elapsed_s']:>8.1f}")
        print()

    plot_comparison(all_results, args.out)


if __name__ == "__main__":
    main()
