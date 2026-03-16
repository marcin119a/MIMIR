"""
Phase 2: Train ConditionalMultiModalWithSharedSpace using CVAE encoders/decoders
conditioned on primary site.

Requires Phase-1 CVAE checkpoints (from train_cvae_autoencoders.py).

Usage:
    python train_cvae_shared.py
    python train_cvae_shared.py --data data/tcga_redo_mlomicZ.pkl \\
        --cvae_dir cvae_phase1 --out cvae_phase2_results --epochs 150
    python train_cvae_shared.py --experiments baseline small_shared_32
"""

import argparse
import json
import os
import pickle
import time
from copy import deepcopy
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data_utils import compute_shared_splits, load_shared_splits_from_json
from src.cvae import (
    ConditionalMultiOmicDataset,
    extract_encoder_decoder_from_cvae,
    get_conditional_dataloader,
    load_conditions_from_json,
    load_cvae_with_config,
)
from src.cvae_phase2 import (
    ConditionalMultiModalWithSharedSpace,
    conditional_eval_finetune_epoch,
    conditional_finetune_epoch,
)


# ─── Experiment configs ───────────────────────────────────────────────────────

BASE_CONFIG = dict(
    shared_dim=128,
    proj_depth=1,
    proj_activation_dropout=0.1,
    lr=3e-4,
    weight_decay=1e-4,
    modality_dropout_prob=0.3,
    feature_mask_p_train=0.2,
    feature_mask_p_val=0.2,
    alpha_mask_recon=0.5,
    lambda_contrast=1.0,
    lambda_impute=1.0,
    gaussian_noise_std=0.0,
    grad_clip=1.0,
)

EXPERIMENTS = {
    "baseline":         {},
    "small_shared_64":  dict(shared_dim=64),
    "small_shared_32":  dict(shared_dim=32),
    "small_shared_16":  dict(shared_dim=16),
    "high_dropout_03":  dict(proj_activation_dropout=0.3),
    "high_dropout_05":  dict(proj_activation_dropout=0.5),
    "high_mask_04":     dict(feature_mask_p_train=0.4, feature_mask_p_val=0.4),
    "high_mask_05":     dict(feature_mask_p_train=0.5, feature_mask_p_val=0.5),
    "high_mod_drop_05": dict(modality_dropout_prob=0.5),
    "low_lr_1e4":       dict(lr=1e-4),
    "low_lr_5e5":       dict(lr=5e-5),
    "high_wd_1e3":      dict(weight_decay=1e-3),
    "high_wd_1e2":      dict(weight_decay=1e-2),
    "low_contrast_05":  dict(lambda_contrast=0.5),
    "low_contrast_02":  dict(lambda_contrast=0.2),
    "gaussian_001":     dict(gaussian_noise_std=0.01),
    "combined_heavy":   dict(
        shared_dim=32,
        proj_activation_dropout=0.3,
        feature_mask_p_train=0.4, feature_mask_p_val=0.4,
        modality_dropout_prob=0.5,
        lr=1e-4, weight_decay=1e-3,
        gaussian_noise_std=0.01,
    ),
}


def build_config(overrides: dict) -> dict:
    cfg = deepcopy(BASE_CONFIG)
    cfg.update(overrides)
    return cfg


# ─── Helpers ──────────────────────────────────────────────────────────────────

def plot_loss_curves(train_hist, val_hist, save_path, title=""):
    loss_types = ["total", "recon", "contrast", "impute"]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title)
    for i, lt in enumerate(loss_types):
        axs[i].plot(train_hist[lt], label="Train")
        axs[i].plot(val_hist[lt],   label="Val")
        axs[i].set_title(f"{lt.capitalize()} Loss")
        axs[i].set_xlabel("Epoch")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


# ─── Single experiment ────────────────────────────────────────────────────────

def run_one_experiment(
    exp_name, cfg,
    multi_omic_data, common_samples, condition_matrix,
    train_idx, val_idx,
    cvae_paths,
    device, epochs, batch_size, out_root,
):
    exp_dir = os.path.join(out_root, f"exp_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)

    # ── Load CVAE encoders/decoders ──────────────────────────────────────────
    encoders, decoders, hidden_dims, mask_values = {}, {}, {}, {}
    for mod, path in cvae_paths.items():
        cvae_m, hidden_dim_m, cfg_m = load_cvae_with_config(path, map_location=device)
        cvae_m = cvae_m.to(device)
        enc, dec = extract_encoder_decoder_from_cvae(cvae_m)
        encoders[mod] = enc
        decoders[mod] = dec
        hidden_dims[mod] = hidden_dim_m
        mask_values[mod] = cfg_m.get("mask_value", 0.0)

    # ── Build dataset / loaders ──────────────────────────────────────────────
    ds = ConditionalMultiOmicDataset(
        {m: df.loc[common_samples] for m, df in multi_omic_data.items()},
        condition_matrix,
    )
    train_loader = get_conditional_dataloader(ds, batch_size=batch_size, shuffle=True,  split_idx=train_idx)
    val_loader   = get_conditional_dataloader(ds, batch_size=batch_size, shuffle=False, split_idx=val_idx)

    # ── Build model ──────────────────────────────────────────────────────────
    model = ConditionalMultiModalWithSharedSpace(
        encoders=encoders,
        decoders=decoders,
        hidden_dims=hidden_dims,
        shared_dim=cfg["shared_dim"],
        proj_depth=cfg["proj_depth"],
        activation_dropout=cfg["proj_activation_dropout"],
    ).to(device)

    train_hist = {"total": [], "recon": [], "contrast": [], "impute": []}
    val_hist   = {"total": [], "recon": [], "contrast": [], "impute": []}
    best_val_total = float("inf")
    best_model_state = None
    best_epoch = 0
    train_at_best = float("inf")
    epochs_no_improve = 0
    PATIENCE = 12

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    t0 = time.time()
    for ep in range(epochs):
        tr = conditional_finetune_epoch(
            model=model, dataloader=train_loader, optimizer=opt, device=device,
            mask_values=mask_values,
            lambda_contrastive=cfg["lambda_contrast"],
            lambda_impute=cfg["lambda_impute"],
            modality_dropout_prob=cfg["modality_dropout_prob"],
            feature_mask_p=cfg["feature_mask_p_train"],
            alpha_mask_recon=cfg["alpha_mask_recon"],
            grad_clip=cfg["grad_clip"],
            gaussian_noise_std=cfg["gaussian_noise_std"],
        )
        vl = conditional_eval_finetune_epoch(
            model=model, dataloader=val_loader, device=device,
            mask_values=mask_values,
            lambda_contrastive=cfg["lambda_contrast"],
            lambda_impute=cfg["lambda_impute"],
            feature_mask_p=cfg["feature_mask_p_val"],
            alpha_mask_recon=cfg["alpha_mask_recon"],
        )

        for k_src, k_dst in [("total_loss", "total"), ("recon_loss", "recon"),
                              ("contrast_loss", "contrast"), ("impute_loss", "impute")]:
            train_hist[k_dst].append(tr[k_src])
            val_hist[k_dst].append(vl[k_src])

        scheduler.step(vl["total_loss"])

        if vl["total_loss"] < best_val_total:
            best_val_total = vl["total_loss"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            train_at_best = tr["total_loss"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"  [Early stop] ep {ep + 1} | no improvement for {PATIENCE} epochs")
            break

    elapsed = time.time() - t0

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    total_epochs_run = len(train_hist["total"])
    gap_at_best = best_val_total - train_at_best

    with open(os.path.join(exp_dir, "loss_history.json"), "w") as f:
        json.dump({"train": train_hist, "val": val_hist}, f)

    plot_loss_curves(train_hist, val_hist,
                     save_path=os.path.join(exp_dir, "loss_curves.png"),
                     title=exp_name)

    torch.save(model.state_dict(), os.path.join(exp_dir, "model_best.pt"))

    result = {
        "exp_name":       exp_name,
        "best_val_total": round(best_val_total, 6),
        "train_at_best":  round(train_at_best, 6),
        "gap_at_best":    round(gap_at_best, 6),
        "best_epoch":     best_epoch,
        "total_epochs":   total_epochs_run,
        "elapsed_s":      round(elapsed, 1),
        "config":         cfg,
    }

    print(
        f"  best_val={best_val_total:.4f}  train@best={train_at_best:.4f}  "
        f"gap={gap_at_best:.4f}  epoch={best_epoch}/{total_epochs_run}  time={elapsed:.0f}s"
    )
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: Train conditional shared-space CVAE")
    p.add_argument("--data",          default="data/tcga_redo_mlomicZ.pkl")
    p.add_argument("--splits",        default="data/splits.json")
    p.add_argument("--primary_sites", default="data/primary_sites.json")
    p.add_argument("--cvae_dir",      default="cvae_phase1",           help="Dir with Phase-1 CVAE checkpoints")
    p.add_argument("--out",           default="cvae_phase2_results")
    p.add_argument("--device",        default=None)
    p.add_argument("--epochs",        type=int, default=150)
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--experiments",   nargs="*", default=None,
                   help="Subset of experiment names to run (default: all)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    with open(args.data, "rb") as f:
        multi_omic_data = pickle.load(f)

    ACTIVE_MODALITIES = ["rna", "methylation"]
    multi_omic_data = {k: v for k, v in multi_omic_data.items() if k in ACTIVE_MODALITIES}
    print(f"Modalities: {list(multi_omic_data.keys())}")

    if os.path.exists(args.splits):
        common_samples, train_idx, val_idx, test_idx = load_shared_splits_from_json(
            multi_omic_data, args.splits
        )
    else:
        common_samples, train_idx, val_idx, test_idx = compute_shared_splits(
            multi_omic_data, val_size=0.1, test_size=0.2, seed=42
        )
    print(f"Samples: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    condition_matrix, class_names = load_conditions_from_json(args.primary_sites, common_samples)
    num_classes = len(class_names)
    print(f"Primary sites: {num_classes} classes")

    name_map = {"rna": "rna", "methylation": "mth"}
    cvae_paths = {
        mod: os.path.join(args.cvae_dir, f"{name_map.get(mod, mod)}_cvae.pt")
        for mod in multi_omic_data
    }

    for mod, path in cvae_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CVAE checkpoint not found: {path}\n"
                f"Run train_cvae_autoencoders.py first."
            )

    to_run = args.experiments if args.experiments else list(EXPERIMENTS.keys())
    missing = [e for e in to_run if e not in EXPERIMENTS]
    if missing:
        print(f"Unknown experiments: {missing}. Available: {list(EXPERIMENTS.keys())}")
        return

    os.makedirs(args.out, exist_ok=True)
    all_results = []

    for exp_name in to_run:
        cfg = build_config(EXPERIMENTS[exp_name])
        overrides = {k: v for k, v in EXPERIMENTS[exp_name].items()}
        print(f"\n{'='*60}")
        print(f"Running: {exp_name}  overrides: {overrides if overrides else '(baseline)'}")

        try:
            result = run_one_experiment(
                exp_name=exp_name,
                cfg=cfg,
                multi_omic_data=multi_omic_data,
                common_samples=common_samples,
                condition_matrix=condition_matrix,
                train_idx=train_idx,
                val_idx=val_idx,
                cvae_paths=cvae_paths,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                out_root=args.out,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR in {exp_name}: {e}")
            all_results.append({"exp_name": exp_name, "error": str(e)})

    # ── Save summary ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out, f"cvae_experiments_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    valid = [r for r in all_results if "error" not in r]
    valid.sort(key=lambda r: r["best_val_total"])

    print(f"\n{'='*70}")
    print(f"{'Experiment':<25} {'best_val':>10} {'train@best':>11} {'gap':>8} {'ep':>5} {'total_ep':>9}")
    print("-" * 70)
    for r in valid:
        print(f"{r['exp_name']:<25} {r['best_val_total']:>10.4f} {r['train_at_best']:>11.4f} "
              f"{r['gap_at_best']:>8.4f} {r['best_epoch']:>5} {r['total_epochs']:>9}")

    print(f"\nResults saved → {json_path}")


if __name__ == "__main__":
    main()
