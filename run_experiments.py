"""
Hyperparameter sweep for the shared-space fine-tuning model.
Runs multiple configurations and logs results to results/experiments.csv + results/experiments.json

Usage:
    python run_experiments.py
    python run_experiments.py --data data/tcga_redo_mlomicZ.pkl --ae_dir aes_redo_z --epochs 150
    python run_experiments.py --experiments baseline small_shared_32 high_dropout_03
"""

import argparse
import csv
import json
import os
import pickle
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data_utils import load_shared_splits_from_json, compute_shared_splits
from src.data_utils import MultiOmicDataset, get_dataloader
from src.mae_masked import (
    MultiModalWithSharedSpace,
    load_modality_with_config,
    extract_encoder_decoder_from_pretrained,
    finetune_epoch,
    eval_finetune_epoch,
)
from src.shared_finetune import save_shared_model


# ─── Experiment configurations ─────────────────────────────────────────────────

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
    freeze_encoders_decoders=False,
    two_path=False,
    gaussian_noise_std=0.0,
    grad_clip=1.0,
)

EXPERIMENTS = {
    "baseline": {},

    # 1. Smaller shared dimension
    "small_shared_64": dict(shared_dim=64),
    "small_shared_32": dict(shared_dim=32),
    "small_shared_16": dict(shared_dim=16),

    # 2. Higher dropout in projection head
    "high_dropout_03": dict(proj_activation_dropout=0.3),
    "high_dropout_05": dict(proj_activation_dropout=0.5),

    # 3. More feature masking
    "high_mask_04": dict(feature_mask_p_train=0.4, feature_mask_p_val=0.4),
    "high_mask_05": dict(feature_mask_p_train=0.5, feature_mask_p_val=0.5),

    # 4. Higher modality dropout
    "high_mod_dropout_05": dict(modality_dropout_prob=0.5),

    # 5. Lower learning rate
    "low_lr_1e4": dict(lr=1e-4),
    "low_lr_5e5": dict(lr=5e-5),

    # 6. Higher weight decay
    "high_wd_5e4": dict(weight_decay=5e-4),
    "high_wd_1e3": dict(weight_decay=1e-3),

    # 6b. Very high weight decay
    "high_wd_1e2": dict(weight_decay=1e-2),

    # 7. Gaussian noise regularization
    "gaussian_noise_001": dict(gaussian_noise_std=0.01),

    # 8. 2-stage training (handled specially – mark with special key)
    "two_stage": dict(_two_stage=True),

    # 8b. Freeze pretrained encoders/decoders for full training
    "freeze_encoders": dict(freeze_encoders_decoders=True),

    # 9. Combined best (aggressive regularization)
    "combined_heavy": dict(
        shared_dim=32,
        proj_activation_dropout=0.3,
        feature_mask_p_train=0.4,
        feature_mask_p_val=0.4,
        modality_dropout_prob=0.5,
        lr=1e-4,
        weight_decay=1e-3,
        gaussian_noise_std=0.01,
    ),

    # 10. Lower contrastive weight (for cases where contrast_loss overfits)
    "low_contrast_05": dict(lambda_contrast=0.5),
    "low_contrast_02": dict(lambda_contrast=0.2),
}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def build_config(overrides: dict) -> dict:
    cfg = deepcopy(BASE_CONFIG)
    cfg.update({k: v for k, v in overrides.items() if not k.startswith("_")})
    return cfg


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


def run_one_experiment(
    exp_name: str,
    cfg: dict,
    multi_omic_data: dict,
    common_samples, train_idx, val_idx, test_idx,
    model_paths: dict,
    device: torch.device,
    epochs: int,
    batch_size: int,
    out_root: str,
    two_stage: bool = False,
    two_stage_freeze_epochs: int = 30,
):
    exp_dir = os.path.join(out_root, f"exp_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)

    # ── Load pretrained AEs ─────────────────────────────────────────────────
    encoders, decoders, hidden_dims, mask_values = {}, {}, {}, {}
    for mod, path in model_paths.items():
        ae_m, hidden_dim_m, cfg_m = load_modality_with_config(path, map_location=device)
        ae_m = ae_m.to(device)
        enc, dec = extract_encoder_decoder_from_pretrained(ae_m)
        encoders[mod] = enc
        decoders[mod] = dec
        hidden_dims[mod] = hidden_dim_m
        mask_values[mod] = cfg_m.get("mask_value", 0.0)

    # ── Build dataloaders ────────────────────────────────────────────────────
    multi_ds = MultiOmicDataset({m: df.loc[common_samples] for m, df in multi_omic_data.items()})
    train_loader = get_dataloader(multi_ds, batch_size=batch_size, shuffle=True,  split_idx=train_idx)
    val_loader   = get_dataloader(multi_ds, batch_size=batch_size, shuffle=False, split_idx=val_idx)

    # ── Build model ──────────────────────────────────────────────────────────
    model = MultiModalWithSharedSpace(
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

    def _train_phase(model, lr, weight_decay, n_epochs, freeze_enc_dec):
        nonlocal best_val_total, best_model_state, best_epoch, train_at_best, epochs_no_improve

        if freeze_enc_dec:
            for p in model.encoders.parameters(): p.requires_grad = False
            for p in model.decoders.parameters(): p.requires_grad = False
            model.encoders.eval()
            model.decoders.eval()
        else:
            for p in model.encoders.parameters(): p.requires_grad = True
            for p in model.decoders.parameters(): p.requires_grad = True
            model.encoders.train()
            model.decoders.train()

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=5, factor=0.5
        )

        for ep in range(n_epochs):
            tr = finetune_epoch(
                model=model, dataloader=train_loader, optimizer=opt, device=device,
                mask_values=mask_values,
                lambda_contrastive=cfg["lambda_contrast"],
                lambda_impute=cfg["lambda_impute"],
                modality_dropout_prob=cfg["modality_dropout_prob"],
                feature_mask_p=cfg["feature_mask_p_train"],
                alpha_mask_recon=cfg["alpha_mask_recon"],
                two_path_clean_for_contrast=cfg["two_path"],
                grad_clip=cfg["grad_clip"],
                gaussian_noise_std=cfg["gaussian_noise_std"],
            )
            vl = eval_finetune_epoch(
                model=model, dataloader=val_loader, device=device,
                mask_values=mask_values,
                lambda_contrastive=cfg["lambda_contrast"],
                lambda_impute=cfg["lambda_impute"],
                feature_mask_p=cfg["feature_mask_p_val"],
                alpha_mask_recon=cfg["alpha_mask_recon"],
                two_path_clean_for_contrast=cfg["two_path"],
            )

            for k_src, k_dst in [("total_loss","total"),("recon_loss","recon"),
                                  ("contrast_loss","contrast"),("impute_loss","impute")]:
                train_hist[k_dst].append(tr[k_src])
                val_hist[k_dst].append(vl[k_src])

            scheduler.step(vl["total_loss"])

            if vl["total_loss"] < best_val_total:
                best_val_total = vl["total_loss"]
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = len(train_hist["total"])
                train_at_best = tr["total_loss"]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"  [Early stop] ep {len(train_hist['total'])} | no improvement for {PATIENCE} epochs")
                break

        return opt

    # ── Run training ─────────────────────────────────────────────────────────
    t0 = time.time()
    if two_stage:
        print(f"  Stage 1: freeze encoders/decoders for {two_stage_freeze_epochs} epochs (lr=3e-4)")
        _train_phase(model, lr=3e-4, weight_decay=cfg["weight_decay"],
                     n_epochs=two_stage_freeze_epochs, freeze_enc_dec=True)
        remaining = max(epochs - two_stage_freeze_epochs, 0)
        print(f"  Stage 2: unfreeze all, {remaining} epochs (lr=1e-4)")
        _train_phase(model, lr=1e-4, weight_decay=cfg["weight_decay"],
                     n_epochs=remaining, freeze_enc_dec=False)
    else:
        _train_phase(model, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                     n_epochs=epochs, freeze_enc_dec=cfg["freeze_encoders_decoders"])

    elapsed = time.time() - t0

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    total_epochs_run = len(train_hist["total"])
    final_train = train_hist["total"][-1] if train_hist["total"] else float("nan")
    final_val   = val_hist["total"][-1]   if val_hist["total"]   else float("nan")
    gap_at_best = best_val_total - train_at_best

    # ── Save artifacts ───────────────────────────────────────────────────────
    hist_path = os.path.join(exp_dir, "loss_history.json")
    with open(hist_path, "w") as f:
        json.dump({"train": train_hist, "val": val_hist}, f)

    plot_loss_curves(train_hist, val_hist,
                     save_path=os.path.join(exp_dir, "loss_curves.png"),
                     title=exp_name)

    result = {
        "exp_name":        exp_name,
        "best_val_total":  round(best_val_total, 6),
        "train_at_best":   round(train_at_best, 6),
        "gap_at_best":     round(gap_at_best, 6),
        "final_val_total": round(final_val, 6),
        "final_train":     round(final_train, 6),
        "best_epoch":      best_epoch,
        "total_epochs":    total_epochs_run,
        "elapsed_s":       round(elapsed, 1),
        "config":          cfg,
    }

    print(f"  best_val={best_val_total:.4f}  train@best={train_at_best:.4f}  "
          f"gap={gap_at_best:.4f}  epoch={best_epoch}/{total_epochs_run}  "
          f"time={elapsed:.0f}s")

    return result


# ─── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/tcga_redo_mlomicZ.pkl")
    p.add_argument("--splits",  default="data/splits.json")
    p.add_argument("--ae_dir",  default="aes_redo_z")
    p.add_argument("--out",     default="results")
    p.add_argument("--device",  default=None)
    p.add_argument("--epochs",  type=int, default=150)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--two_stage_freeze_epochs", type=int, default=30)
    p.add_argument("--experiments", nargs="*", default=None,
                   help="Subset of experiment names to run (default: all)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Load data
    with open(args.data, "rb") as f:
        multi_omic_data = pickle.load(f)

    ACTIVE_MODALITIES = ["rna", "methylation"]
    multi_omic_data = {k: v for k, v in multi_omic_data.items() if k in ACTIVE_MODALITIES}
    print(f"Modalities: {list(multi_omic_data.keys())}")

    # Splits
    if os.path.exists(args.splits):
        from src.data_utils import load_shared_splits_from_json
        common_samples, train_idx, val_idx, test_idx = load_shared_splits_from_json(
            multi_omic_data, args.splits
        )
    else:
        from src.data_utils import compute_shared_splits
        common_samples, train_idx, val_idx, test_idx = compute_shared_splits(
            multi_omic_data, val_size=0.1, test_size=0.2, seed=42
        )
    print(f"Samples: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # AE paths
    name_map = {"rna": "rna", "methylation": "mth"}
    model_paths = {mod: os.path.join(args.ae_dir, f"{name_map.get(mod, mod)}_ae.pt")
                   for mod in multi_omic_data.keys()}

    # Select experiments
    to_run = args.experiments if args.experiments else list(EXPERIMENTS.keys())
    missing = [e for e in to_run if e not in EXPERIMENTS]
    if missing:
        print(f"Unknown experiments: {missing}. Available: {list(EXPERIMENTS.keys())}")
        return

    os.makedirs(args.out, exist_ok=True)

    # Ensure CUDA-safe multiprocessing (use 'spawn' instead of default 'fork')
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method was already set; ignore
        pass

    all_results = []

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
        futures = {}

        for exp_name in to_run:
            overrides = EXPERIMENTS[exp_name]
            two_stage = overrides.get("_two_stage", False)
            cfg = build_config(overrides)

            print(f"\n{'='*60}")
            print(f"Running: {exp_name}")
            cfg_display = {k: v for k, v in cfg.items() if v != BASE_CONFIG.get(k)}
            print(f"  Overrides: {cfg_display if cfg_display else '(baseline)'}")

            future = executor.submit(
                run_one_experiment,
                exp_name,
                cfg,
                multi_omic_data,
                common_samples,
                train_idx,
                val_idx,
                test_idx,
                model_paths,
                device,
                args.epochs,
                args.batch_size,
                args.out,
                two_stage,
                args.two_stage_freeze_epochs,
            )
            futures[future] = exp_name

        for future in as_completed(futures):
            exp_name = futures[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR in {exp_name}: {e}")
                all_results.append({"exp_name": exp_name, "error": str(e)})

    # ── Save summary ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out, f"experiments_{timestamp}.json")
    csv_path  = os.path.join(args.out, f"experiments_{timestamp}.csv")

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV (flat, skip config column)
    csv_fields = ["exp_name", "best_val_total", "train_at_best", "gap_at_best",
                  "final_val_total", "final_train", "best_epoch", "total_epochs",
                  "elapsed_s", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in csv_fields})

    # Print sorted summary table
    valid = [r for r in all_results if "error" not in r]
    valid.sort(key=lambda r: r["best_val_total"])

    print(f"\n{'='*70}")
    print(f"{'Experiment':<25} {'best_val':>10} {'train@best':>11} {'gap':>8} {'ep':>5} {'total_ep':>9}")
    print("-" * 70)
    for r in valid:
        print(f"{r['exp_name']:<25} {r['best_val_total']:>10.4f} {r['train_at_best']:>11.4f} "
              f"{r['gap_at_best']:>8.4f} {r['best_epoch']:>5} {r['total_epochs']:>9}")

    print(f"\nResults saved → {csv_path}")
    print(f"Full JSON     → {json_path}")


if __name__ == "__main__":
    main()
