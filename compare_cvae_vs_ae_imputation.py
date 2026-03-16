"""
compare_cvae_vs_ae_imputation.py
=================================
Porównuje imputację modalności brakujących dla dwóch modeli:
  - AE      : vanilla autoencoder (Phase 1) + dzielona przestrzeń (Phase 2)
  - CVAE    : warunkowy autoencoder (Phase 1, conditioned on primary site) + Phase 2

Oba modele ładowane z gotowych checkpointów, imputacja na zbiorze testowym (LOO).
Wynik: tabela w konsoli + compare_cvae_vs_ae_imputation.png

Ścieżki domyślne:
  Vanilla AE Phase 1:   aes_redo_z/{rna,mth}_ae.pt
  Vanilla AE Phase 2:   checkpoints/finetuned/shared_model_ep200.pt
  CVAE Phase 1:         cvae_phase1/{rna,mth}_cvae.pt
  CVAE Phase 2:         cvae_phase2_results/exp_baseline/model_best.pt
  Dane:                 data/tcga_redo_mlomicZ.pkl
  Splity:               data/splits.json
  Primary sites:        data/primary_sites.json

Usage:
    python compare_cvae_vs_ae_imputation.py
    python compare_cvae_vs_ae_imputation.py --device cuda --batch_size 256
"""

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_utils import (
    MultiOmicDataset,
    compute_shared_splits,
    get_dataloader,
    load_shared_splits_from_json,
)
from src.evaluation import evaluate_imputations
from src.mae_masked import (
    MultiModalWithSharedSpace,
    extract_encoder_decoder_from_pretrained,
    load_modality_with_config,
)
from src.cvae import (
    ConditionalMultiOmicDataset,
    extract_encoder_decoder_from_cvae,
    get_conditional_dataloader,
    load_conditions_from_json,
    load_cvae_with_config,
)
from src.cvae_phase2 import ConditionalMultiModalWithSharedSpace
from src.shared_finetune import load_shared_model


# ─── Helpers ──────────────────────────────────────────────────────────────────

MODALITY_DISPLAY = {"rna": "mRNA", "methylation": "Methylation"}
MODEL_COLORS = {"AE": "tab:blue", "CVAE": "tab:orange"}


def impute_vanilla(
    model: MultiModalWithSharedSpace,
    multi_omic_data: Dict[str, pd.DataFrame],
    test_samples: List[str],
    mask_value: float,
    batch_size: int,
    device: torch.device,
) -> Dict[Tuple, pd.DataFrame]:
    """LOO imputation with vanilla MultiModalWithSharedSpace."""
    mask_values = {m: mask_value for m in model.modalities}
    all_modalities = list(model.modalities)
    predictions = {}

    model.eval()
    for target_mod in all_modalities:
        present_mods = [m for m in all_modalities if m != target_mod]
        data_present = {m: multi_omic_data[m].loc[test_samples] for m in present_mods}

        ds = MultiOmicDataset(data_present)
        loader = get_dataloader(ds, batch_size=batch_size, shuffle=False)
        samples_used = ds.common_samples

        chunks = []
        with torch.no_grad():
            for batch in loader:
                batch = {m: x.to(device) for m, x in batch.items()}
                batch_clean = {}
                for mod, xb in batch.items():
                    xb_c = xb.clone()
                    xb_c[torch.isnan(xb_c)] = mask_values.get(mod, 0.0)
                    batch_clean[mod] = xb_c

                shared = {}
                for mod, xb_c in batch_clean.items():
                    h = model.encoders[mod](xb_c)
                    z = model.projections[mod](h)
                    shared[mod] = z

                z_mean = torch.stack([shared[m] for m in present_mods], 0).mean(0)
                h_hat = model.rev_projections[target_mod](z_mean)
                x_imp = model.decoders[target_mod](h_hat)
                chunks.append(x_imp.cpu())

        X_imp = torch.cat(chunks, 0).numpy()
        df_imp = pd.DataFrame(
            X_imp,
            index=samples_used,
            columns=multi_omic_data[target_mod].columns,
        )
        key = (tuple(sorted(present_mods)), target_mod)
        predictions[key] = df_imp

    return predictions


def impute_cvae(
    model: ConditionalMultiModalWithSharedSpace,
    multi_omic_data: Dict[str, pd.DataFrame],
    test_samples: List[str],
    condition_matrix_full: np.ndarray,
    all_samples: List[str],
    mask_value: float,
    batch_size: int,
    device: torch.device,
) -> Dict[Tuple, pd.DataFrame]:
    """LOO imputation with ConditionalMultiModalWithSharedSpace."""
    # Build a mapping sample → condition row
    sample_to_idx = {s: i for i, s in enumerate(all_samples)}
    test_cond = torch.tensor(
        condition_matrix_full[[sample_to_idx[s] for s in test_samples]],
        dtype=torch.float32,
    )

    all_modalities = list(model.modalities)
    predictions = {}

    model.eval()
    for target_mod in all_modalities:
        present_mods = [m for m in all_modalities if m != target_mod]
        data_present = {m: multi_omic_data[m].loc[test_samples] for m in present_mods}

        ds = MultiOmicDataset(data_present)
        loader = get_dataloader(ds, batch_size=batch_size, shuffle=False)
        samples_used = ds.common_samples

        # Rebuild sample → condition mapping for this (potentially reordered) dataset
        cond_rows = torch.tensor(
            condition_matrix_full[[sample_to_idx[s] for s in samples_used]],
            dtype=torch.float32,
        )

        chunks = []
        offset = 0
        with torch.no_grad():
            for batch in loader:
                bsz = next(iter(batch.values())).shape[0]
                cb = cond_rows[offset: offset + bsz].to(device)
                offset += bsz

                batch = {m: x.to(device) for m, x in batch.items()}
                batch_clean = {}
                for mod, xb in batch.items():
                    xb_c = xb.clone()
                    xb_c[torch.isnan(xb_c)] = mask_value
                    batch_clean[mod] = xb_c

                shared = {}
                for mod, xb_c in batch_clean.items():
                    h = model.encoders[mod](xb_c, cb)
                    z = model.projections[mod](h, cb)
                    shared[mod] = z

                z_mean = torch.stack([shared[m] for m in present_mods], 0).mean(0)
                h_hat = model.rev_projections[target_mod](z_mean, cb)
                x_imp = model.decoders[target_mod](h_hat, cb)
                chunks.append(x_imp.cpu())

        X_imp = torch.cat(chunks, 0).numpy()
        df_imp = pd.DataFrame(
            X_imp,
            index=samples_used,
            columns=multi_omic_data[target_mod].columns,
        )
        key = (tuple(sorted(present_mods)), target_mod)
        predictions[key] = df_imp

    return predictions


def print_comparison_table(ae_metrics, cvae_metrics):
    header = f"{'Target':14s} {'Metric':8s} {'AE':>10s} {'CVAE':>10s} {'Δ (CVAE-AE)':>13s}"
    print("\n" + "="*60)
    print(header)
    print("-"*60)
    for key in sorted(ae_metrics.keys()):
        present_mods, target = key
        target_disp = MODALITY_DISPLAY.get(target, target)
        ae_m   = ae_metrics[key]
        cvae_m = cvae_metrics.get(key, {})
        for metric in ("pearson", "spearman", "mse"):
            v_ae   = ae_m.get(metric, float("nan"))
            v_cvae = cvae_m.get(metric, float("nan"))
            delta  = v_cvae - v_ae
            arrow  = "↑" if delta > 0 else "↓"
            # For MSE lower is better: flip arrow
            if metric == "mse":
                arrow = "↓" if delta < 0 else "↑"
            print(f"{target_disp:14s} {metric:8s} {v_ae:10.4f} {v_cvae:10.4f}"
                  f"  {arrow}{abs(delta):>10.4f}")
        print()


def plot_comparison(ae_metrics, cvae_metrics, save_path):
    targets = sorted({key[1] for key in ae_metrics})
    metrics = ["pearson", "spearman", "mse"]
    metric_labels = ["Pearson r", "Spearman ρ", "MSE"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.5))

    for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[col]
        x = np.arange(len(targets))
        w = 0.35

        ae_vals   = []
        cvae_vals = []
        for t in targets:
            key = next(k for k in ae_metrics if k[1] == t)
            ae_vals.append(ae_metrics[key].get(metric, float("nan")))
            cvae_vals.append(cvae_metrics.get(key, {}).get(metric, float("nan")))

        bars_ae   = ax.bar(x - w/2, ae_vals,   w, label="AE",   color=MODEL_COLORS["AE"])
        bars_cvae = ax.bar(x + w/2, cvae_vals, w, label="CVAE", color=MODEL_COLORS["CVAE"])

        ymax = max(max(v for v in ae_vals + cvae_vals if not np.isnan(v)), 0)
        offset = 0.01 * (ymax or 1)
        for bar, val in zip(list(bars_ae) + list(bars_cvae),
                            ae_vals + cvae_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([MODALITY_DISPLAY.get(t, t) for t in targets], rotation=0)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("CVAE vs Vanilla AE — LOO Imputation (test set)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {save_path}")


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",           default="data/tcga_redo_mlomicZ.pkl")
    p.add_argument("--splits",         default="data/splits.json")
    p.add_argument("--primary_sites",  default="data/primary_sites.json")
    p.add_argument("--ae_rna",         default="aes_redo_z/rna_ae.pt")
    p.add_argument("--ae_mth",         default="aes_redo_z/mth_ae.pt")
    p.add_argument("--ae_phase2",      default="checkpoints/finetuned/shared_model_ep200.pt")
    p.add_argument("--ae_shared_dim",  type=int, default=256)
    p.add_argument("--cvae_rna",       default="cvae_phase1/rna_cvae.pt")
    p.add_argument("--cvae_mth",       default="cvae_phase1/mth_cvae.pt")
    p.add_argument("--cvae_phase2",    default="cvae_phase2_results/exp_baseline/model_best.pt")
    p.add_argument("--cvae_shared_dim",type=int, default=128)
    p.add_argument("--device",         default=None)
    p.add_argument("--batch_size",     type=int, default=128)
    p.add_argument("--out",            default="compare_cvae_vs_ae_imputation.png")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\nLoading data …")
    with open(args.data, "rb") as f:
        multi_omic_data = pickle.load(f)
    ACTIVE = ["rna", "methylation"]
    multi_omic_data = {k: v for k, v in multi_omic_data.items() if k in ACTIVE}
    print(f"Modalities: {list(multi_omic_data.keys())}")

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

    test_samples = [common_samples[i] for i in test_idx]
    print(f"Test samples: {len(test_samples)}")

    # ── Vanilla AE model ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Loading Vanilla AE model …")
    print(f"{'='*55}")

    ae_paths = {"rna": args.ae_rna, "methylation": args.ae_mth}
    ae_encoders, ae_decoders, ae_hidden_dims = {}, {}, {}
    for mod, path in ae_paths.items():
        ae_m, hdim, cfg = load_modality_with_config(path, map_location=device)
        ae_m = ae_m.to(device)
        enc, dec = extract_encoder_decoder_from_pretrained(ae_m)
        ae_encoders[mod] = enc
        ae_decoders[mod] = dec
        ae_hidden_dims[mod] = hdim
        print(f"  AE {mod}: input_dim={cfg['input_dim']} hidden={cfg['hidden_layers']} latent={hdim}")

    ae_model = load_shared_model(
        MultiModalWithSharedSpace,
        ae_encoders, ae_decoders, ae_hidden_dims,
        shared_dim=args.ae_shared_dim,
        proj_depth=1,
        checkpoint_path=args.ae_phase2,
        map_location=device,
    )
    ae_model = ae_model.to(device)

    # ── CVAE model ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Loading CVAE model …")
    print(f"{'='*55}")

    condition_matrix, class_names = load_conditions_from_json(
        args.primary_sites, common_samples
    )
    num_classes = len(class_names)
    print(f"  Condition classes: {num_classes}")

    cvae_paths = {"rna": args.cvae_rna, "methylation": args.cvae_mth}
    cvae_encoders, cvae_decoders, cvae_hidden_dims = {}, {}, {}
    for mod, path in cvae_paths.items():
        cvae_m, hdim, cfg = load_cvae_with_config(path, map_location=device)
        cvae_m = cvae_m.to(device)
        enc, dec = extract_encoder_decoder_from_cvae(cvae_m)
        cvae_encoders[mod] = enc
        cvae_decoders[mod] = dec
        cvae_hidden_dims[mod] = hdim
        print(f"  CVAE {mod}: input_dim={cfg['input_dim']} hidden={cfg['hidden_layers']} latent={hdim} classes={cfg['num_classes']}")

    cvae_model = ConditionalMultiModalWithSharedSpace(
        encoders=cvae_encoders,
        decoders=cvae_decoders,
        hidden_dims=cvae_hidden_dims,
        shared_dim=args.cvae_shared_dim,
        num_classes=num_classes,
        proj_depth=1,
    )
    cvae_state = torch.load(args.cvae_phase2, map_location=device, weights_only=False)
    cvae_model.load_state_dict(cvae_state)
    cvae_model = cvae_model.to(device)
    cvae_model.eval()
    print(f"  [Loaded] CVAE phase 2 from {args.cvae_phase2}")

    # ── LOO Imputation ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Running AE LOO imputation on {len(test_samples)} test samples …")
    print(f"{'='*55}")
    ae_preds = impute_vanilla(
        ae_model, multi_omic_data, test_samples,
        mask_value=0.0, batch_size=args.batch_size, device=device,
    )

    print(f"\n{'='*55}")
    print(f"  Running CVAE LOO imputation on {len(test_samples)} test samples …")
    print(f"{'='*55}")
    cvae_preds = impute_cvae(
        cvae_model, multi_omic_data, test_samples,
        condition_matrix_full=condition_matrix,
        all_samples=common_samples,
        mask_value=0.0,
        batch_size=args.batch_size,
        device=device,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Evaluating …")
    print(f"{'='*55}")

    # Filter multi_omic_data to test samples for evaluation
    test_data = {m: df.loc[df.index.intersection(test_samples)] for m, df in multi_omic_data.items()}

    ae_metrics   = evaluate_imputations(ae_preds,   test_data, plot_scatter=False)
    cvae_metrics = evaluate_imputations(cvae_preds, test_data, plot_scatter=False)

    # ── Print results ─────────────────────────────────────────────────────────
    print_comparison_table(ae_metrics, cvae_metrics)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(ae_metrics, cvae_metrics, args.out)

    # ── CSV summary ───────────────────────────────────────────────────────────
    rows = []
    for key in sorted(ae_metrics.keys()):
        present_mods, target = key
        ae_m   = ae_metrics[key]
        cvae_m = cvae_metrics.get(key, {})
        for metric in ("pearson", "spearman", "mse"):
            rows.append({
                "target": target,
                "present_mods": ", ".join(present_mods),
                "metric": metric,
                "AE": ae_m.get(metric),
                "CVAE": cvae_m.get(metric),
                "delta_cvae_minus_ae": cvae_m.get(metric, float("nan")) - ae_m.get(metric, float("nan")),
            })
    csv_path = args.out.replace(".png", ".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
