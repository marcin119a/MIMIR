
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_hist(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["val"]

ae_path   = "checkpoints/baseline_ae/loss_history_ep100.json"
vae_path  = "checkpoints/baseline_vae/loss_history_ep100.json"
cvae_path = "checkpoints/baseline_cvae/exp_baseline/loss_history.json"

hists = {}
if os.path.exists(ae_path):
    hists["Mimir-AE (Original Baseline)"] = load_hist(ae_path)
if os.path.exists(vae_path):
    hists["Mimir-VAE"] = load_hist(vae_path)
if os.path.exists(cvae_path):
    hists["Mimir-CVAE (Marcin)"] = load_hist(cvae_path)

if hists:
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    metrics = ["total", "recon", "contrast", "impute"]
    titles = ["Total Loss", "Reconstruction MSE", "Contrastive Loss", "Imputation MSE"]
    
    # Common epochs count
    min_ep = min(len(h["total"]) for h in hists.values())
    
    for i, m in enumerate(metrics):
        for label, h in hists.items():
            axes[i].plot(h[m][:min_ep], label=label)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Epoch")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("ae_vae_cvae_comparison.png", dpi=150)
    print("Comprehensive comparison plot saved to ae_vae_cvae_comparison.png")
    
    print("\n--- Final Metrics Summary (at epoch 100) ---")
    header = f"{'Metric':12s} | {'Mimir-AE':>10s} | {'Mimir-VAE':>10s} | {'Mimir-CVAE':>10s}"
    print(header)
    print("-" * len(header))
    
    for m in metrics:
        v_ae = hists.get("Mimir-AE (Original Baseline)", {}).get(m, [float("nan")])[-1]
        v_vae = hists.get("Mimir-VAE", {}).get(m, [float("nan")])[-1]
        v_cvae = hists.get("Mimir-CVAE (Marcin)", {}).get(m, [float("nan")])[-1]
        print(f"{m:12s} | {v_ae:10.4f} | {v_vae:10.4f} | {v_cvae:10.4f}")

else:
    print("No histories found to compare.")

# Additional Checks
with open("src/mae_masked.py", "r") as f: mae_code = f.read()
with open("src/cvae.py", "r") as f: cvae_code = f.read()

print("\n--- Structural Findings ---")
print(f"AE/VAE Architecture: Uses LayerNorm ({'nn.LayerNorm' in mae_code})")
print(f"CVAE Architecture:   Uses LayerNorm ({'nn.LayerNorm' in cvae_code})")
if "self._last_kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()" in cvae_code:
    print("CVAE KL Penalty:     Weak (scaled by 1/D via .mean())")
else:
    print("CVAE KL Penalty:     Standard")
