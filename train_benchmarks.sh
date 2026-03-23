#!/bin/bash
# train_benchmarks.sh
# Automates training of 6 model variants (AE, VAE, CVAE) x 2 (Baseline, Modified loss)
# for 10 epochs to extract comparison metrics.

set -e

ENV_PYTHON=".venv/bin/python"
EPOCHS=100
SHARED_DIM=128

echo "==========================================="
echo "   STARTING MULTI-VARIANT BENCHMARKING     "
echo "==========================================="

# Helper to swap hard negative mining in the loss
set_hard_negative() {
    weight=$1
    echo ">> Setting hard_negative_weight to $weight in src/mae_masked.py"
    # Using python to precisely replace the argument
    $ENV_PYTHON -c "
import re, sys
weight = sys.argv[1]
with open('src/mae_masked.py', 'r') as f:
    code = f.read()
code = re.sub(r'hard_negative_weight:\s*float\s*=\s*[0-9\.]+', f'hard_negative_weight: float={weight}', code)
with open('src/mae_masked.py', 'w') as f:
    f.write(code)
" "$weight"
}

# 1. BASELINE TRAINING
set_hard_negative 0.0

echo "[1/6] Training Baseline Shared AE..."
$ENV_PYTHON train_shared.py --epochs $EPOCHS --out checkpoints/baseline_ae --shared_dim $SHARED_DIM

echo "[2/6] Training Baseline Shared VAE..."
$ENV_PYTHON train_shared_vae.py --epochs $EPOCHS --out checkpoints/baseline_vae --shared_dim $SHARED_DIM

echo "[3/6] Training Baseline Shared CVAE..."
$ENV_PYTHON train_cvae_shared.py --epochs $EPOCHS --out checkpoints/baseline_cvae --experiments baseline

# 2. MODIFIED TRAINING
set_hard_negative 1.0

echo "[4/6] Training Modified Shared AE..."
$ENV_PYTHON train_shared.py --epochs $EPOCHS --out checkpoints/modified_ae --shared_dim $SHARED_DIM

echo "[5/6] Training Modified Shared VAE..."
$ENV_PYTHON train_shared_vae.py --epochs $EPOCHS --out checkpoints/modified_vae --shared_dim $SHARED_DIM

echo "[6/6] Training Modified Shared CVAE..."
$ENV_PYTHON train_cvae_shared.py --epochs $EPOCHS --out checkpoints/modified_cvae --experiments baseline

echo "Training completed for all 6 variants."
