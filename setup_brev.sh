#!/bin/bash
# setup_brev.sh – clone MIMIR, download data, install deps, run Phase 1 training
# Tested on NVIDIA Brev (Ubuntu 22.04 + CUDA 12.x)
set -euo pipefail

REPO_URL="git@github.com:marcin119a/MIMIR.git"
REPO_DIR="MIMIR"

# Google Drive folder IDs (from README)
DATA_FOLDER_ID="1340tEG3_bL9ojHJ8hQmMkBoZ9dSKYUhV"
DATA_FILE_NAME="tcga_redo_mlomicZ.pkl"
SPLITS_FILE_NAME="splits.json"

# ── 1. Clone repo ─────────────────────────────────────────────────────────────
echo "==> Cloning repository..."
if [ -d "$REPO_DIR" ]; then
    echo "    Directory '$REPO_DIR' already exists – pulling latest..."
    git -C "$REPO_DIR" pull
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── 2. Python environment ─────────────────────────────────────────────────────
echo "==> Setting up Python virtual environment..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip --quiet

# ── 3. Install PyTorch with CUDA support ──────────────────────────────────────
echo "==> Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

# ── 4. Install remaining requirements ─────────────────────────────────────────
echo "==> Installing project requirements..."
pip install \
    numpy pandas scikit-learn matplotlib seaborn tqdm \
    fancyimpute gdown \
    --quiet

# ── 5. Verify GPU is visible ──────────────────────────────────────────────────
echo "==> Checking GPU availability..."
python - <<'PYEOF'
import torch
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    CUDA: {torch.version.cuda}")
else:
    print("    WARNING: CUDA not available – training will run on CPU")
PYEOF

# ── 6. Download data ──────────────────────────────────────────────────────────
echo "==> Downloading data from Google Drive..."
mkdir -p data

# Download entire folder (gdown >= 4.6 supports --folder)
if [ ! -f "data/$DATA_FILE_NAME" ] || [ ! -f "data/$SPLITS_FILE_NAME" ]; then
    gdown --folder "https://drive.google.com/drive/folders/$DATA_FOLDER_ID" \
          --output data/ --remaining-ok --quiet
    echo "    Data downloaded to data/"
else
    echo "    Data files already present – skipping download."
fi

# ── 7. Run Phase 1 training ───────────────────────────────────────────────────
echo "==> Starting Phase 1 autoencoder training..."
python train_autoencoders.py \
    --data   "data/$DATA_FILE_NAME" \
    --splits "data/$SPLITS_FILE_NAME" \
    --out    aes_redo_z

echo ""
echo "==> Done! Checkpoints saved to aes_redo_z/"


# Run Phase 2 training
echo "==> Starting Phase 2 training..."
python train_shared.py \
    --data   "data/$DATA_FILE_NAME" \
    --splits "data/$SPLITS_FILE_NAME" \
    --out    checkpoints/finetuned

echo ""
echo "==> Done! Checkpoints saved to checkpoints/finetuned/"