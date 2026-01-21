# MIMIR: Multi-omic Imputation through Modality Integration and Representation learning

MIMIR is a framework for imputing missing modalities and missing values in multi-omic datasets using a shared latent representation.
The method combines modality-specific denoising autoencoders with a jointly learned shared space that enables robust cross-modal reconstruction under arbitrary missingness patterns.

This repository contains:
- Reusable PyTorch-based model and imputation code
- A sequence of notebooks implementing the full experimental pipeline
- Benchmarking code against classical and latent-factor baselines

---

## Repository structure

```text
MIMIR/
├── src/                         # Reusable, experiment-agnostic code
│   ├── data_utils.py            # Data loading, alignment, masking utilities
│   ├── mae_masked.py            # Denoising autoencoders (Phase 1)
│   ├── shared_finetune.py       # Shared latent space training (Phase 2)
│   ├── translation.py           # Cross-modality translation utilities
│   ├── impute1.py               # Missing-value masking and imputation helpers
│   ├── evaluation.py            # Metrics, aggregation, plotting utilities
│   ├── others/                  # Baseline methods
│   │   ├── knn_imp.py           # KNN-based imputation
│   │   ├── softimpv2.py         # SoftImpute (low-rank matrix completion)
│   │   ├── mofa_imputer.py      # MOFA+ baseline
│   │   └── tobmi.py             # TOBMI-style translation baseline
│   └── __init__.py
│
├── 1_Phase1_Train_Autoencoders.ipynb
├── 2_Phase2_Train_MAE.ipynb
├── 3_Imputation_Missing_Modality.ipynb
├── 4_Benchmark_Missing_Modalities.ipynb
├── 5_Imputation_Missing_Values.ipynb
└── 6_Benchmark_Missing_Values.ipynb
````
---
## Requirements

The codebase is implemented in Python and primarily uses PyTorch.

### Core dependencies

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm

### Optional dependencies (baseline methods only)

- `fancyimpute` (SoftImpute, KNN baselines)
- `mofapy2` (MOFA+ baseline)
- `mofax` (MOFA+ model handling and I/O)

### Installation

We recommend using a virtual environment or conda environment:

```bash
conda create -n mimir python=3.10
conda activate mimir
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm
````
---

## Trained models

To avoid retraining, we provide trained checkpoints for both the shared MIMIR model
and the modality-specific autoencoders.

### Phase 1: Modality-specific autoencoders

Pretrained denoising autoencoders for each modality (trained in Phase 1):

- **CNV autoencoder**
- **mRNA autoencoder**
- **miRNA autoencoder**
- **DNA methylation autoencoder**

Download (all modalities):  
https://<link-to-aes_redo_z>

Expected location after download:
``` aes_redo_z/```

These checkpoints are loaded automatically by the Phase 2 and Phase 3–6 notebooks.

### Phase 2: Shared latent space model

- **Shared MIMIR model checkpoint**  
  Trained using Phase 1 autoencoders and Phase 2 joint fine-tuning.

  Download:  
  https://<link-to-phase2-checkpoint>

  Expected location after download:
  ``` checkpoints/finetuned/```
---

## Data availability

For reproducibility, we provide the processed multi-omic data and fixed train/validation/test splits
used in all experiments.

### Processed data

- **Multi-omic data pickle**  
  Contains aligned, z-scored data for all modalities.

  Download:  
  https://<[link-to-data-pickle](https://drive.google.com/drive/u/3/folders/1340tEG3_bL9ojHJ8hQmMkBoZ9dSKYUhV)>

### Dataset splits

- **Train/validation/test splits**  
JSON file defining fixed sample splits used across all notebooks.

  Download:
  https://<[link-to-splits-json](https://drive.google.com/drive/u/3/folders/1340tEG3_bL9ojHJ8hQmMkBoZ9dSKYUhV)>
---

## Pipeline overview

The notebooks are intended to be run in numerical order.
Each notebook has a single, clearly defined responsibility.

### Phase 1: Modality-specific autoencoders  
**`1_Phase1_Train_Autoencoders.ipynb`**

- Train denoising autoencoders independently for each omics modality
- Use feature-level masking to encourage robust reconstruction
- Save modality-specific encoder and decoder checkpoints


### Phase 2: Shared latent space training  
**`2_Phase2_Train_MAE.ipynb`**

- Initialize from pretrained modality-specific autoencoders
- Jointly fine-tune all modalities into a shared latent space
- Optimize a combination of:
  - reconstruction loss
  - contrastive alignment loss
  - cross-modality imputation loss
- Save the trained shared-space model


### Phase 3: Imputation with missing modalities  
**`3_Imputation_Missing_Modality.ipynb`**

- Generate missing-modality imputation results using the shared model
- Evaluate leave-one-modality-out and all modality-availability patterns
- Save scenario definitions and MIMIR predictions
- No benchmarking is performed in this notebook


### Phase 4: Benchmarking missing-modality imputation  
**`4_Benchmark_Missing_Modalities.ipynb`**

- Benchmark MIMIR against TOBMI-style translation and MOFA+
- Use identical missing-modality scenarios for all methods
- Report global and feature-wise reconstruction performance


### Phase 5: Imputation with missing values  
**`5_Imputation_Missing_Values.ipynb`**

- Generate missing-value imputation results using MIMIR
- Consider two missingness mechanisms:
  - MCAR (Missing Completely At Random)
  - MNAR (Missing Not At Random)
- Save masks, corrupted inputs, and predictions
- No baseline comparisons are performed in this notebook


### Phase 6: Benchmarking missing-value imputation  
**`6_Benchmark_Missing_Values.ipynb`**

- Benchmark MIMIR against SoftImpute and KNN-based imputation
- Use identical masks and corrupted inputs for all methods
- Evaluate MCAR and MNAR settings separately

---


## Notes

- Notebooks are intended to be executed sequentially.
- Legacy or exploratory cells are explicitly marked and are not part of the final pipeline.
- All evaluation metrics are computed on held-out test data only.

---

## Citation

If you use this code, please cite the accompanying manuscript .
