import pickle
import pandas as pd
import numpy as np

data_path = "data/tcga_redo_mlomicZ.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)

meth = data["methylation"]
print("Methylation data shape:", meth.shape)
print("Min:", meth.min().min())
print("Max:", meth.max().max())
print("Mean:", meth.mean().mean())
print("Std (overall):", meth.values.flatten().std())
print("Sample of means per feature:", meth.mean(axis=0).head())
print("Sample of stds per feature:", meth.std(axis=0).head())
