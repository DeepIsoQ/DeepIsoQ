import numpy as np
import torch
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
import os

# ------------------------------
# Config
# ------------------------------
SEED            = 42
TEST_FRAC       = 0.15
VAL_FRAC        = 0.15
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
AMP             = (DEVICE == "cuda")

# ------------------------------
# Data path (portable)
# ------------------------------
DATA_PT = os.environ.get("DATA_PT")

if DATA_PT is None:
    # Fall back to $BLACKHOLE/$USER/data.pt
    bh = os.environ.get("BLACKHOLE")
    user = os.environ.get("USER")
    if bh is None or user is None:
        raise RuntimeError(
            "DATA_PT not set and BLACKHOLE/USER env vars missing. "
            "Either export DATA_PT or run on DTU HPC where BLACKHOLE & USER are defined."
        )
    DATA_PT = os.path.join(bh, user, "data.pt")

print(f"[INFO] Loading tensors from: {DATA_PT}")
data = torch.load(DATA_PT, map_location="cpu", weights_only=False)


# --- Load data ---
X = data["Xg_log1p"].float().cpu().numpy()                 # (N, G)
Y = torch.log1p(data["Y_tx"].float()).cpu().numpy()        # (N, I)

# --- Standardize ---
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X = scaler_x.fit_transform(X)
Y = scaler_y.fit_transform(Y)

# --- Train/test split (sample-wise) ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_FRAC, random_state=SEED, shuffle=True
)

# --- PCA only on training X ---
n_components = min(50, X_train.shape[0], X_train.shape[1])
pca = PCA(n_components=n_components)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# --- Convert to PyTorch ---
X_train_t = torch.tensor(X_train_pca, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_pca, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test,  dtype=torch.float32)

# --- DataLoaders ---
train_loader = DataLoader(
    TensorDataset(X_train_t, Y_train_t),
    batch_size=32, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test_t, Y_test_t),
    batch_size=32, shuffle=False
)