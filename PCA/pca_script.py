import numpy as np
import torch
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import scanpy as sc
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------------
# Config
# ------------------------------
SEED            = 42
TEST_FRAC       = 0.15
VAL_FRAC        = 0.15
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
AMP             = (DEVICE == "cuda")

PCA_DIM        = 1000  # Number of PCA dimensions. 
#45,000+ genes in full dataset, so 1000 might be a decent reduction.

BATCH_SIZE     = 128 # 256 would also work if memory allows. 

N_EPOCHS       = 30 #The amount of epochs used for training


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


###########################################################################

#Load data and preprocess with PCA: 


# --- Load data ---
X = data["Xg_log1p"].float().cpu().numpy()
Y = torch.log1p(data["Y_tx"].float()).cpu().numpy()

# --- 1. Split Data FIRST (Train/Val/Test) ---
# First split off Test
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=TEST_FRAC, random_state=SEED, shuffle=True
)

# Then split remaining "temp" into Train and Val
# Adjust val_size to be relative to the temp size
relative_val_size = VAL_FRAC / (1 - TEST_FRAC)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=relative_val_size, random_state=SEED, shuffle=True
)

# --- 2. Standardize (Fit on Train ONLY) ---
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_val   = scaler_x.transform(X_val)   # transform only
X_test  = scaler_x.transform(X_test)  # transform only

scaler_y = StandardScaler()
Y_train = scaler_y.fit_transform(Y_train)
Y_val   = scaler_y.transform(Y_val)
Y_test  = scaler_y.transform(Y_test)

# --- 3. PCA (Fit on Train ONLY) ---
pca = PCA(n_components=PCA_DIM)
X_train_pca = pca.fit_transform(X_train)
X_val_pca   = pca.transform(X_val)
X_test_pca  = pca.transform(X_test)

#Note: It would create a bias if we use the standard deviation and mean from the whole dataset to standardize.
#Instead we only use the training data to fit the scaler, and then apply the same transformation to validation and test data.


# --- Convert to Tensors ---
X_train_t = torch.tensor(X_train_pca, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_pca,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test_pca,  dtype=torch.float32)

Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test,  dtype=torch.float32)


# --- Step 6: DataLoaders ---

# Create datasets for all three splits
train_dataset = TensorDataset(X_train_t, Y_train_t)
val_dataset   = TensorDataset(X_val_t,   Y_val_t)
test_dataset  = TensorDataset(X_test_t,  Y_test_t)

# Create loaders (Note: it uses the BATCH_SIZE from config, at the top of the code)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False) # <--- Validate on this
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False) # <--- Save this for the very end

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")



##################################################################

#Defining model - For PCA input only: 

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model definition ---
class IsoformPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# --- Initialize model, optimizer, and loss ---
input_dim = X_train_t.shape[1]
output_dim = Y_train_t.shape[1]

model = IsoformPredictor(input_dim, output_dim).to(device) #The code is flexible to input PCA dimensions. 
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_fn = nn.MSELoss() #Mean Squared Error Loss

#This loss function should be safe to use. 
#As we already used torch.log1p (will squash outliers), 
#and subsequently we used StandardScaler (centers data). 



#####################################################################################

#Training the model:

# --- Training loop ---

#Note: We define the N_EPOCHS in the config section, at the top of the code. 

train_losses = []
val_losses = []  # Rename to val_losses

for epoch in range(N_EPOCHS):
    # --- Training ---
    model.train()
    total_train_loss = 0
    for xb, yb in train_loader: # Use the data loader for the training dataset, i.e. train_loader
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- Validation ---
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for xb, yb in val_loader: #Use validation dataset loader
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{N_EPOCHS}]  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")

# --- Final Test Evaluation ---
print("\n[INFO] Running Final Test Evaluation...")
model.eval()
total_test_loss = 0
with torch.no_grad():
    for xb, yb in test_loader: #Finally, use test dataset loader for the final evaluation. 
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        total_test_loss += loss.item()
print(f"FINAL TEST LOSS: {total_test_loss / len(test_loader):.4f}")

