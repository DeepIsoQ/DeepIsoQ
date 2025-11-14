
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import matplotlib.pyplot as plt
import os

# --------------------
# CONFIG
# --------------------
RUNDIR = os.path.join(os.environ["BLACKHOLE"], os.environ["USER"])
DATA_PT   = f"{RUNDIR}/data.pt"  # path to preprocessed data
VAL_SIZE  = 0.15
SEED      = 42
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# UTILS
# --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------
# DATA (desde data.pt)
# --------------------
print(f"Loading preprocessed tensors from {DATA_PT}...")
data = torch.load(DATA_PT, map_location="cpu")

# Xg_log1p ya está en log1p → usamos tal cual
X = data["Xg_log1p"].numpy()          # (N, G)

# Y_tx está crudo → aplicamos log1p como antes
Y = data["Y_tx"].numpy()              # (N, I)
Y = np.log1p(Y)

print("Shapes from .pt:", X.shape, Y.shape)

# Estandarización igual que antes
xscaler = StandardScaler()
yscaler = StandardScaler()
X = xscaler.fit_transform(X)
Y = yscaler.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=VAL_SIZE, random_state=SEED
)

X_train, X_val = map(torch.tensor, [X_train, X_val])
Y_train, Y_val = map(torch.tensor, [Y_train, Y_val])
X_train, X_val, Y_train, Y_val = (
    X_train.to(DEVICE), X_val.to(DEVICE),
    Y_train.to(DEVICE), Y_val.to(DEVICE)
)

print(f"Data shapes: X={X.shape}, Y={Y.shape}, "
      f"train={X_train.shape[0]}, val={X_val.shape[0]}")

# --------------------
# MODEL
# --------------------
class FFNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, act="relu", dropout=0.0, batchnorm=False):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU() if act == "relu" else nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --------------------
# TRAIN + EVAL
# --------------------
def pearson_mean(y_true, y_pred):
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    r = (yt*yp).sum(dim=0) / (torch.sqrt((yt**2).sum(dim=0)) * torch.sqrt((yp**2).sum(dim=0)) + 1e-8)
    return torch.nanmean(r).item()

def train_model(model, Xtr, Ytr, Xva, Yva, lr=1e-3, epochs=200, patience=20, weight_decay=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(1, patience//2))
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    patience_left = patience

    hist = {"train_mse": [], "val_mse": [], "val_pearson": []}

    for epoch in range(1, epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(Xtr)
        loss = loss_fn(pred, Ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_loss = loss_fn(val_pred, Yva).item()
            val_r = pearson_mean(Yva, val_pred)
            train_loss = loss_fn(pred, Ytr).item()

        hist["train_mse"].append(train_loss)
        hist["val_mse"].append(val_loss)
        hist["val_pearson"].append(val_r)

        sched.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | Val r: {val_r:.4f}")

        if patience_left == 0:
            print("Early stopping!")
            break

    model.load_state_dict(best_state)

    # eval final
    model.eval()
    with torch.no_grad():
        val_pred = model(Xva)
        final_mse = loss_fn(val_pred, Yva).item()
        final_r   = pearson_mean(Yva, val_pred)
    return final_mse, final_r, hist

# --------------------
# CONFIGURACIONES
# --------------------
configs = [
    {"name": "shallow",      "hidden": [1024],                 "dropout": 0.1,  "act": "relu", "batchnorm": False, "lr": 1e-3},
    {"name": "deep",         "hidden": [1024, 512, 256],       "dropout": 0.2,  "act": "relu", "batchnorm": True,  "lr": 8e-4},
    {"name": "gelu_reslike", "hidden": [2048, 1024],           "dropout": 0.1,  "act": "gelu", "batchnorm": True,  "lr": 1e-3},
    {"name": "narrow_deep",  "hidden": [512, 512, 512, 512],   "dropout": 0.25, "act": "relu", "batchnorm": True,  "lr": 7e-4},
    {"name": "bottleneck",   "hidden": [2048, 256, 2048],      "dropout": 0.1,  "act": "relu", "batchnorm": True,  "lr": 1e-3},
    {"name": "small",        "hidden": [256, 256],             "dropout": 0.4,  "act": "relu", "batchnorm": False, "lr": 1.2e-3},
]

# --------------------
# RUN + PLOTS
# --------------------
set_seed(SEED)
results = []
histories = {}

for cfg in configs:
    print("="*80)
    print(f"Training {cfg['name']} ...")
    model = FFNN(
        in_dim=X.shape[1],
        out_dim=Y.shape[1],
        hidden=cfg["hidden"],
        act=cfg.get("act", "relu"),
        dropout=cfg.get("dropout", 0.0),
        batchnorm=cfg.get("batchnorm", False)
    ).to(DEVICE)

    mse_val, pearson_val, hist = train_model(
        model, X_train, Y_train, X_val, Y_val,
        lr=cfg.get("lr", 1e-3), epochs=200, patience=20, weight_decay=1e-4
    )
    results.append({"name": cfg["name"], "val_mse": mse_val, "val_pearson": pearson_val})
    histories[cfg["name"]] = hist

    # ---- Plot de pérdidas por arquitectura (una figura por modelo) ----
    plt.figure()
    plt.plot(hist["train_mse"], label="Train MSE")
    plt.plot(hist["val_mse"], label="Val MSE")
    plt.title(f"Loss curves - {cfg['name']}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(f"MSE_epoch_{cfg}.png", dpi=300)
    plt.show()

print("\nResultados finales:")
df = pd.DataFrame(results)
print(df.sort_values("val_pearson", ascending=False))

# ---- Gráfico resumen de Pearson por arquitectura ----
order = df.sort_values("val_pearson", ascending=False)
plt.figure()
plt.bar(order["name"], order["val_pearson"])
plt.title("Validation Pearson (mean over outputs)")
plt.xlabel("Architecture")
plt.ylabel("Pearson r")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("val_mse.png", dpi=300, bbox_inches="tight")
plt.show()
