#!/usr/bin/env python3

# Date: 24/11-2025
# Purpose: This script performs architecture search for Feed-Forward Neural Networks (FFNNs)
# that predict isoform-level transcript expression from gene-level expression data.
# This version is adapted to work with PCA-reduced input data while maintaining
# full-dimensional transcript expression output.

"""
FFNN Random Search with GPU Acceleration and Automatic Mixed Precision (AMP)

SCIENTIFIC CONTEXT:
-------------------
This script trains feed-forward neural networks to map gene expression levels to 
transcript/isoform expression levels. In biological terms:
- Input (X): Gene-level expression (aggregated from all isoforms of a gene)
- Output (Y): Transcript/isoform-level expression (detailed breakdown)

This is a complex regression problem because one gene can produce multiple isoforms
through alternative splicing, and the model must learn these patterns.

HYPERPARAMETER OPTIMIZATION (HPO):
-----------------------------------
The script performs random search over:
1. Network architectures: Different layer configurations (depth & width)
2. Activation functions: ReLU, GELU, Tanh, LeakyReLU
3. Regularization: Dropout rates (0.0, 0.1, 0.2)
4. Normalization: BatchNorm on/off
5. Training dynamics: Learning rates and batch sizes
6. Early stopping to prevent overfitting

OUTPUTS:
--------
1. arch_search_results.csv: Detailed metrics for all trials
2. arch_search_summary.json: Best model info + top 5 performers
3. best_model.pt: Saved weights + preprocessing statistics for deployment
4. figs_trials/curves_*.png: Training/validation curves for each trial
5. summary_val_mse_bar.png: Bar chart comparing all models
6. top5_val_curves.png: Overlay of best 5 validation curves
"""

# ============================================================================
# IMPORTS AND ENVIRONMENT SETUP
# ============================================================================

import os, json, math, random, csv, time, pathlib

# Memory optimization for PyTorch CUDA allocator
# "expandable_segments" allows the allocator to create new memory segments more flexibly
# This helps prevent out-of-memory errors during training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np  # For numerical operations and data handling
import torch  # PyTorch for neural network operations
import torch.nn as nn  # Neural network modules (layers, loss functions, etc.)
from sklearn.model_selection import train_test_split  # For splitting data into train/val/test

# ============================================================================
# CONFIGURATION AND HYPERPARAMETERS
# ============================================================================

# Reproducibility seed - ensures same random numbers across runs
SEED = 42

# Data split proportions
TEST_FRAC = 0.15   # 15% of data for final testing (held out until end)
VAL_FRAC = 0.15    # 15% of remaining data for validation (model selection)
                   # This means ~72% train, ~13% val, ~15% test

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
AMP = (DEVICE == "cuda")  # Automatic Mixed Precision (FP16/FP32) - faster on modern GPUs

# Random search configuration
N_TRIALS = 12  # Number of different hyperparameter combinations to try

# Training hyperparameters
MAX_EPOCHS = 120    # Maximum epochs per trial
PATIENCE = 10       # Early stopping: stop if no improvement for this many checks
EVAL_EVERY = 5      # Evaluate validation set every N epochs (saves time)
GRAD_CLIP = 1.0     # Gradient clipping threshold to prevent exploding gradients

# Hyperparameter search space (HPO will randomly sample from these)
BATCHES = [128, 192, 256]           # Different mini-batch sizes
LRS = [1e-3, 2e-3, 3e-3]            # Learning rates to try
DROPOUTS = [0.0, 0.1, 0.2]          # Dropout probabilities (0.0 = no dropout)
BATCHNORMS = [False, True]          # Whether to use Batch Normalization
ACTS = ["tanh", "relu", "gelu", "leakyrelu"]  # Activation functions

# Output file paths
RESULTS_CSV = "ffnn_search/results/arch_search_results.csv"
BEST_MODEL_PT = "ffnn_search/results/best_model.pt"
SUMMARY_JSON = "ffnn_search/results/arch_search_summary.json"

# Set matplotlib to non-interactive backend (important for cluster environments without display)
os.environ.setdefault("MPLBACKEND", "Agg")

# Figure output paths
TRIAL_FIG_DIR = "ffnn_search/results/figs_trials"  # Individual trial curves
SUMMARY_FIG_BAR = "ffnn_search/results/summary_val_mse_bar.png"  # Bar chart of all models
SUMMARY_FIG_TOP5 = "ffnn_search/results/top5_val_curves.png"  # Top 5 models overlay
BEST_FIG_CURVES = "ffnn_search/results/best_model_curves.png"  # Best model curves

# ============================================================================
# REPRODUCIBILITY AND PERFORMANCE OPTIMIZATION
# ============================================================================

def set_seed(s):
    """
    Set random seeds for all libraries to ensure reproducible results.
    
    Why this matters:
    - Neural network initialization uses random values
    - Data shuffling uses random order
    - Dropout masks are random
    Without fixing seeds, you get different results each run.
    
    Args:
        s (int): Seed value
    """
    random.seed(s)              # Python's built-in random
    np.random.seed(s)           # NumPy's random number generator
    torch.manual_seed(s)        # PyTorch CPU random numbers
    torch.cuda.manual_seed_all(s)  # PyTorch GPU random numbers (all GPUs)

set_seed(SEED)  # Apply seed globally

# Performance optimization for NVIDIA GPUs with Tensor Cores
# TF32 (TensorFloat-32) uses fewer bits but maintains good accuracy while being faster
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

# ============================================================================
# DATA LOADING (Portable across environments)
# ============================================================================

# This section makes the script work both locally and on DTU's HPC cluster
# It tries multiple strategies to find the data file

DATA_PT = os.environ.get("DATA_PT")  # First, try explicit environment variable

if DATA_PT is None:
    # Fallback: Construct path using DTU HPC environment variables
    # $BLACKHOLE is DTU's fast scratch storage
    # $USER is the username
    bh = os.environ.get("BLACKHOLE")
    user = os.environ.get("USER")
    
    if bh is None or user is None:
        raise RuntimeError(
            "DATA_PT not set and BLACKHOLE/USER env vars missing. "
            "Either export DATA_PT or run on DTU HPC where BLACKHOLE & USER are defined."
        )
    
    DATA_PT = os.path.join(bh, user, "data.pt")

print(f"[INFO] Loading tensors from: {DATA_PT}")

# Load the pre-processed data
# weights_only=False allows loading custom objects (needed for complex data structures)
data = torch.load(DATA_PT, map_location="cpu", weights_only=False)

# Extract input and output matrices
# X: Gene expression (log1p transformed for variance stabilization)
# Y: Transcript expression (will also be log1p transformed)
X = data["Xg_log1p"].float().cpu().numpy()
Y = torch.log1p(data["Y_tx"].float()).cpu().numpy()

# Get dimensions
N, G = X.shape  # N = number of samples, G = number of genes (input features)
_, I = Y.shape  # I = number of transcripts/isoforms (output features)

print(f"[INFO] Shapes: X={X.shape}, Y={Y.shape}")

# ============================================================================
# DATA SPLITTING (Train / Validation / Test)
# ============================================================================

"""
Three-way split strategy:
1. First split: Separate test set (held out completely)
2. Second split: Divide remaining into train and validation
3. Train: Used for gradient updates
4. Validation: Used for model selection and early stopping
5. Test: Used ONLY at the very end for final performance assessment

This prevents "data leakage" where test performance influences model selection.
"""

all_idx = np.arange(N)  # Array of all sample indices [0, 1, 2, ..., N-1]

# Split 1: Hold out test set
trval_idx, te_idx = train_test_split(
    all_idx, 
    test_size=TEST_FRAC,  # 15% for test
    random_state=SEED,    # Reproducible split
    shuffle=True          # Randomize before splitting
)

# Split 2: Divide remaining into train and validation
# Calculate relative validation fraction (since we already removed test set)
val_rel = VAL_FRAC / (1.0 - TEST_FRAC)  # ~17.6% of remaining data

tr_idx, va_idx = train_test_split(
    trval_idx,
    test_size=val_rel,
    random_state=SEED,
    shuffle=True
)

print(f"[INFO] Split sizes: train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

# ============================================================================
# DATA NORMALIZATION (Z-score standardization)
# ============================================================================

"""
Normalization is critical for neural networks:
- Makes training more stable
- Allows higher learning rates
- Reduces sensitivity to feature scales

We use ONLY training set statistics to normalize all sets.
This simulates real-world deployment where we won't know test set statistics.
"""

# Calculate mean and std from training set ONLY
X_mean = X[tr_idx].mean(axis=0)  # Mean per gene (shape: [G])
X_std = X[tr_idx].std(axis=0) + 1e-8  # Std per gene + small epsilon to avoid division by zero

# Apply z-score normalization: z = (x - μ) / σ
Xz = (X - X_mean) / X_std

# Convert to PyTorch tensors and move to GPU (if available)
# float() ensures 32-bit precision (standard for deep learning)
Xt = torch.from_numpy(Xz).to(DEVICE).float()
Yt = torch.from_numpy(Y).to(DEVICE).float()

# Convert indices to tensors on the same device (for efficient indexing)
tr_idx_t = torch.from_numpy(tr_idx).to(DEVICE)
va_idx_t = torch.from_numpy(va_idx).to(DEVICE)
te_idx_t = torch.from_numpy(te_idx).to(DEVICE)

# ============================================================================
# MINI-BATCH ITERATOR
# ============================================================================

def batch_iter(idxs_t, batch_size, shuffle=True):
    """
    Generator that yields mini-batches of data.
    
    Mini-batch training:
    - More efficient than single-sample updates
    - More stable than full-batch updates
    - Allows training on datasets larger than GPU memory
    
    Args:
        idxs_t (torch.Tensor): Indices of samples to iterate over
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to randomize order (True for training, False for evaluation)
    
    Yields:
        tuple: (x_batch, y_batch) - Input and output tensors for the batch
    """
    if shuffle:
        # Randomize order at the start of each epoch (prevents overfitting to order)
        idxs_t = idxs_t[torch.randperm(idxs_t.numel(), device=idxs_t.device)]
    
    # Iterate over batches
    for i in range(0, idxs_t.numel(), batch_size):
        j = idxs_t[i:i+batch_size]  # Get batch indices
        # index_select is efficient for gathering specific rows on GPU
        yield Xt.index_select(0, j), Yt.index_select(0, j)

# ============================================================================
# ACTIVATION FUNCTIONS AND WEIGHT INITIALIZATION
# ============================================================================

def get_activation(name: str):
    """
    Factory function to create activation function layers.
    
    Activation functions introduce non-linearity, allowing networks to learn
    complex patterns. Different activations have different properties:
    
    - ReLU: Fast, but can cause "dying neurons" (always output 0)
    - GELU: Smooth, probabilistic, works well with transformers
    - LeakyReLU: Like ReLU but allows small negative gradients
    - Tanh: Classic, outputs in [-1, 1], but can saturate
    
    Args:
        name (str): Name of activation function
    
    Returns:
        nn.Module: Activation layer
    """
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "gelu":
        return nn.GELU()
    if n in ("leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(0.01)  # Small slope (0.01) for negative values
    if n == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

def init_linear(m, act: str):
    """
    Initialize weights of linear layers based on activation function.
    
    Proper initialization is crucial for training:
    - Too small: Gradients vanish (no learning)
    - Too large: Gradients explode (unstable training)
    
    Xavier (for Tanh): Assumes linear activation, variance-preserving
    Kaiming/He (for ReLU-family): Accounts for ReLU zeroing negative values
    
    Args:
        m (nn.Module): Module to initialize
        act (str): Activation function name (determines initialization scheme)
    """
    if not isinstance(m, nn.Linear):
        return  # Only initialize linear layers
    
    a = act.lower()
    if a == "tanh":
        # Xavier initialization: Good for symmetric activations like tanh
        nn.init.xavier_normal_(m.weight)
    else:
        # Kaiming initialization: Good for ReLU and variants
        # Accounts for the fact that ReLU zeros out half the activations
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    
    # Initialize biases to zero (standard practice)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class FFNN(nn.Module):
    """
    Feed-Forward Neural Network (FFNN) with configurable architecture.
    
    Architecture pattern (for each hidden layer):
    1. Linear transformation: y = Wx + b
    2. [Optional] Batch Normalization: Normalizes activations
    3. Activation function: Introduces non-linearity
    4. [Optional] Dropout: Randomly zeros activations (regularization)
    
    Final layer: Linear transformation to output dimension (no activation)
    
    Args:
        in_dim (int): Input dimension (number of genes)
        out_dim (int): Output dimension (number of transcripts)
        hidden (list): List of hidden layer sizes, e.g., [1024, 512] for two layers
        act (str): Activation function name
        dropout (float): Dropout probability (0.0 = no dropout)
        batchnorm (bool): Whether to use Batch Normalization
    """
    def __init__(self, in_dim, out_dim, hidden, act="gelu", dropout=0.0, batchnorm=False):
        super().__init__()  # Initialize parent class (required)
        
        layers = []  # List to accumulate layers
        prev = in_dim  # Track previous layer size
        
        # Build hidden layers
        for h in hidden:
            # 1. Linear layer
            layers.append(nn.Linear(prev, h))
            
            # 2. Optional Batch Normalization
            # Normalizes inputs to next activation, stabilizes training
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            
            # 3. Activation function
            layers.append(get_activation(act))
            
            # 4. Optional Dropout
            # Randomly zeros activations during training (prevents overfitting)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev = h  # Update for next layer
        
        # Output layer: No activation (regression problem)
        layers.append(nn.Linear(prev, out_dim))
        
        # Combine all layers into a sequential module
        self.net = nn.Sequential(*layers)
        
        # Store activation name for initialization
        self.act_name = act
    
    def forward(self, x):
        """
        Forward pass: Compute predictions from inputs.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, in_dim]
        
        Returns:
            torch.Tensor: Predictions [batch_size, out_dim]
        """
        return self.net(x)

# ============================================================================
# LOSS FUNCTION AND EVALUATION METRICS
# ============================================================================

# Mean Squared Error: Standard loss for regression
# MSE = (1/n) Σ(y_pred - y_true)²
criterion = nn.MSELoss()

def evaluate_on(model, idxs_t, batch_size):
    """
    Evaluate model performance on a dataset (validation or test).
    
    Key differences from training:
    - model.eval(): Disables dropout, switches BatchNorm to inference mode
    - torch.no_grad(): Disables gradient computation (saves memory, faster)
    - shuffle=False: Maintain order (not needed for evaluation)
    
    Args:
        model (nn.Module): Neural network to evaluate
        idxs_t (torch.Tensor): Indices of samples to evaluate on
        batch_size (int): Batch size for evaluation
    
    Returns:
        float: Average MSE loss over all samples
    """
    model.eval()  # Set to evaluation mode
    total, seen = 0.0, 0  # Track cumulative loss and sample count
    
    with torch.no_grad():  # Disable gradient computation
        for xb, yb in batch_iter(idxs_t, batch_size=batch_size, shuffle=False):
            # Use mixed precision if enabled (faster on modern GPUs)
            with torch.cuda.amp.autocast(enabled=AMP):
                pb = model(xb)  # Forward pass
                loss = criterion(pb, yb)  # Compute loss
            
            # Accumulate weighted loss (weight by batch size for correct averaging)
            total += float(loss.item()) * yb.size(0)
            seen += yb.size(0)
    
    return total / max(1, seen)  # Return average loss

def pearson_mean_gpu(Y_true, Y_pred):
    """
    Compute mean Pearson correlation coefficient across all outputs (GPU-accelerated).
    
    Pearson correlation measures linear relationship between predicted and true values:
    - r = 1: Perfect positive correlation
    - r = 0: No linear relationship
    - r = -1: Perfect negative correlation
    
    For each transcript, we compute:
    r = Σ[(y_true - mean(y_true)) * (y_pred - mean(y_pred))] / 
        [sqrt(Σ(y_true - mean)²) * sqrt(Σ(y_pred - mean)²)]
    
    Then average across all transcripts.
    
    Args:
        Y_true (torch.Tensor): True values [N, I] on GPU
        Y_pred (torch.Tensor): Predicted values [N, I] on GPU
    
    Returns:
        float: Mean Pearson correlation across all transcripts
    """
    # Center the data (subtract mean)
    yt = Y_true - Y_true.mean(dim=0, keepdim=True)
    yp = Y_pred - Y_pred.mean(dim=0, keepdim=True)
    
    # Numerator: Covariance
    num = (yt * yp).sum(dim=0)
    
    # Denominator: Product of standard deviations
    # +1e-8 prevents division by zero for constant predictions
    den = torch.sqrt((yt * yt).sum(dim=0)) * torch.sqrt((yp * yp).sum(dim=0)) + 1e-8
    
    # Compute correlation for each transcript
    r = num / den
    
    # Return mean correlation (nanmean ignores NaN values if any)
    return r.nanmean().item()

# ============================================================================
# TRAINING FUNCTION (Single Trial)
# ============================================================================

def train_once(hp, trial_seed):
    """
    Train a single model with given hyperparameters.
    
    This function encapsulates the entire training loop:
    1. Initialize model with given architecture
    2. Set up optimizer and learning rate scheduler
    3. Train for multiple epochs with early stopping
    4. Track training and validation curves
    5. Save best model based on validation performance
    6. Generate visualization
    
    Args:
        hp (dict): Hyperparameters dictionary containing:
            - name: Model identifier
            - hidden: List of hidden layer sizes
            - act: Activation function
            - dropout: Dropout probability
            - batchnorm: Whether to use BatchNorm
            - lr: Learning rate
            - batch_size: Mini-batch size
            - epochs: Maximum epochs
        trial_seed (int): Random seed for this trial
    
    Returns:
        tuple: (record_dict, trained_model, curves_dict)
            - record_dict: Metrics and hyperparameters
            - trained_model: The trained PyTorch model
            - curves_dict: Training and validation loss curves
    """
    # Set seed for reproducibility of this trial
    set_seed(trial_seed)
    
    # Initialize model with specified architecture
    model = FFNN(
        G, I,  # Input and output dimensions (global variables)
        hidden=hp["hidden"],
        act=hp["act"],
        dropout=hp["dropout"],
        batchnorm=hp["batchnorm"]
    ).to(DEVICE)
    
    # Initialize weights appropriately for the activation function
    model.apply(lambda m: init_linear(m, hp["act"]))
    
    # ========================================================================
    # OPTIMIZER SETUP
    # ========================================================================
    
    # AdamW: Adam with decoupled weight decay (better than L2 regularization)
    # - Adaptive learning rates per parameter
    # - Momentum-like behavior
    # - Weight decay: Prevents weights from growing too large
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=1e-4  # L2 regularization strength
    )
    
    # Learning rate scheduler: Reduce LR when validation stops improving
    # - factor=0.5: Multiply LR by 0.5 when triggered
    # - patience=2: Wait 2 evaluations before reducing
    # This helps escape plateaus and fine-tune at the end
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        factor=0.5,
        patience=2
    )
    
    # Gradient scaler for mixed precision training
    # Scales loss to prevent underflow in FP16, then unscales before optimizer step
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    
    # ========================================================================
    # TRAINING STATE TRACKING
    # ========================================================================
    
    best_val = math.inf  # Best validation loss seen so far
    best_state = None    # Model weights at best validation
    noimp = 0            # Epochs without improvement (for early stopping)
    train_curve = []     # Training loss per epoch
    val_curve = []       # Validation loss per evaluation
    
    t0 = time.time()  # Start timing
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    for epoch in range(1, hp["epochs"] + 1):
        model.train()  # Set to training mode (enables dropout, BatchNorm training)
        total, seen = 0.0, 0  # Accumulate loss and sample count
        
        # Iterate over mini-batches
        for xb, yb in batch_iter(tr_idx_t, batch_size=hp["batch_size"], shuffle=True):
            # Zero gradients from previous iteration
            # set_to_none=True is slightly faster than zero_grad()
            opt.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=AMP):
                pb = model(xb)  # Predictions
                loss = criterion(pb, yb)  # Compute loss
            
            # Backward pass (compute gradients)
            # Scale loss to prevent underflow in FP16
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            if GRAD_CLIP is not None:
                scaler.unscale_(opt)  # Unscale before clipping
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            # Optimizer step: Update weights
            scaler.step(opt)
            scaler.update()  # Update scaler for next iteration
            
            # Accumulate training loss
            total += float(loss.item()) * yb.size(0)
            seen += yb.size(0)
        
        # Calculate average training loss for this epoch
        epoch_train = total / max(1, seen)
        train_curve.append(epoch_train)
        
        # ====================================================================
        # VALIDATION AND EARLY STOPPING
        # ====================================================================
        
        # Evaluate on validation set (not every epoch to save time)
        if epoch == 1 or epoch % EVAL_EVERY == 0:
            val_mse = evaluate_on(model, va_idx_t, batch_size=hp["batch_size"])
            val_curve.append(val_mse)
            
            # Update learning rate based on validation performance
            sch.step(val_mse)
            
            # Check if this is the best model so far
            if val_mse < best_val - 0.0:  # Improved
                best_val = val_mse
                # Save model state (deep copy to avoid reference issues)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                noimp = 0  # Reset patience counter
            else:  # No improvement
                noimp += 1
            
            # Print progress
            print(f"[{hp['name']}] ep {epoch:03d} | act={hp['act']} | "
                  f"train {epoch_train:.5f} | val {val_mse:.5f} | "
                  f"lr {opt.param_groups[0]['lr']:.4g}")
            
            # Early stopping check
            if noimp >= PATIENCE:
                print(f"[{hp['name']}] Early stopping.")
                break
    
    # Restore best model weights
    if best_state:
        model.load_state_dict(best_state)
    
    # Final validation evaluation
    val_mse = evaluate_on(model, va_idx_t, batch_size=hp["batch_size"])
    train_time = time.time() - t0
    
    # ========================================================================
    # VISUALIZATION: Training/Validation Curves
    # ========================================================================
    
    pathlib.Path(TRIAL_FIG_DIR).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        
        fig_path = os.path.join(TRIAL_FIG_DIR, f"curves_{hp['name']}.png")
        
        plt.figure(figsize=(7.5, 4.5))
        
        # Plot training curve (every epoch)
        plt.plot(train_curve, label="train MSE")
        
        # Plot validation curve (only evaluated epochs)
        xs_v = [e for e in range(1, len(train_curve)+1) 
                if e == 1 or e % EVAL_EVERY == 0][:len(val_curve)]
        plt.plot(xs_v, val_curve, "o-", label="val MSE")
        
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.title(f"{hp['name']} — act={hp['act']}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] Plot failed for {hp['name']}: {e}")
    
    # ========================================================================
    # RETURN RESULTS
    # ========================================================================
    
    # Create record dictionary with all metrics and hyperparameters
    rec = {
        "name": hp["name"],
        "hidden": hp["hidden"],
        "act": hp["act"],
        "dropout": hp["dropout"],
        "batchnorm": hp["batchnorm"],
        "lr": hp["lr"],
        "momentum": "-",      # Not used (AdamW has its own momentum)
        "step_size": "-",     # Not used (ReduceLROnPlateau instead)
        "gamma": "-",         # Not used
        "batch_size": hp["batch_size"],
        "epochs_trained": len(train_curve),
        "val_mse": float(val_mse),
        "val_pearson": float("nan"),  # Will be computed only for best model
        "train_time_sec": round(train_time, 1),
    }
    
    curves = {"train": train_curve, "val": val_curve}
    
    return rec, model, curves

# ============================================================================
# HYPERPARAMETER SEARCH SPACE DEFINITION
# ============================================================================

# Different network architectures to try
# Each tuple: (name, [hidden_layer_sizes])
ARCHS = [
    # Shallow but wide - good for simple patterns
    ("shallow_1024", [1024]),
    
    # Deep with gradual narrowing - classic approach
    ("deep_2048_1024", [2048, 1024]),
    
    # Deep narrow - more layers, less parameters
    ("narrow_x4_512", [512, 512, 512, 512]),
    
    # Bottleneck - compression in middle (like autoencoder)
    ("bottleneck_2048_256", [2048, 256, 2048]),
    
    # Gradual narrowing - smooth transition
    ("medium_1536_768_384", [1536, 768, 384]),
    
    # Small and shallow - baseline/fast option
    ("small_256_256", [256, 256]),
]

def sample_hp(t):
    """
    Sample a random hyperparameter configuration for trial t.
    
    Uses a separate random number generator seeded with trial number
    to ensure reproducibility while allowing different configs per trial.
    
    Args:
        t (int): Trial number
    
    Returns:
        dict: Hyperparameter configuration
    """
    # Create trial-specific random generator
    rnd = random.Random(SEED + 1000 + t)
    
    # Randomly select architecture
    name, hidden = rnd.choice(ARCHS)
    
    return {
        "name": f"{name}_t{t}",  # Add trial number to name
        "hidden": hidden,
        "act": rnd.choice(ACTS),
        "dropout": rnd.choice(DROPOUTS),
        "batchnorm": rnd.choice(BATCHNORMS),
        "lr": rnd.choice(LRS),
        "batch_size": rnd.choice(BATCHES),
        "epochs": MAX_EPOCHS,
    }

# ============================================================================
# MAIN HYPERPARAMETER SEARCH LOOP
# ============================================================================

print(f"[INFO] Starting random search with {N_TRIALS} trials...")

# Storage for all results
results = []  # List of result dictionaries
curves_by_name = {}  # Maps model name to training curves

# Track best model globally
best_rec = None      # Best result record
best_model = None    # Best model weights
best_hp = None       # Best hyperparameters

# ============================================================================
# INITIALIZE RESULTS CSV
# ============================================================================

# Create CSV file with header
with open(RESULTS_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "name", "hidden", "act", "dropout", "batchnorm", "lr", 
        "momentum", "step_size", "gamma", "batch_size",
        "epochs_trained", "val_mse", "val_pearson", "train_time_sec"
    ])
    w.writeheader()

# ============================================================================
# RUN ALL TRIALS
# ============================================================================

for t in range(N_TRIALS):
    # Sample hyperparameters for this trial
    hp = sample_hp(t)
    print(f"\n=== Trial {t+1}/{N_TRIALS}: {hp} ===")
    
    # Train the model
    rec, model, curves = train_once(hp, trial_seed=SEED + t)
    
    # Append results to CSV immediately (don't lose data if crash occurs)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name", "hidden", "act", "dropout", "batchnorm", "lr",
            "momentum", "step_size", "gamma", "batch_size",
            "epochs_trained", "val_mse", "val_pearson", "train_time_sec"
        ])
        w.writerow(rec)
    
    # Store results in memory
    results.append(rec)
    curves_by_name[rec["name"]] = curves
    
    # Update best model if this one is better
    if best_rec is None or rec["val_mse"] < best_rec["val_mse"]:
        best_rec = rec
        best_model = model
        best_hp = hp
        
        # Save best model to disk
        torch.save({
            "state_dict": best_model.state_dict(),  # Model weights
            "hparams": {k: v for k, v in hp.items()},  # Hyperparameters
            "X_mean": X_mean,  # Preprocessing statistics (needed for deployment)
            "X_std": X_std,
            "meta": {"G": G, "I": I, "seed": SEED}  # Metadata
        }, BEST_MODEL_PT)
        
        print(f"[BEST] Updated best by Val: {rec['name']} (val_mse={rec['val_mse']:.6f})")

# ============================================================================
# FINAL EVALUATION ON TEST SET (Best Model Only)
# ============================================================================

"""
IMPORTANT: We only evaluate the test set ONCE, at the very end, using the
model that performed best on validation. This prevents "information leakage"
where test performance influences model selection.

This is the correct way to assess generalization performance.
"""

print("\n[INFO] Evaluating test set for the best-by-val model only...")

best_bs = best_hp["batch_size"]
best_model.eval()

# Generate predictions for entire test set
preds = []
with torch.no_grad():
    for xb, _ in batch_iter(te_idx_t, batch_size=best_bs, shuffle=False):
        with torch.cuda.amp.autocast(enabled=AMP):
            preds.append(best_model(xb))

# Get ground truth and predictions
Y_test_t = Yt[te_idx_t]  # True values (on GPU)
Y_pred_t = torch.cat(preds, dim=0)  # Concatenate all prediction batches

# Compute test metrics
test_mse = float(((Y_test_t - Y_pred_t)**2).mean().item())
test_r = pearson_mean_gpu(Y_test_t, Y_pred_t)

# Add test metrics to best model record
best_rec["test_mse"] = test_mse
best_rec["test_pearson"] = test_r

print(f"[BEST on TEST] act={best_hp['act']} | MSE: {test_mse:.6f} | r: {test_r:.4f}")

# ============================================================================
# GENERATE SUMMARY FILES
# ============================================================================

# Sort all results by validation MSE (best first)
results_sorted = sorted(results, key=lambda r: r["val_mse"])

# Create summary dictionary
summary = {
    "best": best_rec,  # Best model info
    "top5": results_sorted[:5],  # Top 5 models
    "n_trials": N_TRIALS,
    "csv": RESULTS_CSV,
    "best_model_pt": BEST_MODEL_PT
}

# Save summary as JSON
with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SEARCH COMPLETE ===")
print(json.dumps(summary, indent=2))

# ============================================================================
# SUMMARY VISUALIZATIONS
# ============================================================================

# -------------------------
# 1. Bar Chart: All Models
# -------------------------
try:
    import matplotlib.pyplot as plt
    
    names = [r["name"] for r in results_sorted]
    vals = [r["val_mse"] for r in results_sorted]
    
    # Dynamic figure width based on number of models
    plt.figure(figsize=(max(8, 0.4*len(names)), 4.8))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("Validation MSE")
    plt.title("Validation MSE by model (activation swept)")
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_BAR, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot summary bar: {e}")

# -------------------------
# 2. Line Plot: Top 5 Models
# -------------------------
try:
    import matplotlib.pyplot as plt
    
    top5 = results_sorted[:5]
    plt.figure(figsize=(8, 5))
    
    for r in top5:
        c = curves_by_name[r["name"]]
        
        # Validation curve (evaluated only some epochs)
        xs_v = [e for e in range(1, len(c["train"])+1) 
                if e == 1 or e % EVAL_EVERY == 0][:len(c["val"])]
        plt.plot(xs_v, c["val"], label=f"{r['name']} (val)", linewidth=2.0)
        
        # Training curve (every epoch, faded)
        plt.plot(range(1, len(c["train"])+1), c["train"], alpha=0.4, linewidth=1.0)
    
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Top-5 models — validation curves")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_TOP5, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot top-5 overlay: {e}")

print("[INFO] All done.")