import torch
import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import nn
from joblib import Parallel, delayed
from tqdm import tqdm

from causalflows.flows import CausalMAF, CausalNSF
from causalflows.scms import SCM, CausalEquations

# Define helper functions first
TensorLike = nn.Module  # For type hinting, can be torch.Tensor or Dataset

def _make_loader(data, batch_size, shuffle):
    if isinstance(data, torch.Tensor):
        data = TensorDataset(data)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

@torch.no_grad()
def _eval_nll(flow, loader, device):
    flow.eval()
    total, count = 0.0, 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        nll = -flow().log_prob(x)
        total += float(nll.sum().item())
        count += x.shape[0]
    return total / max(count, 1)

def fit(
    flow: nn.Module,
    train_data,
    val_data=None,
    *,
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 100,
    device: torch.device = None,
    print_every: int = 1,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 1e-4,
):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow.to(dev)
    train_loader = _make_loader(train_data, batch_size, shuffle=True)
    val_loader = _make_loader(val_data, batch_size, shuffle=False) if val_data is not None else None
    optim = torch.optim.Adam(flow.parameters(), lr=lr)
    history = {
        "train_nll": [],
        "val_nll": [] if val_loader is not None else None,
        "stopped_early": False,
        "best_epoch": 0
    }
    best_val_nll = float('inf')
    patience_counter = 0
    best_model_state = None
    for epoch in tqdm(range(1, epochs + 1), disable=True):
        flow.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(dev, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = -flow().log_prob(x).mean()
            loss.backward()
            optim.step()
            running += float(loss.item()) * x.size(0)
            seen += x.size(0)
        train_nll = running / max(seen, 1)
        history["train_nll"].append(train_nll)
        if val_loader is not None:
            val_nll = _eval_nll(flow, val_loader, dev)
            history["val_nll"].append(val_nll)
            if early_stopping_patience > 0:
                if val_nll < best_val_nll - early_stopping_min_delta:
                    best_val_nll = val_nll
                    patience_counter = 0
                    history["best_epoch"] = epoch
                    best_model_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n[{epoch:03d}/{epochs:03d}] Early stopping triggered!")
                    print(f"   train_nll={train_nll:.4f}  val_nll={val_nll:.4f}")
                    print(f"   Best epoch: {history['best_epoch']}, Best val_nll: {best_val_nll:.4f}")
                    history["stopped_early"] = True
                    if best_model_state is not None:
                        flow.load_state_dict(best_model_state)
                    break
            if epoch % print_every == 0:
                patience_str = f"  patience={patience_counter}/{early_stopping_patience}" if early_stopping_patience > 0 else ""
                print(f"[{epoch:03d}/{epochs:03d}] train_nll={train_nll:.4f}  val_nll={val_nll:.4f}{patience_str}")
        else:
            if epoch % print_every == 0:
                print(f"[{epoch:03d}/{epochs:03d}] train_nll={train_nll:.4f}")
    return history

# Load data from scratch (original)
data_df = pd.read_csv("TFM_dataset.csv")
data_df = data_df.drop(columns=['randid'])

# Fill missing BMI values with mode
bmi_mode = data_df['bmi'].mode()[0]
data_df['bmi'] = data_df['bmi'].fillna(bmi_mode)

# Identify binary/continuous variables (once)
binary_vars = []
for col in data_df.columns:
    unique_vals = sorted(data_df[col].unique())
    if len(unique_vals) == 2 and set(unique_vals).issubset({0.0, 1.0}):
        binary_vars.append(col)

continuous_vars = [col for col in data_df.columns if col not in binary_vars]

# Split data (before any processing)
diabetes_labels = data_df['diabetes mellitus'].values

# First split: separate test set (10%)
train_val_idx, test_idx = train_test_split(
    np.arange(len(data_df)),
    test_size=0.1,
    stratify=diabetes_labels,
    random_state=42
)

# Second split: separate validation set from training 
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.11,  # ~10% of total for validation
    stratify=diabetes_labels[train_val_idx],
    random_state=42
)

# Configuration
cfg_balanced = {
    'hidden_dims': [32, 32],
    'lr': 1e-3,
    'batch_size': 256,
    'epochs': 1000,
    'early_stopping_patience': 100,
    'early_stopping_min_delta': 1e-4,
}

context = 0

# Create directories for saving models

os.makedirs('saved_models_validation', exist_ok=True)

print("Training Ground Truth Flow Model")

# load ground truth graph
with open("ground_truth_graph.json", 'r', encoding='utf-8') as f:
    ground_truth_data = json.load(f)

adjacency_gt = np.array(ground_truth_data['adjacency_matrix'])
np.fill_diagonal(adjacency_gt, 1)
adjacency_gt = torch.tensor(adjacency_gt, dtype=torch.bool)

gt_variables = ground_truth_data['variables']


# 1. Reorder to GT variable order
data_gt = data_df[gt_variables].copy()

# 2. FIT_TRANSFORM scaler on ALL data (same as selected graphs)
scaler_gt = StandardScaler()
data_norm_gt = data_gt.copy()
gt_continuous = [v for v in gt_variables if v in continuous_vars]

if len(gt_continuous) > 0:
    gt_cont_indices = [gt_variables.index(v) for v in gt_continuous]
    # FIT_TRANSFORM on all data
    data_norm_gt.iloc[:, gt_cont_indices] = scaler_gt.fit_transform(data_gt[gt_continuous])

# 3. Add noise to binary variables
np.random.seed(42)
for col in binary_vars:
    if col in gt_variables:
        noise_std = 0.03 if (data_df[col] == 1).mean() < 0.05 else 0.05
        noise = np.random.normal(0, noise_std, size=len(data_df))
        data_norm_gt[col] = data_gt[col].astype(float) + noise

# 4. Create tensors (train, val, test)
data_tensor_gt = torch.tensor(data_norm_gt.values, dtype=torch.float32)
train_data_gt = data_tensor_gt[train_idx]
val_data_gt = data_tensor_gt[val_idx]
test_data_gt = data_tensor_gt[test_idx]

# 5. Train GT model with early stopping
features = len(gt_variables)
flow_gt = CausalNSF(
    features,
    context,
    adjacency=adjacency_gt,
    hidden_features=cfg_balanced['hidden_dims']
)

history_gt = fit(
    flow_gt,
    train_data_gt,
    val_data=val_data_gt,
    lr=cfg_balanced['lr'],
    batch_size=cfg_balanced['batch_size'],
    epochs=cfg_balanced['epochs'],
    print_every=200,
    early_stopping_patience=cfg_balanced['early_stopping_patience'],
    early_stopping_min_delta=cfg_balanced['early_stopping_min_delta'],
)

# 6. Evaluate on test set (final evaluation)
test_nll_gt = _eval_nll(flow_gt, _make_loader(test_data_gt, cfg_balanced['batch_size'], False),
                        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Print summary for GT model
if history_gt.get('stopped_early', False):
    print(f"Ground Truth Model: Early stopping at epoch {len(history_gt['train_nll'])}")
    print(f"  Best epoch: {history_gt['best_epoch']}, Best val_nll: {min(history_gt['val_nll']):.4f}")
else:
    print(f"Ground Truth Model: Completed all {cfg_balanced['epochs']} epochs")
print(f"  Test NLL: {test_nll_gt:.4f}")

# Save ground truth model
model_path_gt = 'saved_models_validation/flow_ground_truth.pt'
torch.save({
    'model_state_dict': flow_gt.state_dict(),
    'adjacency': adjacency_gt,
    'variables': gt_variables,
    'features': len(gt_variables),
    'config': cfg_balanced,
    'history': history_gt,
    'test_nll': test_nll_gt,
    'scaler': scaler_gt,
}, model_path_gt)
print(f"Saved ground truth model to {model_path_gt}")

print("\nGround Truth Model Training Complete\n")


### IMPORT OUT KNOWLEDGE GRAPHS 
with open("new_ordered_graphs.json", 'r', encoding='utf-8') as f:
    loaded_graphs = json.load(f)
    loaded_graphs = loaded_graphs[0:101]


### IMPORT DATASET AND CLEANING
data_df = pd.read_csv("TFM_dataset.csv")
data_df = data_df.drop(columns=['randid'])
bmi_mode = data_df['bmi'].mode()[0]
data_df['bmi'] = data_df['bmi'].fillna(bmi_mode)

binary_vars = []
for col in data_df.columns:
    unique_vals = sorted(data_df[col].unique())
    if len(unique_vals) == 2 and set(unique_vals).issubset({0.0, 1.0}):
        binary_vars.append(col)
continuous_vars = [col for col in data_df.columns if col not in binary_vars]
diabetes_labels = data_df['diabetes mellitus'].values
# Note: Using train_idx, val_idx, test_idx already defined at the beginning of the script

# Configuration
context = 0
cfg_balanced = {
    'hidden_dims': [64, 64],
    'lr': 5e-4,
    'batch_size': 128,
    'epochs': 2000,
    'early_stopping_patience': 100,
    'early_stopping_min_delta': 1e-4,
}

os.makedirs('saved_models_validation', exist_ok=True)

def train_and_save_selected_graph(graph_idx, selected_graph):
    print(f"Processing Graph {graph_idx + 1}/{len(loaded_graphs)}")
    variables = selected_graph['variables']
    adjacency_matrix = selected_graph['adjacency_matrix']
    adj_array = np.array(adjacency_matrix)
    np.fill_diagonal(adj_array, 1)
    adjacency = torch.tensor(adj_array, dtype=torch.bool)
    # Reorder to selected graph variable order
    data_selected = data_df[variables].copy()

   # FIT_TRANSFORM scaler
    scaler_selected = StandardScaler()
    data_norm_selected = data_selected.copy()
    sel_continuous = [v for v in variables if v in continuous_vars]
    if len(sel_continuous) > 0:
        sel_cont_indices = [variables.index(v) for v in sel_continuous]
        data_norm_selected.iloc[:, sel_cont_indices] = scaler_selected.fit_transform(data_selected[sel_continuous])
    
    # Add noise to binary variables (same seed for consistency)
    np.random.seed(42)
    for col in binary_vars:
        if col in variables:
            noise_std = 0.03 if (data_df[col] == 1).mean() < 0.05 else 0.05
            noise = np.random.normal(0, noise_std, size=len(data_df))
            data_norm_selected[col] = data_selected[col].astype(float) + noise

    # Create tensors (train, val, test)
    data_tensor = torch.tensor(data_norm_selected.values, dtype=torch.float32)
    train_data = data_tensor[train_idx]
    val_data = data_tensor[val_idx]
    test_data = data_tensor[test_idx]
    
    flow = CausalNSF(
        len(variables),
        context,
        adjacency=adjacency,
        hidden_features=cfg_balanced['hidden_dims']
    )

    # Train with early stopping using validation set
    history = fit(
        flow,
        train_data,
        val_data=val_data,
        lr=cfg_balanced['lr'],
        batch_size=cfg_balanced['batch_size'],
        epochs=cfg_balanced['epochs'],
        print_every=200,
        early_stopping_patience=cfg_balanced['early_stopping_patience'],
        early_stopping_min_delta=cfg_balanced['early_stopping_min_delta'],
    )

    # Evaluate on test set (final evaluation)
    test_nll = _eval_nll(flow, _make_loader(test_data, cfg_balanced['batch_size'], False),
                        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Print summary
    if history.get('stopped_early', False):
        print(f"Graph {graph_idx}: Early stopping at epoch {len(history['train_nll'])}")
        print(f"  Best epoch: {history['best_epoch']}, Best val_nll: {min(history['val_nll']):.4f}")
    else:
        print(f"Graph {graph_idx}: Completed all {cfg_balanced['epochs']} epochs")
    print(f"  Test NLL: {test_nll:.4f}")

    # Save model with its scaler
    model_path_selected = f'saved_models_validation/flow_selected_graph_{graph_idx}.pt'
    torch.save({
        'model_state_dict': flow.state_dict(),
        'adjacency': adjacency,
        'variables': variables,
        'features': len(variables),
        'config': cfg_balanced,
        'history': history,
        'test_nll': test_nll,
        'scaler': scaler_selected,
    }, model_path_selected)
    print(f"Saved selected graph model to {model_path_selected}\n")

# Use Parallel to train models for all selected graphs 2 n_jobs to use 2 cores locally, in winscp use 13 cores
Parallel(n_jobs=13)(
    delayed(train_and_save_selected_graph)(graph_idx, selected_graph)
    for graph_idx, selected_graph in enumerate(loaded_graphs)
)

print("All models trained and saved.")
