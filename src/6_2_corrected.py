import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

data = np.load("picking_time_data.npz")
X_all, y_all = data["X"], data["y"]
# Shuffle and split: 20% held-out test, 80% for cross-validation
n = len(X_all)
idx = np.random.default_rng(0).permutation(n)
n_test = int(0.2 * n)
X_test_raw = X_all[idx[:n_test]]
y_test_raw = y_all[idx[:n_test]]
X_cv_raw = X_all[idx[n_test:]]
y_cv_raw = y_all[idx[n_test:]]
print(f"Cross-validation pool: {len(X_cv_raw)} examples")
print(f"Held-out test set:     {len(X_test_raw)} examples")


def normalize(X_train, y_train, X_val, y_val):
    """Normalize features and targets using training-set statistics."""
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(), y_train.std()
    return (
        (X_train - X_mean) / X_std,  # normalized train features
        (y_train - y_mean) / y_std,  # normalized train targets
        (X_val - X_mean) / X_std,  # normalized val features (same stats!)
        (y_val - y_mean) / y_std,  # normalized val targets
        X_mean,
        X_std,
        y_mean,
        y_std,  # save stats for denormalization later
    )


def make_loader(X, y, batch_size=32, shuffle=True):
    """Wrap NumPy arrays in a PyTorch DataLoader."""
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),  # (N,) -> (N,1)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_model(n_hidden_layers=3):
    """Create a feedforward network with variable depth.
    Uses widths [32, 16, 8], truncated to n_hidden_layers.
    This lets us compare depths in the experiment below.
    """
    widths = [32, 16, 8][:n_hidden_layers]
    layers = []
    d_in = 5  # number of input features
    for w in widths:
        layers.append(nn.Linear(d_in, w))  # linear transform: W @ a + b
        layers.append(nn.ReLU())  # element-wise activation
        d_in = w
    layers.append(nn.Linear(d_in, 1))  # output layer (no activation)
    return nn.Sequential(*layers)


def train_model(model, train_loader, X_val_t, y_val_t, epochs=200, lr=1e-3):
    """Train a model and return per-epoch (train_losses, val_losses)."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        # --- Training phase ---
        model.train()  # set training mode
        epoch_loss, n_batches = 0.0, 0
        for X_batch, y_batch in train_loader:
            pred = model(X_batch)  # 1. forward pass
            loss = criterion(pred, y_batch)  # 2. compute MSE loss
            optimizer.zero_grad()  # 3. clear previous gradients
            loss.backward()  # 4. backpropagation
            optimizer.step()  # 5. Adam parameter update
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)
        # --- Validation phase (no gradient computation) ---
        model.eval()  # set evaluation mode
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)
    return train_losses, val_losses


K = 5  # number of folds
fold_size = len(X_cv_raw) // K
fold_indices = [np.arange(i * fold_size, (i + 1) * fold_size) for i in range(K)]
# Assign remainder examples to the last fold
if K * fold_size < len(X_cv_raw):
    fold_indices[-1] = np.arange((K - 1) * fold_size, len(X_cv_raw))
cv_rmses = []
cv_train_curves = []
cv_val_curves = []
for fold in range(K):
    # --- Split this fold ---
    val_idx = fold_indices[fold]
    train_idx = np.concatenate([fold_indices[j] for j in range(K) if j != fold])
    # --- Normalize (fit on training fold only) ---
    (X_tr_n, y_tr_n, X_val_n, y_val_n, X_mean, X_std, y_mean, y_std) = normalize(
        X_cv_raw[train_idx],
        y_cv_raw[train_idx],
        X_cv_raw[val_idx],
        y_cv_raw[val_idx],
    )
    # --- Build DataLoader and tensors ---
    train_loader = make_loader(X_tr_n, y_tr_n)
    X_val_t = torch.tensor(X_val_n, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_n, dtype=torch.float32).unsqueeze(1)
    # --- Train this fold ---
    torch.manual_seed(fold)  # reproducible initialization per fold
    model = make_model(n_hidden_layers=3)
    train_losses, val_losses = train_model(model, train_loader, X_val_t, y_val_t)
    cv_train_curves.append(train_losses)
    cv_val_curves.append(val_losses)
    # --- Compute RMSE in original units ---
    model.eval()
    with torch.no_grad():
        pred_norm = model(X_val_t).squeeze().numpy()
    pred_orig = pred_norm * y_std + y_mean  # denormalize
    fold_rmse = np.sqrt(np.mean((y_cv_raw[val_idx] - pred_orig) ** 2))
    cv_rmses.append(fold_rmse)
    print(f"Fold {fold + 1}: val RMSE = {fold_rmse:.2f} sec")
print(f"\nCV RMSE: {np.mean(cv_rmses):.2f} +/- {np.std(cv_rmses):.2f} sec")

# Normalize using full CV pool statistics
(X_cv_n, y_cv_n, X_test_n, y_test_n, X_mean, X_std, y_mean, y_std) = normalize(
    X_cv_raw,
    y_cv_raw,
    X_test_raw,
    y_test_raw,
)
full_loader = make_loader(X_cv_n, y_cv_n)
X_test_t = torch.tensor(X_test_n, dtype=torch.float32)
y_test_t = torch.tensor(y_test_n, dtype=torch.float32).unsqueeze(1)
torch.manual_seed(0)
final_model = make_model(n_hidden_layers=3)
train_losses, test_losses = train_model(
    final_model,
    full_loader,
    X_test_t,
    y_test_t,
)
# Denormalize predictions and compute final RMSE
final_model.eval()
with torch.no_grad():
    test_pred_norm = final_model(X_test_t).squeeze().numpy()
test_pred_orig = test_pred_norm * y_std + y_mean
nn_rmse = np.sqrt(np.mean((y_test_raw - test_pred_orig) ** 2))
print(f"Final test RMSE (neural network): {nn_rmse:.2f} sec")

# OLS via normal equation: w = (X^T X)^{-1} X^T y
X_cv_aug = np.column_stack([X_cv_n, np.ones(len(X_cv_n))])  # add bias column
X_test_aug = np.column_stack([X_test_n, np.ones(len(X_test_n))])
w_lin = np.linalg.lstsq(X_cv_aug, y_cv_n, rcond=None)[0]
# Predict and denormalize
y_pred_lin_norm = X_test_aug @ w_lin
y_pred_lin_orig = y_pred_lin_norm * y_std + y_mean
lin_rmse = np.sqrt(np.mean((y_test_raw - y_pred_lin_orig) ** 2))
print(f"Linear baseline test RMSE: {lin_rmse:.2f} sec")
print(f"Neural network test RMSE:  {nn_rmse:.2f} sec")
print(f"MSE improvement: {(1 - (nn_rmse / lin_rmse) ** 2) * 100:.1f}%")
