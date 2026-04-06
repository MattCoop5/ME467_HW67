import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

data = np.load("picking_time_data.npz")
X, y = data["X"], data["y"]
feature_names = list(data["feature_names"])


def _safe_std(values: np.ndarray) -> np.ndarray:
    """Prevent divide-by-zero when a feature/target has zero variance."""
    std = np.asarray(values.std(axis=0))
    return np.where(std == 0, 1.0, std)


def make_fold_dataloaders(
    X_all: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, dict[str, np.ndarray]]:
    """Create normalized train/val DataLoaders for one fold.

    Normalization is fit on the fold's training data only:
    - Features: zero mean, unit variance
    - Targets: zero mean, unit variance
    """
    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]

    X_mean = X_train.mean(axis=0)
    X_std = _safe_std(X_train)
    y_mean = y_train.mean(axis=0)
    y_std = _safe_std(np.atleast_1d(y_train))

    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std

    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_norm, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    stats = {
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": np.atleast_1d(y_mean),
        "y_std": np.atleast_1d(y_std),
    }

    return train_loader, val_loader, stats


def make_all_folds_dataloaders(
    X_all: np.ndarray,
    y_all: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    batch_size: int = 32,
) -> list[tuple[DataLoader, DataLoader, dict[str, np.ndarray]]]:
    """Build normalized DataLoaders for every fold split."""
    fold_loaders = []
    for train_idx, val_idx in fold_indices:
        fold_loaders.append(
            make_fold_dataloaders(
                X_all=X_all,
                y_all=y_all,
                train_idx=train_idx,
                val_idx=val_idx,
                batch_size=batch_size,
            )
        )
    return fold_loaders


def fit_linear_regression_normal_equation(
    X_train_norm: np.ndarray,
    y_train_norm: np.ndarray,
    X_test_norm: np.ndarray,
    y_test_orig: np.ndarray,
    y_mean: float | np.ndarray,
    y_std: float | np.ndarray,
) -> float:
    """Fit linear regression via normal equation on normalized data.

    Returns test RMSE in original units after denormalization.

    Args:
        X_train_norm: normalized training features
        y_train_norm: normalized training targets
        X_test_norm: normalized test features
        y_test_orig: original (non-normalized) test targets
        y_mean: mean used for target normalization
        y_std: std used for target normalization
    """
    X_train_aug = np.column_stack([np.ones(X_train_norm.shape[0]), X_train_norm])
    beta = np.linalg.lstsq(X_train_aug, y_train_norm, rcond=None)[0]

    X_test_aug = np.column_stack([np.ones(X_test_norm.shape[0]), X_test_norm])
    y_test_pred_norm = X_test_aug @ beta

    y_test_pred_orig = y_test_pred_norm * y_std + y_mean

    rmse = np.sqrt(np.mean((y_test_pred_orig - y_test_orig) ** 2))
    return rmse


class FeedforwardNN(nn.Module):
    """Feedforward network: 32 -> 16 -> 8 hidden units with ReLU."""

    def __init__(self, input_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VariableDepthNN(nn.Module):
    """Feedforward network with variable number of hidden layers.

    Args:
        input_dim: input feature dimension
        hidden_dims: list of hidden layer sizes, e.g. [32, 16, 8]
        output_dim: output dimension (default 1)
    """

    def __init__(
        self, input_dim: int, hidden_dims: list[int], output_dim: int = 1
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def compare_depths_cv(
    X_all: np.ndarray,
    y_all: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    depths: list[list[int]],
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str | torch.device | None = None,
) -> dict[str, list[float]]:
    """Compare network architectures via cross-validation.

    Args:
        X_all: all features
        y_all: all targets
        fold_indices: list of (train_idx, val_idx) tuples
        depths: list of hidden_dims lists, e.g. [[32], [32, 16], [32, 16, 8]]
        epochs: training epochs per fold
        lr: learning rate
        batch_size: DataLoader batch size
        device: torch device

    Returns:
        dict with key per depth and values = CV validation losses across folds
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {str(d): [] for d in depths}

    for train_idx, val_idx in fold_indices:
        train_loader, val_loader, stats = make_fold_dataloaders(
            X_all, y_all, train_idx, val_idx, batch_size
        )

        X_train_norm = X_all[train_idx]
        X_train_mean = X_train_norm.mean(axis=0)
        X_train_std = _safe_std(X_train_norm)
        X_train_norm = (X_all[train_idx] - X_train_mean) / X_train_std

        input_dim = X_train_norm.shape[1]

        for hidden_dims in depths:
            model = VariableDepthNN(input_dim, hidden_dims, output_dim=1)
            history = train_model(
                model, train_loader, val_loader, epochs=epochs, lr=lr, device=device
            )
            final_val_loss = history["val_loss"][-1]
            results[str(hidden_dims)].append(final_val_loss)

    return results


def summarize_cv_results(
    cv_results: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Summarize CV results: mean and std validation loss per depth."""
    summary = {}
    for depth_str, losses in cv_results.items():
        summary[depth_str] = {
            "mean_val_loss": np.mean(losses),
            "std_val_loss": np.std(losses),
            "folds": losses,
        }
    return summary


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
) -> dict[str, list[float]]:
    """Train with Adam + MSE and track train/validation loss per epoch."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            preds = model(x_batch).float()

            if preds.ndim > y_batch.ndim:
                y_batch = y_batch.unsqueeze(-1)

            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = x_batch.size(0)
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        epoch_train_loss = train_loss_sum / max(train_samples, 1)

        model.eval()
        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).float()
                preds = model(x_batch).float()

                if preds.ndim > y_batch.ndim:
                    y_batch = y_batch.unsqueeze(-1)

                loss = criterion(preds, y_batch)
                batch_size = x_batch.size(0)
                val_loss_sum += loss.item() * batch_size
                val_samples += batch_size

        epoch_val_loss = val_loss_sum / max(val_samples, 1)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

    return history


class ThreeClassCNN(nn.Module):
    """CNN with 3 conv blocks and 3-class softmax output.

    Architecture:
    - Conv2d -> ReLU -> MaxPool2d (16 filters)
    - Conv2d -> ReLU -> MaxPool2d (32 filters)
    - Conv2d -> ReLU -> MaxPool2d (64 filters)
    - AdaptiveAvgPool2d -> Linear(64 -> 3) -> Softmax
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = self.softmax(logits)
        return probs
