from pathlib import Path

import numpy as np


def generate_dataset(
    n_samples: int = 400, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic warehouse picking-time data.

    Features (in order):
    1) distance (m)
    2) load (kg)
    3) congestion (robots in aisle)
    4) battery level (%)
    5) aisle width (m)
    """
    rng = np.random.default_rng(seed)

    distance_m = rng.uniform(5.0, 120.0, n_samples)
    load_kg = rng.uniform(0.5, 25.0, n_samples)
    congestion_robots = rng.integers(0, 13, n_samples).astype(float)
    battery_level = rng.uniform(15.0, 100.0, n_samples)
    aisle_width_m = rng.uniform(1.2, 5.0, n_samples)

    X = np.column_stack(
        [distance_m, load_kg, congestion_robots, battery_level, aisle_width_m]
    ).astype(np.float32)

    # Nonlinear ground-truth relationship + noise
    noise = rng.normal(0.0, 3.5, n_samples)
    y = (
        18.0
        + 0.33 * distance_m
        + 1.15 * load_kg
        + 2.1 * congestion_robots
        - 0.10 * battery_level
        - 3.8 * aisle_width_m
        + 0.010 * distance_m * congestion_robots
        + 0.035 * (load_kg**2)
        + 16.0 / aisle_width_m
        + noise
    ).astype(np.float32)

    feature_names = np.array(
        [
            "distance_m",
            "load_kg",
            "congestion_robots",
            "battery_level",
            "aisle_width_m",
        ],
        dtype=str,
    )

    return X, y, feature_names


def save_dataset(output_path: str | Path = "picking_time_data.npz") -> Path:
    output_path = Path(output_path)
    X, y, feature_names = generate_dataset(n_samples=400, seed=42)
    np.savez(output_path, X=X, y=y, feature_names=feature_names)
    return output_path


def load_dataset(
    path: str | Path = "picking_time_data.npz",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load dataset exactly in assignment format."""
    data = np.load(path, allow_pickle=True)
    X, y = data["X"], data["y"]
    feature_names = list(data["feature_names"])
    return X, y, feature_names


if __name__ == "__main__":
    root_npz = Path(__file__).resolve().parents[1] / "picking_time_data.npz"
    save_dataset(root_npz)

    # Load it as follows:
    data = np.load(root_npz, allow_pickle=True)
    X, y = data["X"], data["y"]
    feature_names = list(data["feature_names"])

    print(f"Saved: {root_npz}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"features: {feature_names}")
