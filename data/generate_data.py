"""
Quantum sensor dataset generation.

Class 0 (Normal):   damped oscillation at low resonant frequency
Class 1 (Anomaly):  damped oscillation at higher frequency + harmonic distortion

This gives well-separated classes that are still non-trivially learnable,
unlike the original near-identical phase-shifted sinusoids.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def generate_sensor_data(
    n_samples: int = 800,
    n_features: int = 8,
    snr: float = 3.0,
    seed: int = 42,
) -> tuple:
    """
    Returns (X, y) where:
      X: shape (n_samples, n_features)  — scaled to [0, pi] for ZZ encoding
      y: shape (n_samples,)             — binary labels {0, 1}

    Class 0 — normal bearing signal:
        A * exp(-zeta * t) * cos(omega_0 * t)
        omega_0 = 2*pi*50 Hz (low resonant frequency)

    Class 1 — fault bearing signal:
        A * exp(-zeta * t) * cos(omega_1 * t) + 0.4 * cos(3 * omega_1 * t)
        omega_1 = 2*pi*180 Hz (higher frequency + harmonic distortion)

    Both classes have additive Gaussian noise controlled by snr.
    """
    rng = np.random.default_rng(seed)

    t = np.linspace(0, 0.02, n_features)  # 20 ms window

    omega_0 = 2 * np.pi * 50    # 50 Hz normal
    omega_1 = 2 * np.pi * 180   # 180 Hz fault
    zeta = 80.0                  # damping coefficient
    A = 1.0

    half = n_samples // 2
    X_list = []
    y_list = []

    # Class 0: normal
    for _ in range(half):
        signal = A * np.exp(-zeta * t) * np.cos(omega_0 * t)
        signal_power = np.mean(signal ** 2)
        noise_std = np.sqrt(signal_power / snr)
        noise = rng.normal(0.0, noise_std, n_features)
        X_list.append(signal + noise)
        y_list.append(0)

    # Class 1: anomaly (different frequency + harmonic)
    for _ in range(n_samples - half):
        signal = (
            A * np.exp(-zeta * t) * np.cos(omega_1 * t)
            + 0.4 * np.cos(3 * omega_1 * t)
        )
        signal_power = np.mean(signal ** 2)
        noise_std = np.sqrt(signal_power / snr)
        noise = rng.normal(0.0, noise_std, n_features)
        X_list.append(signal + noise)
        y_list.append(1)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=int)

    # Shuffle
    idx = rng.permutation(n_samples)
    X, y = X[idx], y[idx]

    # Scale to [0, pi] for angle/ZZ encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    return X, y


def load_and_split(config: dict, seed: int = 42) -> tuple:
    """
    Convenience wrapper: generates data and returns stratified splits.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    d = config["data"]
    X, y = generate_sensor_data(
        n_samples=d["n_samples"],
        n_features=d["n_features"],
        snr=d.get("signal_snr", 3.0),
        seed=seed,
    )

    test_split = d.get("test_split", 0.20)
    val_split = d.get("val_split", 0.15)

    # First carve out test set
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )

    # Then carve out val set from train+val
    val_frac_of_tv = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac_of_tv, random_state=seed, stratify=y_tv
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# Backward-compatible shim used by old scripts
def generate_data(samples: int = 200, features: int = 6) -> tuple:
    X, y = generate_sensor_data(n_samples=samples, n_features=features)
    return X, y
