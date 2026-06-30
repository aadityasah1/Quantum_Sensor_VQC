"""Physics-inspired vibration-sensor signal generation (raw, unscaled).

Two classes of damped oscillation that stand in for a rolling-element bearing:

    normal (0):  A e^{-zeta t} cos(w0 t)                  w0 = 2*pi*50 Hz
    fault  (1):  A e^{-zeta t} cos(w1 t) + 0.4 cos(3 w1 t) w1 = 2*pi*180 Hz

The fault class adds a third-harmonic term (the usual signature of mechanical
damage). Scaling/PCA is done later by the preprocessing pipeline, so these
functions return the raw signal samples.
"""
from __future__ import annotations
import numpy as np

N_FEATURES = 8          # time samples per signal, over a 20 ms window
SNR = 3.0
_A, _ZETA = 1.0, 80.0
_W0, _W1 = 2 * np.pi * 50, 2 * np.pi * 180


def _one(fault: bool, t: np.ndarray, rng: np.random.Generator, snr: float) -> np.ndarray:
    if fault:
        s = _A * np.exp(-_ZETA * t) * np.cos(_W1 * t) + 0.4 * np.cos(3 * _W1 * t)
    else:
        s = _A * np.exp(-_ZETA * t) * np.cos(_W0 * t)
    noise_std = np.sqrt(np.mean(s ** 2) / snr)
    return s + rng.normal(0.0, noise_std, t.size)


def generate_one(fault: bool = False, snr: float = SNR, seed: int | None = None,
                 n_features: int = N_FEATURES) -> np.ndarray:
    """Return a single raw sensor signal (shape ``(n_features,)``)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 0.02, n_features)
    return _one(bool(fault), t, rng, snr)


def generate_dataset(n_samples: int = 240, snr: float = SNR, seed: int = 42,
                     n_features: int = N_FEATURES):
    """Return ``(X, y)`` of raw signals; balanced and shuffled."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 0.02, n_features)
    half = n_samples // 2
    X = [_one(False, t, rng, snr) for _ in range(half)]
    X += [_one(True, t, rng, snr) for _ in range(n_samples - half)]
    y = [0] * half + [1] * (n_samples - half)
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]
