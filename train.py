"""
Main training script for the Noise-Resilient VQC Sensor Classifier.

Fixes over v1:
  - Metadata bug fixed: 'features' key now uses actual n_qubits value
  - Noise model properly injected into VQC sampler
  - SPSA optimizer with convergence history tracking
  - 5-fold CV with full metrics
  - Config-driven (loads config/base_config.yaml)
  - CLI flags: --no-noise, --max-iter, --seed, --config

Usage:
  python train.py                            # default config, noisy VQC
  python train.py --no-noise                 # ideal VQC baseline
  python train.py --max-iter 100 --seed 123  # quick test run
  python train.py --config config/base_config.yaml
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import yaml

from sklearn.model_selection import StratifiedKFold

from data.generate_data import generate_sensor_data
from models.vqc_model import create_vqc
from noise.noise_model import (
    noise_model_from_config,
    build_noisy_sampler_and_pm,
    build_ideal_sampler_and_pm,
)
from evaluation.metrics import compute_metrics, aggregate_cv_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train VQC Sensor Classifier")
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--no-noise", action="store_true",
                        help="Use ideal (noiseless) sampler")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Override SPSA max iterations")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def train(config: dict, use_noise: bool = True, seed: int = None) -> dict:
    seed = seed or config["experiment"]["seeds"][0]
    n_folds = config["experiment"]["n_folds"]
    n_qubits = config["vqc"]["n_qubits"]
    max_iter = config["vqc"]["max_iter"]
    shots = config["vqc"]["shots"]

    print(f"\n{'='*60}")
    print(f"  VQC Sensor Training  |  noise={'ON' if use_noise else 'OFF'}  "
          f"|  seed={seed}  |  {n_folds}-fold CV")
    print(f"  n_qubits={n_qubits}  reps={config['vqc']['reps']}  "
          f"max_iter={max_iter}  shots={shots}")
    print("=" * 60)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nGenerating sensor dataset ...")
    X, y = generate_sensor_data(
        n_samples=config["data"]["n_samples"],
        n_features=config["data"]["n_features"],
        snr=config["data"].get("signal_snr", 3.0),
        seed=seed,
    )
    print(f"  X shape: {X.shape}  |  class balance: {np.bincount(y).tolist()}")

    # ── Sampler setup ──────────────────────────────────────────────────────────
    if use_noise:
        nm = noise_model_from_config(config)
        sampler, pm = build_noisy_sampler_and_pm(nm, shots=shots, seed=seed)
        print(f"  Noise: depolarizing_2q={config['noise']['depolarizing_2q']}, "
              f"readout={config['noise']['readout_error']}, "
              f"T1={config['noise']['t1_us']}µs")
    else:
        sampler, pm = build_ideal_sampler_and_pm(shots=shots, seed=seed)
        print("  Noise: NONE (ideal simulator)")

    # ── 5-fold cross-validation ───────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    loss_histories = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Track convergence via SPSA callback
        history = []
        def callback(nfev, params, value, metadata, accepted=None):
            history.append(float(value))

        print(f"\n  Fold {fold_i+1}/{n_folds} — training {len(X_tr)} samples ...")
        t0 = time.time()

        vqc = create_vqc(
            n_qubits=n_qubits,
            reps=config["vqc"]["reps"],
            max_iter=max_iter,
            sampler=sampler,
            pass_manager=pm,
            seed=seed + fold_i,
            callback=callback,
        )
        vqc.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        y_pred = vqc.predict(X_val)
        try:
            y_proba = vqc.predict_proba(X_val)
        except Exception:
            y_proba = None

        metrics = compute_metrics(y_val, y_pred, y_proba)
        fold_metrics.append(metrics)
        loss_histories.append(history)

        print(f"    acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
              f"auc={metrics['auc_roc']:.4f}  time={elapsed:.1f}s")

        # Save best-fold model (track global best, not just previous fold)
        if fold_i == 0 or metrics["accuracy"] > fold_metrics[best_fold]["accuracy"]:
            best_vqc = vqc
            best_fold = fold_i

    # ── Aggregate results ──────────────────────────────────────────────────────
    agg = aggregate_cv_metrics(fold_metrics)
    print(f"\n{'─'*50}")
    print("  Cross-Validation Summary:")
    for k, v in agg.items():
        print(f"    {k:<20} {v['mean']:.4f} ± {v['std']:.4f}")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    os.makedirs("saved_models", exist_ok=True)

    with open("saved_models/vqc_params.pkl", "wb") as f:
        pickle.dump(best_vqc.weights, f)

    with open("saved_models/vqc_model.pkl", "wb") as f:
        pickle.dump(best_vqc, f)

    metadata = {
        "n_qubits": n_qubits,          # was bugged as "features": 4
        "n_features": config["data"]["n_features"],
        "n_samples": config["data"]["n_samples"],
        "reps": config["vqc"]["reps"],
        "max_iter": max_iter,
        "use_noise": use_noise,
        "seed": seed,
        "cv_fold_metrics": fold_metrics,
        "cv_mean_accuracy": agg["accuracy"]["mean"],
        "cv_std_accuracy": agg["accuracy"]["std"],
        "cv_mean_f1": agg["f1"]["mean"],
        "cv_mean_auc": agg["auc_roc"]["mean"],
        "loss_histories": loss_histories,
        "best_fold": best_fold,
    }

    with open("saved_models/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n  Saved model to saved_models/vqc_params.pkl")
    print(f"  Saved metadata to saved_models/metadata.pkl")

    # ── Convergence plot ───────────────────────────────────────────────────────
    if any(len(h) > 0 for h in loss_histories):
        from evaluation.visualization import plot_convergence
        import matplotlib.pyplot as plt
        fig = plot_convergence(
            [h for h in loss_histories if len(h) > 0],
            labels=[f"Fold {i+1}" for i in range(len(loss_histories)) if len(loss_histories[i]) > 0],
            output_dir="results",
        )
        plt.close(fig)
        print("  Saved convergence plot to results/convergence_curves.png")

    return metadata


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.max_iter is not None:
        config["vqc"]["max_iter"] = args.max_iter

    train(
        config,
        use_noise=not args.no_noise,
        seed=args.seed,
    )
