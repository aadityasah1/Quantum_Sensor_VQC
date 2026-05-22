"""
Quantum noise resilience analysis.

CRITICAL FIX over v1:
  v1 added Gaussian noise to input FEATURES (classical noise, not quantum).
  This version adds actual QUANTUM CIRCUIT NOISE via the AerSampler noise model.

  Each noise level: build a real NoiseModel → inject into AerSampler → train VQC.
  The classical SVM is unaffected (it never runs quantum circuits), so its
  accuracy serves as a flat noise-immune reference line.

Experiments:
  1. run_quantum_noise_sweep  — accuracy vs depolarizing_2q error rate
  2. run_noise_type_ablation  — which channel hurts most?
  3. run_zne_effectiveness    — how much does ZNE recover?
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from data.generate_data import generate_sensor_data
from models.vqc_model import create_vqc
from models.classical_model import create_classical_model
from noise.noise_model import (
    build_noise_model,
    build_noisy_sampler_and_pm,
    build_ideal_sampler_and_pm,
    noise_model_from_config,
)
from evaluation.metrics import compute_metrics, aggregate_cv_metrics
from evaluation.visualization import (
    plot_noise_sweep,
    plot_noise_type_ablation,
)


def _cv_accuracy(model_factory, X, y, n_folds=3, seed=42):
    """
    Run stratified k-fold CV. model_factory() is called fresh each fold.
    Returns (mean_accuracy, std_accuracy).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = model_factory()
        model.fit(X_tr, y_tr)
        accs.append(float(model.score(X_val, y_val)))
    return float(np.mean(accs)), float(np.std(accs))


def run_quantum_noise_sweep(config: dict, output_dir: str = "results") -> pd.DataFrame:
    """
    Sweep depolarizing_2q error rate and measure VQC / SVM accuracy.

    For each noise level:
      - Builds a real quantum NoiseModel
      - Injects it into AerSampler → passed to VQC(sampler=...)
      - Runs 3-fold CV (full 5-fold is slow; use n_folds=3 for sweep)
      - Runs same CV folds for classical SVM (noise-immune reference)
      - Optionally applies ZNE mitigation

    Results saved to results/quantum_noise_sweep.csv
    """
    print("\n" + "=" * 60)
    print("  Quantum Noise Sweep (real circuit noise, not Gaussian)")
    print("=" * 60)

    noise_levels = config["sweep"]["noise_levels"]
    n_qubits = config["vqc"]["n_qubits"]
    max_iter = config["vqc"]["max_iter"]
    shots = config["vqc"]["shots"]
    seed = config["experiment"]["seeds"][0]
    n_folds = 3  # keep sweep fast

    X, y = generate_sensor_data(
        n_samples=config["data"]["n_samples"],
        n_features=config["data"]["n_features"],
        snr=config["data"].get("signal_snr", 3.0),
        seed=seed,
    )

    rows = []

    for noise_level in tqdm(noise_levels, desc="noise sweep"):
        # ── Build quantum noise model at this level ────────────────────────
        nm = build_noise_model(
            depolarizing_1q=config["noise"]["depolarizing_1q"],
            depolarizing_2q=noise_level,
            t1_us=config["noise"]["t1_us"],
            t2_us=config["noise"]["t2_us"],
            readout_error=config["noise"]["readout_error"],
        )
        noisy_sampler, noisy_pm = build_noisy_sampler_and_pm(nm, shots=shots, seed=seed)

        # ── VQC with quantum noise ─────────────────────────────────────────
        def vqc_factory(sampler=noisy_sampler, pm=noisy_pm):
            return create_vqc(
                n_qubits=n_qubits,
                max_iter=max_iter,
                sampler=sampler,
                pass_manager=pm,
                seed=seed,
            )

        print(f"  noise={noise_level:.4f}  training VQC ...", end=" ", flush=True)
        vqc_mean, vqc_std = _cv_accuracy(vqc_factory, X, y, n_folds, seed)
        print(f"acc={vqc_mean:.3f}±{vqc_std:.3f}")

        # ── Classical SVM (quantum noise has no effect) ────────────────────
        svm_mean, svm_std = _cv_accuracy(
            lambda: create_classical_model("svm_rbf", seed), X, y, n_folds, seed
        )

        rows.append({
            "noise_level": noise_level,
            "vqc_mean": vqc_mean,
            "vqc_std": vqc_std,
            "svm_mean": svm_mean,
            "svm_std": svm_std,
        })

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "quantum_noise_sweep.csv"), index=False)

    fig = plot_noise_sweep(df, output_dir=output_dir)
    plt_close(fig)

    print("\nNoise sweep complete.")
    print(df.to_string(index=False))
    return df


def run_noise_type_ablation(config: dict, output_dir: str = "results") -> pd.DataFrame:
    """
    Tests each noise channel IN ISOLATION to find the most damaging one.

    Channels tested:
      ideal         : no noise at all
      depolarizing  : only depolarizing (1Q + 2Q), no T1/T2, no readout
      thermal       : only T1/T2 relaxation, no depolarizing, no readout
      readout       : only readout error, no gate noise
      combined      : all channels together (from config)
    """
    print("\n" + "=" * 60)
    print("  Noise Channel Ablation Study")
    print("=" * 60)

    n_qubits = config["vqc"]["n_qubits"]
    max_iter = config["vqc"]["max_iter"]
    shots = config["vqc"]["shots"]
    seed = config["experiment"]["seeds"][0]
    n_folds = 3

    X, y = generate_sensor_data(
        n_samples=config["data"]["n_samples"],
        n_features=config["data"]["n_features"],
        snr=config["data"].get("signal_snr", 3.0),
        seed=seed,
    )

    profiles = {
        "ideal": build_noise_model(0, 0, 1e9, 1e9, 0),
        "depolarizing_only": build_noise_model(
            config["noise"]["depolarizing_1q"],
            config["noise"]["depolarizing_2q"],
            1e9, 1e9, 0,
        ),
        "thermal_only": build_noise_model(
            0, 0,
            config["noise"]["t1_us"],
            config["noise"]["t2_us"],
            0,
        ),
        "readout_only": build_noise_model(
            0, 0, 1e9, 1e9,
            config["noise"]["readout_error"],
        ),
        "all_combined": noise_model_from_config(config),
    }

    rows = []
    for name, nm in profiles.items():
        sampler, pm = build_noisy_sampler_and_pm(nm, shots=shots, seed=seed)

        def vqc_factory(s=sampler, p=pm):
            return create_vqc(n_qubits=n_qubits, max_iter=max_iter, sampler=s, pass_manager=p, seed=seed)

        print(f"  {name} ...", end=" ", flush=True)
        mean_acc, std_acc = _cv_accuracy(vqc_factory, X, y, n_folds, seed)
        print(f"acc={mean_acc:.3f}±{std_acc:.3f}")
        rows.append({"noise_type": name, "accuracy_mean": mean_acc, "accuracy_std": std_acc})

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "noise_type_ablation.csv"), index=False)

    fig = plot_noise_type_ablation(df, output_dir=output_dir)
    plt_close(fig)

    print("\nAblation results:")
    print(df.to_string(index=False))
    return df


def run_zne_effectiveness(config: dict, output_dir: str = "results") -> dict:
    """
    Compares: no mitigation vs ZNE mitigation under medium noise.
    Returns dict with 'unmitigated' and 'zne' accuracy values.
    """
    print("\n" + "=" * 60)
    print("  ZNE Mitigation Effectiveness")
    print("=" * 60)

    from mitigation.zero_noise_extrapolation import ZNEMitigator
    from sklearn.model_selection import train_test_split

    seed = config["experiment"]["seeds"][0]
    n_qubits = config["vqc"]["n_qubits"]
    max_iter = config["vqc"]["max_iter"]
    shots = config["vqc"]["shots"]
    zne_scales = config["mitigation"].get("zne_scale_factors", [1, 3, 5])

    X, y = generate_sensor_data(
        n_samples=config["data"]["n_samples"],
        n_features=config["data"]["n_features"],
        snr=config["data"].get("signal_snr", 3.0),
        seed=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    nm = noise_model_from_config(config)
    noisy_sampler, noisy_pm = build_noisy_sampler_and_pm(nm, shots=shots, seed=seed)

    print("  Training VQC with medium noise ...")
    vqc = create_vqc(n_qubits=n_qubits, max_iter=max_iter, sampler=noisy_sampler, pass_manager=noisy_pm, seed=seed)
    vqc.fit(X_train, y_train)

    # Unmitigated accuracy
    y_pred_noisy = vqc.predict(X_test)
    acc_unmitigated = float(np.mean(y_pred_noisy == y_test))
    print(f"  Unmitigated accuracy: {acc_unmitigated:.4f}")

    # ZNE mitigated accuracy
    mitigator = ZNEMitigator(
        scale_factors=zne_scales,
        noise_config=config["noise"],
    )
    print(f"  Applying ZNE (scales={zne_scales}) ...")
    y_pred_zne = mitigator.mitigated_predict(vqc, X_test)
    acc_zne = float(np.mean(y_pred_zne == y_test))
    print(f"  ZNE-mitigated accuracy: {acc_zne:.4f}")
    print(f"  ZNE improvement: {acc_zne - acc_unmitigated:+.4f}")

    result = {
        "unmitigated_accuracy": acc_unmitigated,
        "zne_accuracy": acc_zne,
        "zne_improvement": acc_zne - acc_unmitigated,
    }

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([result]).to_csv(
        os.path.join(output_dir, "zne_effectiveness.csv"), index=False
    )
    return result


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    with open("config/base_config.yaml") as f:
        config = yaml.safe_load(f)

    run_quantum_noise_sweep(config)
    run_noise_type_ablation(config)
    run_zne_effectiveness(config)
