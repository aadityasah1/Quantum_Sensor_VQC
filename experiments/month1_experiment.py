"""
Month-1 progress experiment for the Noise-Resilient Quantum Sensor VQC project.

Pipeline
--------
1. Generate physics-inspired sensor signals (Normal vs Fault bearing vibration).
2. PCA -> 4 components, MinMax scale to [0, 1] for angle/ZZ encoding.
3. Classical baselines: SVM-RBF, Logistic Regression, MLP.
4. Quantum-kernel classifier: QSVC with a ZZ feature-map fidelity kernel.
5. Variational Quantum Classifier (VQC) trained on the ideal statevector
   simulator, with convergence tracking.
6. Noise-resilience study: evaluate the *trained* VQC weights under an
   increasing 2-qubit depolarizing error rate at inference time, and apply
   Zero-Noise Extrapolation (ZNE) at the device-level noise point.

All figures are written to results/ at 300 DPI (PNG + PDF) and all numeric
results to results/*.csv.

Usage
-----
    python -m experiments.month1_experiment            # full run
    python -m experiments.month1_experiment --quick     # fast smoke run
"""

import os
import sys
# Reproducibility: relaunch the process ONCE with every thread pool pinned to 1
# BEFORE numpy and qiskit load. Single-threaded math is deterministic; without
# this the VQC training drifts run-to-run (multi-threaded float reduction order
# plus qiskit's rayon parallelism) and will not reproduce the paper. Env vars
# only take effect if set before the libraries initialise, hence the relaunch.
if os.environ.get("_VQC_DETERMINISTIC") != "1":
    import subprocess
    _env = dict(os.environ, _VQC_DETERMINISTIC="1",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1", RAYON_NUM_THREADS="1", QISKIT_NUM_PROCESSES="1")
    sys.exit(subprocess.call([sys.executable, *sys.argv], env=_env))
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from qiskit.circuit.library import zz_feature_map, efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel, FidelityStatevectorKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from scipy.optimize import minimize as scipy_minimize

try:
    from qiskit_algorithms.optimizers import COBYLA, SPSA
except ImportError:
    from qiskit.algorithms.optimizers import COBYLA, SPSA

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.generate_data import generate_sensor_data

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
N_QUBITS = 4
SEED = 42

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 150,
})


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def build_dataset(n_samples, seed=SEED):
    X, y = generate_sensor_data(n_samples=n_samples, n_features=8, snr=3.0, seed=seed)
    # generate_sensor_data already MinMax-scales to [0, pi]; undo by re-deriving
    # from raw is not exposed, so we PCA then rescale to [0, 1] which is the
    # correct range for ZZ angle encoding (avoids 2*pi angle wrap-around).
    pca = PCA(n_components=N_QUBITS, random_state=seed)
    Xp = pca.fit_transform(X)
    Xp = MinMaxScaler((0.0, 1.0)).fit_transform(Xp)
    Xtr, Xte, ytr, yte = train_test_split(
        Xp, y, test_size=0.30, random_state=seed, stratify=y)
    return Xtr, Xte, ytr, yte, pca.explained_variance_ratio_


# -----------------------------------------------------------------------------
# Quantum building blocks
# -----------------------------------------------------------------------------
def make_circuit(fm_reps, ans_reps):
    fm = zz_feature_map(feature_dimension=N_QUBITS, reps=fm_reps)
    ans = efficient_su2(num_qubits=N_QUBITS, reps=ans_reps, entanglement="linear")
    qc = fm.compose(ans)
    return qc, fm, ans


def _parity(x):
    return bin(x).count("1") % 2


def make_qnn(sampler, pass_manager, fm_reps, ans_reps):
    qc, fm, ans = make_circuit(fm_reps, ans_reps)
    qnn = SamplerQNN(
        circuit=qc,
        input_params=list(fm.parameters),
        weight_params=list(ans.parameters),
        interpret=_parity,
        output_shape=2,
        sampler=sampler,
        pass_manager=pass_manager,
    )
    return qnn, ans.num_parameters


def build_noise_model(depol_2q, depol_1q=0.001, readout=0.02):
    nm = NoiseModel()
    if depol_1q > 0:
        nm.add_all_qubit_quantum_error(
            depolarizing_error(depol_1q, 1), ["rx", "ry", "rz", "u", "u3", "x", "h"])
    if depol_2q > 0:
        nm.add_all_qubit_quantum_error(
            depolarizing_error(depol_2q, 2), ["cx", "cz", "ecr"])
    if readout > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return nm


def noisy_sampler_and_pm(depol_2q, shots, seed=SEED):
    nm = build_noise_model(depol_2q)
    backend = AerSimulator(noise_model=nm, seed_simulator=seed)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    sampler = AerSamplerV2.from_backend(backend)
    sampler.options.default_shots = shots
    return sampler, pm


# -----------------------------------------------------------------------------
# Zero-Noise Extrapolation (global gate folding + linear Richardson)
# -----------------------------------------------------------------------------
def fold_global(circuit, scale_factor):
    """Unitary folding: C -> C (C^dag C)^k  to amplify noise by ~odd integer."""
    from qiskit import QuantumCircuit
    if scale_factor <= 1:
        return circuit
    k = (int(scale_factor) - 1) // 2
    folded = circuit.copy()
    inv = circuit.inverse()
    for _ in range(k):
        folded.barrier()          # stop the transpiler cancelling C^dag C
        folded = folded.compose(inv)
        folded.barrier()
        folded = folded.compose(circuit)
    return folded


# -----------------------------------------------------------------------------
# Advanced VQC: data re-uploading + expectation-value readout (EstimatorQNN)
# -----------------------------------------------------------------------------
def make_vqc_circuit(L):
    """L blocks of [ZZ feature map (re-uploaded) + EfficientSU2 trainable block].

    Returns (circuit, feature_map, weight_params, observable). The observable is
    the average of single-qubit Z, giving a smooth <O> in [-1, 1] that trains far
    better than the parity bit used by the earlier SamplerQNN model.
    """
    fmap = zz_feature_map(feature_dimension=N_QUBITS, reps=1)
    qc = QuantumCircuit(N_QUBITS)
    weights = []
    for l in range(L):
        qc.compose(fmap, inplace=True)                       # re-upload the data
        blk = efficient_su2(N_QUBITS, reps=1, entanglement="linear",
                            parameter_prefix=f"w{l}")
        qc.compose(blk, inplace=True)
        weights += list(blk.parameters)
    obs = SparsePauliOp.from_list(
        [("I" * i + "Z" + "I" * (N_QUBITS - 1 - i), 1.0 / N_QUBITS) for i in range(N_QUBITS)])
    return qc, fmap, weights, obs


def ideal_estimator():
    # default_precision=0.0 => exact expectation (no shot noise) => deterministic.
    return StatevectorEstimator(default_precision=0.0, seed=SEED)


def noisy_estimator(depol_2q, shots=None, seed=SEED):
    # Exact noisy expectation via density-matrix simulation (no shot noise) so the
    # noise sweep and ZNE are smooth and bit-reproducible. 4 qubits => cheap.
    nm = build_noise_model(depol_2q)
    return AerEstimatorV2(options={
        "backend_options": {"noise_model": nm, "method": "density_matrix",
                            "seed_simulator": seed,
                            "max_parallel_threads": 1, "max_parallel_experiments": 1},
        "default_precision": 0.0})


def make_estimator_qnn(estimator, L):
    qc, fmap, weights, obs = make_vqc_circuit(L)
    qnn = EstimatorQNN(circuit=qc, observables=obs, input_params=list(fmap.parameters),
                       weight_params=weights, estimator=estimator)
    return qnn, len(weights)


def vqc_predict(qnn, X, weights):
    """<O> > 0 -> class 1, else class 0  (labels were trained to +-1)."""
    out = np.asarray(qnn.forward(X, weights)).reshape(-1)
    return (out > 0).astype(int)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main(quick=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(SEED)
    t_start = time.time()

    if quick:
        n_samples, fm_reps, L_reup = 80, 1, 2
        max_iter, shots = 80, 512
        noise_levels = [0.0, 0.02, 0.05, 0.1]
    else:
        n_samples, fm_reps, L_reup = 240, 2, 3
        max_iter, shots = 400, 1024
        noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15]

    Xtr, Xte, ytr, yte, evr = build_dataset(n_samples)
    print(f"[data] train={Xtr.shape} test={Xte.shape} "
          f"balance(train)={np.bincount(ytr).tolist()}  "
          f"PCA var explained={evr.sum():.3f}")

    summary_rows = []

    # ---- Classical baselines ------------------------------------------------
    print("\n[classical] training baselines ...")
    classical = {
        "SVM-RBF": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=SEED),
        "Logistic-Reg": LogisticRegression(max_iter=1000, random_state=SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             early_stopping=True, random_state=SEED),
    }
    for name, clf in classical.items():
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred)
        summary_rows.append({"Model": name, "Type": "Classical",
                             "Accuracy": acc, "F1": f1})
        print(f"  {name:<14} acc={acc:.3f} f1={f1:.3f}")
    svm_acc = summary_rows[0]["Accuracy"]

    # ---- Quantum kernel (QSVC) ---------------------------------------------
    print("\n[quantum] QSVC fidelity-kernel ...")
    # exact statevector kernel (no shot sampling) => deterministic / reproducible
    qkernel = FidelityStatevectorKernel(
        feature_map=zz_feature_map(feature_dimension=N_QUBITS, reps=fm_reps))
    qsvc = QSVC(quantum_kernel=qkernel)
    t0 = time.time()
    qsvc.fit(Xtr, ytr)
    qsvc_pred = qsvc.predict(Xte)
    qsvc_acc = accuracy_score(yte, qsvc_pred)
    qsvc_f1 = f1_score(yte, qsvc_pred)
    summary_rows.append({"Model": "QSVC (quantum kernel)", "Type": "Quantum",
                         "Accuracy": qsvc_acc, "F1": qsvc_f1})
    print(f"  QSVC           acc={qsvc_acc:.3f} f1={qsvc_f1:.3f}  ({time.time()-t0:.0f}s)")

    # ---- VQC (ideal training): data re-uploading + <Z> readout + COBYLA ------
    print("\n[quantum] VQC training (data re-uploading, <Z> readout, COBYLA) ...")
    qnn_ideal, n_params = make_estimator_qnn(ideal_estimator(), L_reup)
    ytr_pm = 2 * ytr - 1                                 # {0,1} -> {-1,+1}
    history = []
    def obj(th):
        out = np.asarray(qnn_ideal.forward(Xtr, th)).reshape(-1)
        loss = float(np.mean((out - ytr_pm) ** 2))
        history.append(loss)
        return loss
    init = np.random.default_rng(SEED).uniform(-0.1, 0.1, n_params)
    t0 = time.time()
    res = scipy_minimize(obj, init, method="COBYLA", options={"maxiter": max_iter})
    trained_w = res.x
    vqc_pred = vqc_predict(qnn_ideal, Xte, trained_w)
    vqc_acc = accuracy_score(yte, vqc_pred)
    vqc_f1 = f1_score(yte, vqc_pred)
    summary_rows.append({"Model": "VQC (ideal)", "Type": "Quantum",
                         "Accuracy": vqc_acc, "F1": vqc_f1})
    print(f"  VQC ideal      acc={vqc_acc:.3f} f1={vqc_f1:.3f}  "
          f"params={n_params}  ({time.time()-t0:.0f}s)")

    # ---- Noise-resilience sweep (inference-time) ---------------------------
    print("\n[quantum] noise-resilience sweep (fixed trained weights) ...")
    sweep_rows = []
    for nl in noise_levels:
        est = ideal_estimator() if nl == 0.0 else noisy_estimator(nl, shots)
        qnn_n, _ = make_estimator_qnn(est, L_reup)
        preds = vqc_predict(qnn_n, Xte, trained_w)
        acc = accuracy_score(yte, preds)
        sweep_rows.append({"noise_level": nl, "vqc_acc": acc, "svm_acc": svm_acc})
        print(f"  depol_2q={nl:<6} VQC acc={acc:.3f}")

    # ---- ZNE at device noise level -----------------------------------------
    device_nl = 0.02
    print(f"\n[quantum] ZNE at depol_2q={device_nl} ...")
    qc_base, fm_b, w_b, obs_b = make_vqc_circuit(L_reup)
    scale_factors = [1, 3, 5]
    zne_accs = []
    for sf in scale_factors:
        folded = fold_global(qc_base, sf)
        qnn_z = EstimatorQNN(
            circuit=folded, observables=obs_b, input_params=list(fm_b.parameters),
            weight_params=w_b, estimator=noisy_estimator(device_nl, shots))
        preds = vqc_predict(qnn_z, Xte, trained_w)
        acc = accuracy_score(yte, preds)
        zne_accs.append(acc)
        print(f"  scale={sf} acc={acc:.3f}")
    # linear (Richardson) extrapolation to zero noise
    coeffs = np.polyfit(scale_factors, zne_accs, 1)
    zne_extrap = float(np.clip(np.polyval(coeffs, 0.0), 0.0, 1.0))   # accuracy <= 1
    noisy_raw = zne_accs[0]
    print(f"  noisy(raw)={noisy_raw:.3f}  ZNE-extrapolated={zne_extrap:.3f}")

    # =========================================================================
    # SAVE results
    # =========================================================================
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "month1_model_comparison.csv"), index=False)
    df_sweep = pd.DataFrame(sweep_rows)
    df_sweep.to_csv(os.path.join(RESULTS_DIR, "month1_noise_sweep.csv"), index=False)
    pd.DataFrame({"scale_factor": scale_factors + ["extrapolated"],
                  "accuracy": zne_accs + [zne_extrap]}).to_csv(
        os.path.join(RESULTS_DIR, "month1_zne.csv"), index=False)

    # ---- Figure 1: convergence ---------------------------------------------
    if history:
        pd.DataFrame({"loss": history}).to_csv(
            os.path.join(RESULTS_DIR, "month1_convergence.csv"), index=False)
        best = np.minimum.accumulate(history)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(history, color="#9bbce0", linewidth=1.2, label="per-evaluation loss")
        ax.plot(best, color="#1F4E79", linewidth=2.0, label="best so far")
        ax.set_xlabel("Optimizer evaluation (COBYLA)")
        ax.set_ylabel("Training loss (MSE)")
        ax.set_title("VQC Training Convergence (ideal simulator)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(RESULTS_DIR, f"month1_convergence.{ext}"),
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ---- Figure 2: model comparison bar ------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C9F70" if t == "Classical" else "#3B6EA5"
              for t in df_summary["Type"]]
    bars = ax.bar(df_summary["Model"], df_summary["Accuracy"], color=colors, alpha=0.9)
    for b, a in zip(bars, df_summary["Accuracy"]):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.axhline(0.5, color="gray", linestyle=":", label="Random baseline")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("Sensor Fault Classification: Classical vs Quantum Models")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS_DIR, f"month1_model_comparison.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: noise sweep ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(df_sweep["noise_level"], df_sweep["vqc_acc"], "o-",
            color="#3B6EA5", linewidth=2, label="VQC (trained, noisy inference)")
    ax.axhline(svm_acc, color="#4C9F70", linestyle="-", linewidth=2,
               label=f"Classical SVM ({svm_acc:.2f})")
    ax.scatter([device_nl], [noisy_raw], color="#C44E52", zorder=5,
               label=f"VQC @ device noise ({noisy_raw:.2f})")
    ax.scatter([0.0], [zne_extrap], color="#DD8452", marker="^", s=90, zorder=5,
               label=f"VQC + ZNE ({zne_extrap:.2f})")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="Random baseline")
    ax.set_xlabel("2-qubit depolarizing error rate")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.3, 1.05)
    ax.set_title("Noise Resilience of the Trained VQC")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS_DIR, f"month1_noise_sweep.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 4: confusion matrices (SVM vs VQC ideal) -------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (title, pred) in zip(axes, [("Classical SVM-RBF", classical["SVM-RBF"].predict(Xte)),
                                        ("VQC (ideal)", vqc_pred)]):
        cm = confusion_matrix(yte, pred)
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Fault"]); ax.set_yticklabels(["Normal", "Fault"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.suptitle("Confusion Matrices (test set)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS_DIR, f"month1_confusion.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] total {time.time()-t_start:.0f}s. Results + figures in {RESULTS_DIR}/")
    print("\n=== SUMMARY ===")
    print(df_summary.to_string(index=False))
    print("\n=== NOISE SWEEP ===")
    print(df_sweep.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    # Force single-threaded BLAS at RUNTIME. Setting env vars in-process is too
    # late once numpy has loaded, so we use threadpoolctl, which limits the
    # already-loaded BLAS/OpenMP pools. Without this the VQC/QSVC results drift
    # run-to-run (multi-threaded float reduction order) and do not reproduce.
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            main(quick=args.quick)
    except ImportError:
        main(quick=args.quick)
