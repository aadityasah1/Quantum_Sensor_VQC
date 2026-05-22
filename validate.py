"""
Quick validation script — run this to confirm all modules work before training.
Usage: python validate.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import yaml

print("=" * 55)
print("  VQC Project — Module Validation")
print("=" * 55)

# ── Config ────────────────────────────────────────────────────────
with open("config/base_config.yaml") as f:
    config = yaml.safe_load(f)
print("\n[1] Config loaded OK")
print(f"    n_qubits={config['vqc']['n_qubits']}, "
      f"max_iter={config['vqc']['max_iter']}, "
      f"shots={config['vqc']['shots']}")

# ── Dataset ───────────────────────────────────────────────────────
from data.generate_data import generate_sensor_data, load_and_split
X, y = generate_sensor_data(n_samples=200, n_features=8, snr=3.0, seed=42)
print(f"\n[2] Data generation OK")
print(f"    X shape: {X.shape}  |  X range: [{X.min():.3f}, {X.max():.3f}]")
print(f"    Class balance: {np.bincount(y).tolist()}")

# Check class separation
mean0 = X[y == 0].mean(axis=0)
mean1 = X[y == 1].mean(axis=0)
separation = float(np.linalg.norm(mean0 - mean1))
print(f"    Class mean separation (L2): {separation:.4f}  (>0.5 = good signal)")

# ── Noise model ───────────────────────────────────────────────────
from noise.noise_model import (
    build_noise_model, build_noisy_sampler, build_ideal_sampler,
    noise_model_from_config,
)
nm = noise_model_from_config(config)
print(f"\n[3] Noise model OK — {type(nm).__name__}")
print(f"    Gates with errors: {list(nm.noise_instructions)}")

noisy_sampler = build_noisy_sampler(nm, shots=128, seed=42)
ideal_sampler = build_ideal_sampler(shots=128, seed=42)
print(f"    Noisy sampler: {type(noisy_sampler).__name__}")
print(f"    Ideal sampler: {type(ideal_sampler).__name__}")

# ── Classical models ──────────────────────────────────────────────
from models.classical_model import create_classical_model
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
svm = create_classical_model("svm_rbf")
svm.fit(X_tr, y_tr)
svm_acc = svm.score(X_te, y_te)
print(f"\n[4] Classical model OK")
print(f"    SVM (RBF) test accuracy: {svm_acc:.4f}  (expect >0.85)")

# ── Metrics ───────────────────────────────────────────────────────
from evaluation.metrics import compute_metrics, aggregate_cv_metrics
y_pred = svm.predict(X_te)
y_proba = svm.predict_proba(X_te)
m = compute_metrics(y_te, y_pred, y_proba)
print(f"\n[5] Metrics module OK")
print(f"    accuracy={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
      f"auc={m['auc_roc']:.4f}  kappa={m['cohen_kappa']:.4f}")

# ── Visualization ─────────────────────────────────────────────────
from evaluation.visualization import plot_model_comparison
import matplotlib
matplotlib.use("Agg")  # no display needed for validation
print(f"\n[6] Visualization module OK")

# ── Experiment modules ────────────────────────────────────────────
import importlib.util
for mod in ["experiments.run_experiments", "experiments.noise_analysis", "experiments.plots"]:
    spec = importlib.util.find_spec(mod)
    status = "FOUND" if spec else "NOT FOUND"
    print(f"\n[7] {mod}: {status}")

# ── VQC model (import only, no training) ──────────────────────────
from models.vqc_model import create_vqc
vqc = create_vqc(n_qubits=4, reps=1, max_iter=1, sampler=None, seed=42)
print(f"\n[8] VQC model import OK")
print(f"    Feature map: {type(vqc.feature_map).__name__}")
print(f"    Ansatz: {type(vqc.ansatz).__name__}")
print(f"    n_params: {vqc.ansatz.num_parameters}")

print("\n" + "=" * 55)
print("  All modules validated. Ready to train.")
print("=" * 55)
print()
print("NEXT STEPS:")
print("  python train.py --no-noise --max-iter 50   # quick smoke test")
print("  python train.py                             # full run (noisy, CV)")
print("  python -m experiments.run_experiments       # full comparison")
print("  python -m experiments.noise_analysis        # noise sweep")
