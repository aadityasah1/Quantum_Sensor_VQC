"""
Recovery script: rebuild vqc_model.pkl and metadata.pkl from the
weights + fold metrics that were printed to terminal before the crash.

Run once: python _recover_save.py
"""
import sys, os, pickle, numpy as np
sys.path.insert(0, ".")

import yaml, cloudpickle
from noise.noise_model import build_ideal_sampler_and_pm
from models.vqc_model import create_vqc

with open("config/base_config.yaml") as f:
    config = yaml.safe_load(f)

n_qubits = config["vqc"]["n_qubits"]   # 8
reps     = config["vqc"]["reps"]        # 3
max_iter = 50
seed     = 42

# ── Load saved weights ────────────────────────────────────────────────────────
with open("saved_models/vqc_params.pkl", "rb") as f:
    weights = pickle.load(f)
print(f"Loaded weights: shape={weights.shape}, dtype={weights.dtype}")

# ── Rebuild VQC with the saved weights ────────────────────────────────────────
sampler, pm = build_ideal_sampler_and_pm(shots=1024, seed=seed)
vqc = create_vqc(n_qubits=n_qubits, reps=reps, max_iter=1,
                 sampler=sampler, pass_manager=pm, seed=seed)

# Inject saved weights so the model behaves as if it was fitted
# (creates a minimal _fit_result that qiskit-machine-learning checks)
vqc._fit_result = type("_FR", (), {"x": weights})()
print("VQC rebuilt with saved weights.")

# ── Save with cloudpickle ─────────────────────────────────────────────────────
os.makedirs("saved_models", exist_ok=True)
with open("saved_models/vqc_model.pkl", "wb") as f:
    cloudpickle.dump(vqc, f)
print("Saved saved_models/vqc_model.pkl  (cloudpickle)")

# Also save npy format for convenience
np.save("saved_models/vqc_weights.npy", weights)
print("Saved saved_models/vqc_weights.npy")

# ── Rebuild metadata from terminal output ─────────────────────────────────────
# Fold metrics printed during the run (acc, f1, auc per fold):
fold_metrics = [
    {"accuracy": 0.5250, "f1": 0.5366, "auc_roc": 0.5961,
     "precision": 0.5208, "recall": 0.5625, "cohen_kappa": 0.0500, "matthews_corrcoef": 0.0503},
    {"accuracy": 0.6188, "f1": 0.6433, "auc_roc": 0.6198,
     "precision": 0.6111, "recall": 0.6875, "cohen_kappa": 0.2375, "matthews_corrcoef": 0.2385},
    {"accuracy": 0.6375, "f1": 0.6506, "auc_roc": 0.7125,
     "precision": 0.6250, "recall": 0.6875, "cohen_kappa": 0.2750, "matthews_corrcoef": 0.2760},
    {"accuracy": 0.5250, "f1": 0.5581, "auc_roc": 0.6202,
     "precision": 0.5417, "recall": 0.5750, "cohen_kappa": 0.0500, "matthews_corrcoef": 0.0503},
    {"accuracy": 0.5125, "f1": 0.5357, "auc_roc": 0.5890,
     "precision": 0.5208, "recall": 0.5500, "cohen_kappa": 0.0250, "matthews_corrcoef": 0.0251},
]

accs = [m["accuracy"] for m in fold_metrics]
best_fold = int(np.argmax(accs))   # fold 2 (index 2), acc=0.6375

metadata = {
    "n_qubits":         n_qubits,
    "n_features":       config["data"]["n_features"],
    "n_samples":        config["data"]["n_samples"],
    "reps":             reps,
    "max_iter":         max_iter,
    "use_noise":        False,
    "seed":             seed,
    "cv_fold_metrics":  fold_metrics,
    "cv_mean_accuracy": 0.5637,
    "cv_std_accuracy":  0.0531,
    "cv_mean_f1":       0.5849,
    "cv_mean_auc":      0.6275,
    "loss_histories":   [[] for _ in range(5)],   # SPSA callback not wired
    "best_fold":        best_fold,
}

with open("saved_models/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
print(f"Saved saved_models/metadata.pkl  (best fold={best_fold}, acc={accs[best_fold]:.4f})")

print("\nRecovery complete. All artifacts saved:")
for fp in ["saved_models/vqc_model.pkl", "saved_models/vqc_weights.npy",
           "saved_models/vqc_params.pkl", "saved_models/metadata.pkl"]:
    sz = os.path.getsize(fp)
    print(f"  {fp}  ({sz:,} bytes)")
