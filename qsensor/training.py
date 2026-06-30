"""Training pipeline: fit preprocessing, train the models, save an artifact bundle.

Run:  python -m qsensor.training            (full, ~10-12 min for the VQC)
      python -m qsensor.training --fast      (fewer iterations, for a quick build)

Saves to artifacts/:
    preprocessor.joblib   sklearn pipeline (scale -> PCA(4) -> scale[0,1])
    vqc_weights.npy       the 48 trained VQC angles
    classical_svm.joblib  RBF-SVM baseline (fast online fallback)
    metadata.json         config, class names, version, held-out metrics
"""
from __future__ import annotations
import os
import json
import time
import argparse
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import minimize

from .signals import generate_dataset, N_FEATURES, SNR
from .quantum import build_qnn, scores, N_QUBITS, L_REUP, SEED

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def build_preprocessor() -> Pipeline:
    """raw signal -> [0, pi] scale -> PCA(4) -> [0, 1] scale (the VQC input range)."""
    return Pipeline([
        ("scale_pi", MinMaxScaler((0.0, np.pi))),
        ("pca", PCA(n_components=N_QUBITS, random_state=SEED)),
        ("scale01", MinMaxScaler((0.0, 1.0))),
    ])


def train(n_samples: int = 240, max_iter: int = 400, out_dir: str = ARTIFACTS_DIR,
          seed: int = SEED) -> dict:
    try:
        from threadpoolctl import threadpool_limits
        ctx = threadpool_limits(limits=1)        # single-threaded => stable training
    except Exception:
        import contextlib
        ctx = contextlib.nullcontext()

    with ctx:
        t0 = time.time()
        X, y = generate_dataset(n_samples, snr=SNR, seed=seed)
        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y)

        pre = build_preprocessor().fit(Xtr_raw)          # fit on TRAIN only (no leakage)
        Xtr, Xte = pre.transform(Xtr_raw), pre.transform(Xte_raw)

        # classical baseline (instant, fast online fallback)
        svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                  random_state=seed).fit(Xtr, ytr)
        svm_acc = float(accuracy_score(yte, svm.predict(Xte)))

        # variational quantum classifier
        qnn, n_params = build_qnn()
        ytr_pm = 2 * ytr - 1
        history: list[float] = []
        print(f"[train] training VQC: COBYLA, up to {max_iter} circuit evaluations "
              f"(~{max_iter * 0.8 / 60:.0f} min on CPU). Progress:", flush=True)

        def objective(theta):
            loss = float(np.mean((scores(qnn, Xtr, theta) - ytr_pm) ** 2))
            history.append(loss)
            if len(history) % 25 == 0:
                print(f"  eval {len(history):4d}/~{max_iter}   loss={loss:.4f}", flush=True)
            return loss

        init = np.random.default_rng(seed).uniform(-0.1, 0.1, n_params)
        res = minimize(objective, init, method="COBYLA", options={"maxiter": max_iter})
        weights = res.x
        vqc_pred = (scores(qnn, Xte, weights) > 0).astype(int)
        vqc_acc = float(accuracy_score(yte, vqc_pred))
        vqc_f1 = float(f1_score(yte, vqc_pred))

        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(pre, os.path.join(out_dir, "preprocessor.joblib"))
        joblib.dump(svm, os.path.join(out_dir, "classical_svm.joblib"))
        np.save(os.path.join(out_dir, "vqc_weights.npy"), weights)
        meta = {
            "version": "1.0.0",
            "n_qubits": N_QUBITS, "L_reup": L_REUP, "n_features": N_FEATURES,
            "n_params": n_params, "seed": seed,
            "classes": {"0": "normal", "1": "fault"},
            "metrics": {"vqc_acc": vqc_acc, "vqc_f1": vqc_f1, "svm_acc": svm_acc},
            "trained_seconds": round(time.time() - t0, 1),
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[train] VQC acc={vqc_acc:.3f} f1={vqc_f1:.3f} | SVM acc={svm_acc:.3f} "
              f"| params={n_params} | {meta['trained_seconds']}s")
        print(f"[train] artifacts saved to {out_dir}")
        return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer COBYLA iterations")
    ap.add_argument("--samples", type=int, default=240)
    args = ap.parse_args()
    train(n_samples=args.samples, max_iter=120 if args.fast else 400)
