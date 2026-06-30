"""Load the saved model bundle and classify a new sensor signal."""
from __future__ import annotations
import os
import json
import numpy as np
import joblib

from .quantum import build_qnn, scores
from .training import ARTIFACTS_DIR


class Predictor:
    """Loads the artifact bundle once and serves predictions.

    Two models are available per request:
      * 'vqc'       - the variational quantum classifier (the project's model)
      * 'classical' - the RBF-SVM baseline (fast fallback)
    """

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        if not os.path.exists(os.path.join(artifacts_dir, "metadata.json")):
            raise FileNotFoundError(
                f"No model artifacts in {artifacts_dir}. Train first: "
                "python -m qsensor.training")
        with open(os.path.join(artifacts_dir, "metadata.json")) as f:
            self.meta = json.load(f)
        self.pre = joblib.load(os.path.join(artifacts_dir, "preprocessor.joblib"))
        self.svm = joblib.load(os.path.join(artifacts_dir, "classical_svm.joblib"))
        self.weights = np.load(os.path.join(artifacts_dir, "vqc_weights.npy"))
        self.qnn, _ = build_qnn(self.meta["n_qubits"], self.meta["L_reup"])
        self.classes = self.meta["classes"]
        self.n_features = int(self.meta["n_features"])

    def predict(self, signal, model: str = "vqc") -> dict:
        x = np.asarray(signal, dtype=float).reshape(1, -1)
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} signal samples, got {x.shape[1]}")
        xt = self.pre.transform(x)

        if model == "classical":
            label = int(self.svm.predict(xt)[0])
            confidence = float(self.svm.predict_proba(xt)[0][label])
            z = None
        elif model == "vqc":
            z = float(scores(self.qnn, xt, self.weights)[0])     # <Z> in [-1, 1]
            label = int(z > 0)
            confidence = float(0.5 + 0.5 * abs(z))               # distance from boundary
        else:
            raise ValueError("model must be 'vqc' or 'classical'")

        return {
            "label": label,
            "prediction": self.classes[str(label)],
            "confidence": round(confidence, 4),
            "model": model,
            "z_expectation": round(z, 4) if z is not None else None,
        }
