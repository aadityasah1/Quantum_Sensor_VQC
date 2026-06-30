"""Standalone convergence-curve run for the report.

Logs the training objective directly inside the loss function (every evaluation),
so the curve does not depend on any optimizer callback firing.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from qiskit.circuit.library import zz_feature_map, efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

# this file lives in scripts/ ; put the project root on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from data.generate_data import generate_sensor_data

RES = os.path.join(ROOT, "results")
SEED = 42
np.random.seed(SEED)

X, y = generate_sensor_data(n_samples=160, n_features=8, snr=3.0, seed=SEED)
X = PCA(n_components=4, random_state=SEED).fit_transform(X)
X = MinMaxScaler((0, 1)).fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
ytr = np.asarray(ytr)

fm = zz_feature_map(feature_dimension=4, reps=2)
ans = efficient_su2(num_qubits=4, reps=3, entanglement="linear")
qc = fm.compose(ans)
qnn = SamplerQNN(circuit=qc, input_params=list(fm.parameters),
                 weight_params=list(ans.parameters),
                 interpret=lambda x: bin(x).count("1") % 2,
                 output_shape=2, sampler=StatevectorSampler())

eps = 1e-9
history = []

def loss(w):
    probs = qnn.forward(Xtr, w)            # shape (n_samples, 2)
    p_true = probs[np.arange(len(ytr)), ytr]
    ce = -np.mean(np.log(np.clip(p_true, eps, 1.0)))
    history.append(float(ce))
    return ce

rng = np.random.default_rng(SEED)
init = rng.uniform(-0.1, 0.1, ans.num_parameters)
res = minimize(loss, init, method="COBYLA", options={"maxiter": 200, "rhobeg": 0.3})

# final test accuracy with optimized weights
probs_te = qnn.forward(Xte, res.x)
pred_te = probs_te.argmax(axis=1)
acc = float((pred_te == np.asarray(yte)).mean())
print("final test acc:", acc, "history len:", len(history))

# best-so-far envelope for a clean monotone view alongside the raw trace
best = np.minimum.accumulate(history)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(history, color="#9bbce0", linewidth=1.2, label="objective per evaluation")
ax.plot(best, color="#1F4E79", linewidth=2.0, label="best so far")
ax.set_xlabel("Optimizer evaluation (COBYLA)")
ax.set_ylabel("Training loss (cross-entropy)")
ax.set_title("VQC Training Convergence (ideal statevector simulator)")
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(RES, f"month1_convergence.{ext}"), dpi=300, bbox_inches="tight")
print("saved month1_convergence.png  (start=%.3f end=%.3f)" % (history[0], history[-1]))
