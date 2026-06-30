"""Prototype: advanced VQC (expectation readout + data re-uploading) that should
beat the 77.8% parity+COBYLA model, while staying deterministic/reproducible.

Forward-only objective (no slow parameter-shift gradients). Compares SPSA and
COBYLA. Single-threaded for reproducibility.
"""
import os, sys, time
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.optimize import minimize as scipy_min
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.generate_data import generate_sensor_data

from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map, efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
try:
    from qiskit_algorithms.optimizers import SPSA
    from qiskit_algorithms.utils import algorithm_globals
except ImportError:
    from qiskit.algorithms.optimizers import SPSA
    from qiskit.utils import algorithm_globals

SEED = 42
N_QUBITS = 4


def build_data(n=240):
    X, y = generate_sensor_data(n_samples=n, n_features=8, snr=3.0, seed=SEED)
    X = PCA(n_components=N_QUBITS, random_state=SEED).fit_transform(X)
    X = MinMaxScaler((0, 1)).fit_transform(X)
    return train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)


def build_qnn(L=3, fm_reps=1):
    fmap = zz_feature_map(feature_dimension=N_QUBITS, reps=fm_reps)
    qc = QuantumCircuit(N_QUBITS)
    weights = []
    for l in range(L):
        qc.compose(fmap, inplace=True)
        blk = efficient_su2(N_QUBITS, reps=1, entanglement="linear", parameter_prefix=f"w{l}")
        qc.compose(blk, inplace=True)
        weights += list(blk.parameters)
    obs = SparsePauliOp.from_list(
        [("I" * i + "Z" + "I" * (N_QUBITS - 1 - i), 1.0 / N_QUBITS) for i in range(N_QUBITS)])
    est = StatevectorEstimator(default_precision=0.0, seed=SEED)  # exact, deterministic
    qnn = EstimatorQNN(circuit=qc, observables=obs,
                       input_params=list(fmap.parameters), weight_params=weights,
                       estimator=est)
    return qnn, len(weights)


def acc(th, qnn, X, y_pm):
    return accuracy_score(y_pm, np.sign(qnn.forward(X, th).reshape(-1)))


def my_spsa(obj, x0, maxiter=150, a=0.25, c=0.10, seed=SEED):
    """Deterministic SPSA: perturbations from a seeded numpy Generator."""
    rng = np.random.default_rng(seed)
    x = np.array(x0, float)
    A = 0.10 * maxiter
    alpha, gamma = 0.602, 0.101
    best_x, best_f = x.copy(), obj(x)
    for k in range(maxiter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)
        fp, fm = obj(x + ck * delta), obj(x - ck * delta)
        x = x - ak * (fp - fm) / (2.0 * ck) * delta
        f = 0.5 * (fp + fm)
        if f < best_f:
            best_f, best_x = f, x.copy()
    return best_x


def run(L=3, maxiter=150):
    Xtr, Xte, ytr, yte = build_data()
    ytr_pm, yte_pm = 2 * ytr - 1, 2 * yte - 1
    qnn, nP = build_qnn(L=L)
    th0 = np.random.default_rng(SEED).uniform(-0.1, 0.1, nP)

    def obj(th):
        out = qnn.forward(Xtr, th).reshape(-1)
        return float(np.mean((out - ytr_pm) ** 2))

    t0 = time.time()
    thx = my_spsa(obj, th0, maxiter=maxiter)
    tr = acc(thx, qnn, Xtr, ytr_pm)
    te = acc(thx, qnn, Xte, yte_pm)
    print(f"[mySPSA] L={L} params={nP} maxiter={maxiter} train={tr:.3f} test={te:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return te


def run_cobyla(L=3, maxiter=400):
    Xtr, Xte, ytr, yte = build_data()
    ytr_pm, yte_pm = 2 * ytr - 1, 2 * yte - 1
    qnn, nP = build_qnn(L=L)
    th0 = np.random.default_rng(SEED).uniform(-0.1, 0.1, nP)

    def obj(th):
        out = qnn.forward(Xtr, th).reshape(-1)
        return float(np.mean((out - ytr_pm) ** 2))

    t0 = time.time()
    res = scipy_min(obj, th0, method="COBYLA", options={"maxiter": maxiter})
    tr = acc(res.x, qnn, Xtr, ytr_pm)
    te = acc(res.x, qnn, Xte, yte_pm)
    print(f"[COBYLA-exact] L={L} params={nP} maxiter={maxiter} train={tr:.3f} test={te:.3f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run_cobyla(L=3, maxiter=400)
        run(L=3, maxiter=200)        # custom seeded SPSA for comparison
