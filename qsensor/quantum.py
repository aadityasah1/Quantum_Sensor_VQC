"""The variational quantum circuit, shared by training and serving.

Architecture (the one that reaches ~90% in the study):
    * data re-uploading: L blocks of [ZZ feature map + EfficientSU2]
    * average single-qubit <Z> expectation readout (smooth, trainable)
    * exact statevector estimator (default_precision=0) -> deterministic

Inference is a single statevector evaluation per signal, so it is fast enough
to serve online.
"""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map, efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

N_QUBITS = 4
L_REUP = 3
SEED = 42


def build_circuit(n_qubits: int = N_QUBITS, L: int = L_REUP):
    """Return (circuit, feature_map, weight_params, observable)."""
    fmap = zz_feature_map(feature_dimension=n_qubits, reps=1)
    qc = QuantumCircuit(n_qubits)
    weights = []
    for layer in range(L):
        qc.compose(fmap, inplace=True)                      # re-upload the data
        block = efficient_su2(n_qubits, reps=1, entanglement="linear",
                              parameter_prefix=f"w{layer}")
        qc.compose(block, inplace=True)
        weights += list(block.parameters)
    obs = SparsePauliOp.from_list(
        [("I" * i + "Z" + "I" * (n_qubits - 1 - i), 1.0 / n_qubits) for i in range(n_qubits)])
    return qc, fmap, weights, obs


def build_qnn(n_qubits: int = N_QUBITS, L: int = L_REUP):
    """Return (EstimatorQNN, n_weight_params)."""
    qc, fmap, weights, obs = build_circuit(n_qubits, L)
    estimator = StatevectorEstimator(default_precision=0.0, seed=SEED)   # exact
    qnn = EstimatorQNN(circuit=qc, observables=obs, input_params=list(fmap.parameters),
                       weight_params=weights, estimator=estimator)
    return qnn, len(weights)


def scores(qnn: EstimatorQNN, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average <Z> in [-1, 1] for each row of X (decision boundary at 0)."""
    return np.asarray(qnn.forward(X, weights)).reshape(-1)
