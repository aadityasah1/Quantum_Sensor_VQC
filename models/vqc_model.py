from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

def create_vqc(num_qubits):

    feature_map = ZZFeatureMap(
        feature_dimension=num_qubits,
        reps=2
    )

    ansatz = RealAmplitudes(
        num_qubits,
        reps=3,
        entanglement='full'
    )

    optimizer = COBYLA(maxiter=100)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer
    )

    return vqc