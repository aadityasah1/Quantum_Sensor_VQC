from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator

def create_noise_simulator(p=0.1):
    noise_model = NoiseModel()

    error = depolarizing_error(p, 1)
    noise_model.add_all_qubit_quantum_error(error, ['rx', 'ry'])

    simulator = AerSimulator(noise_model=noise_model)

    return simulator