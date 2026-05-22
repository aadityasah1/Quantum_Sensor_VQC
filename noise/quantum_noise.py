from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError
)


def create_quantum_noise_model(noise_level=0.05):

    noise_model = NoiseModel()

    # -----------------------------
    # Gate Noise
    # -----------------------------

    single_qubit_error = depolarizing_error(
        noise_level,
        1
    )

    two_qubit_error = depolarizing_error(
        noise_level * 2,
        2
    )

    noise_model.add_all_qubit_quantum_error(
        single_qubit_error,
        ['rx', 'ry', 'rz']
    )

    noise_model.add_all_qubit_quantum_error(
        two_qubit_error,
        ['cx']
    )

    # -----------------------------
    # Readout Noise
    # -----------------------------

    readout_error = ReadoutError([
        [1 - noise_level, noise_level],
        [noise_level, 1 - noise_level]
    ])

    noise_model.add_all_qubit_readout_error(
        readout_error
    )

    return noise_model