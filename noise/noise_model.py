"""
Quantum noise model builder.

Qiskit 2.x compatibility note:
  - qiskit-machine-learning 0.9+ uses the V2 Sampler PUB API
  - AerSimulator requires circuits to be transpiled to basis gates
    before execution (ZZFeatureMap etc. are high-level library gates)
  - Solution: pair SamplerV2.from_backend() with a PassManager so that
    VQC transpiles circuits to AerSimulator basis gates internally
  - StatevectorSampler handles library gate decomposition natively
    (no pass manager needed for ideal simulation)

Sampler functions return (sampler, pass_manager) tuples so that callers
can pass both into create_vqc().
"""

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def build_noise_model(
    depolarizing_1q: float = 0.001,
    depolarizing_2q: float = 0.01,
    t1_us: float = 100.0,
    t2_us: float = 80.0,
    readout_error: float = 0.02,
    gate_time_1q_ns: float = 50.0,
) -> NoiseModel:
    """
    Returns a composite NoiseModel.

    Error channels:
      1. Depolarizing (1Q gates) + thermal relaxation — pre-composed per gate
      2. Depolarizing (2Q gates — cx, cz, ecr, swap)
      3. Symmetric readout error on all qubits
    """
    t2_us = min(t2_us, 2.0 * t1_us)
    nm = NoiseModel()

    has_depol = depolarizing_1q > 0
    has_thermal = t1_us < 1e9 and t2_us < 1e9

    rotation_gates = ["rx", "ry", "rz", "u", "u1", "u2", "u3"]
    other_1q_gates = ["x", "y", "z", "h"]

    if has_depol and has_thermal:
        t1_ns = t1_us * 1_000.0
        t2_ns = min(t2_us, 2.0 * t1_us) * 1_000.0
        err_1q = depolarizing_error(depolarizing_1q, 1)
        err_relax = thermal_relaxation_error(t1_ns, t2_ns, gate_time_1q_ns)
        nm.add_all_qubit_quantum_error(err_1q.compose(err_relax), rotation_gates)
        nm.add_all_qubit_quantum_error(err_1q, other_1q_gates)
    elif has_depol:
        err_1q = depolarizing_error(depolarizing_1q, 1)
        nm.add_all_qubit_quantum_error(err_1q, rotation_gates + other_1q_gates)
    elif has_thermal:
        t1_ns = t1_us * 1_000.0
        t2_ns = min(t2_us, 2.0 * t1_us) * 1_000.0
        nm.add_all_qubit_quantum_error(
            thermal_relaxation_error(t1_ns, t2_ns, gate_time_1q_ns), rotation_gates
        )

    if depolarizing_2q > 0:
        nm.add_all_qubit_quantum_error(
            depolarizing_error(depolarizing_2q, 2), ["cx", "cz", "ecr", "swap"]
        )

    if readout_error > 0:
        p = readout_error
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))

    return nm


def build_noisy_sampler_and_pm(
    noise_model: NoiseModel,
    shots: int = 1024,
    seed: int = 42,
) -> tuple:
    """
    Returns (SamplerV2, PassManager) for noisy VQC simulation.

    The PassManager transpiles high-level library circuits (ZZFeatureMap,
    EfficientSU2) to AerSimulator basis gates before execution.
    Both objects must be passed to create_vqc().
    """
    backend = AerSimulator(
        noise_model=noise_model,
        shots=shots,
        seed_simulator=seed,
    )
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    sampler = AerSamplerV2.from_backend(backend)
    sampler.options.default_shots = shots
    return sampler, pm


def build_ideal_sampler_and_pm(shots: int = 1024, seed: int = 42) -> tuple:
    """
    Returns (StatevectorSampler, None) for ideal (noiseless) simulation.

    StatevectorSampler decomposes library gates internally — no PassManager needed.
    """
    return StatevectorSampler(), None


# ── Backward-compatible shims (kept so old scripts don't break) ───────────────
def build_noisy_sampler(noise_model, shots=1024, seed=42):
    """Shim: returns only the sampler (no pass manager). Use build_noisy_sampler_and_pm instead."""
    sampler, _ = build_noisy_sampler_and_pm(noise_model, shots, seed)
    return sampler


def build_ideal_sampler(shots=1024, seed=42):
    """Shim: returns only the sampler. Use build_ideal_sampler_and_pm instead."""
    return StatevectorSampler()


def noise_model_from_config(config: dict, depolarizing_2q_override: float = None) -> NoiseModel:
    """Builds NoiseModel from the 'noise' section of base_config.yaml."""
    n = config["noise"]
    dep_2q = depolarizing_2q_override if depolarizing_2q_override is not None else n["depolarizing_2q"]
    return build_noise_model(
        depolarizing_1q=n["depolarizing_1q"],
        depolarizing_2q=dep_2q,
        t1_us=n["t1_us"],
        t2_us=n["t2_us"],
        readout_error=n["readout_error"],
        gate_time_1q_ns=n.get("gate_time_1q_ns", 50.0),
    )
