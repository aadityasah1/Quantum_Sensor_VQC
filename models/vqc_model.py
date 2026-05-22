"""
VQC model factory — Qiskit 2.x / qiskit-machine-learning 0.9 compatible.

Key design decisions:
  1. EfficientSU2 with LINEAR entanglement avoids barren plateaus
     (old code used RealAmplitudes with FULL entanglement)
  2. SPSA optimizer: 2 circuit evals/step vs O(n_params) for COBYLA
  3. Near-identity init: params in (-0.1, 0.1) → non-vanishing gradients
  4. pass_manager: required when sampler uses AerSimulator backend.
     ZZFeatureMap / EfficientSU2 are high-level library gates; AerSimulator
     cannot execute them directly — the pass_manager decomposes them to
     basis gates before execution. StatevectorSampler handles this itself.
  5. Noise is injected via SamplerV2.from_backend(AerSimulator(noise_model=...))
"""

import warnings
import logging
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit_machine_learning.algorithms import VQC

logging.getLogger("qiskit_machine_learning").setLevel(logging.ERROR)
logging.getLogger("qiskit_machine_learning.neural_networks").setLevel(logging.ERROR)

try:
    from qiskit_algorithms.optimizers import SPSA, COBYLA
except ImportError:
    from qiskit.algorithms.optimizers import SPSA, COBYLA


def create_vqc(
    n_qubits: int = 8,
    reps: int = 3,
    max_iter: int = 300,
    sampler=None,
    pass_manager=None,
    seed: int = 42,
    callback=None,
) -> VQC:
    """
    Builds a VQC classifier compatible with qiskit 2.x.

    Parameters
    ----------
    n_qubits     : number of qubits (= number of input features)
    reps         : EfficientSU2 variational layers
    max_iter     : SPSA optimizer iterations
    sampler      : V2 Sampler. Use build_noisy_sampler_and_pm() or
                   build_ideal_sampler_and_pm() from noise.noise_model.
                   If None, qiskit-machine-learning uses its default sampler.
    pass_manager : Required when sampler wraps AerSimulator — decomposes
                   library gates to basis gates before circuit execution.
                   Pass None when using StatevectorSampler (ideal).
    seed         : for reproducible parameter initialization
    callback     : called each SPSA step as callback(nfev, params, value, meta)
    """
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)

    ansatz = EfficientSU2(
        num_qubits=n_qubits,
        reps=reps,
        entanglement="linear",   # linear avoids barren plateaus vs 'full'
        insert_barriers=False,
    )

    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(-0.1, 0.1, ansatz.num_parameters)

    optimizer = SPSA(maxiter=max_iter, callback=callback)

    vqc_kwargs = dict(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
    )
    if sampler is not None:
        vqc_kwargs["sampler"] = sampler
    if pass_manager is not None:
        vqc_kwargs["pass_manager"] = pass_manager

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No gradient function provided")
        warnings.filterwarnings("ignore", message=".*pass manager.*")
        return VQC(**vqc_kwargs)


def create_vqc_from_config(
    config: dict,
    sampler=None,
    pass_manager=None,
    seed: int = 42,
    callback=None,
) -> VQC:
    """Builds VQC from the 'vqc' section of base_config.yaml."""
    v = config["vqc"]
    return create_vqc(
        n_qubits=v["n_qubits"],
        reps=v["reps"],
        max_iter=v["max_iter"],
        sampler=sampler,
        pass_manager=pass_manager,
        seed=seed,
        callback=callback,
    )
