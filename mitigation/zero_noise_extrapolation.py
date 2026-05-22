"""
Zero-Noise Extrapolation (ZNE) for quantum error mitigation.

Algorithm (Temme et al. 2017, Li & Benjamin 2017):
  1. Run the noisy circuit at scale factors lambda = [1, 3, 5, ...]
     Gate folding: replace each gate G with G (G† G)^k so scale = 2k+1.
     This multiplies the noise approximately by lambda while keeping the
     ideal unitary unchanged (G G† G = G for unitary G).
  2. Record the noisy expectation value E(lambda) at each scale.
  3. Extrapolate the polynomial E(lambda) back to lambda=0 using
     Richardson extrapolation (numpy polynomial fit).

For classification we apply ZNE per sample to the predicted probability
vector and re-take argmax.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from copy import deepcopy


# ── Gate folding ──────────────────────────────────────────────────────────────

def fold_circuit_gates(circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
    """
    Apply gate folding with noise scale factor `scale` (must be odd >= 1).

    Each gate G becomes G (G† G)^((scale-1)//2):
      scale=1 → G            (original, no folding)
      scale=3 → G G† G       (noise ~3x)
      scale=5 → G G† G G† G  (noise ~5x)
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"scale must be a positive odd integer, got {scale}")
    if scale == 1:
        return deepcopy(circuit)

    folded = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    k = (scale - 1) // 2

    for instruction in circuit.data:
        op = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits

        # Append original gate
        folded.append(op, qubits, clbits)

        # Append (G† G) k times
        try:
            inv_op = op.inverse()
        except Exception:
            # Gate has no inverse (e.g. measure, reset) — skip folding
            continue
        for _ in range(k):
            folded.append(inv_op, qubits, clbits)
            folded.append(op, qubits, clbits)

    return folded


# ── Richardson extrapolation ──────────────────────────────────────────────────

def richardson_extrapolate(
    scale_factors: list,
    expectation_values: list,
) -> float:
    """
    Polynomial extrapolation to lambda=0 (zero-noise limit).

    Fits a degree-(n-1) polynomial through the n (scale, value) points
    and evaluates it at scale=0.
    """
    coeffs = np.polyfit(scale_factors, expectation_values, deg=len(scale_factors) - 1)
    return float(np.polyval(coeffs, 0.0))


# ── ZNE Mitigator class ───────────────────────────────────────────────────────

class ZNEMitigator:
    """
    Wraps a trained VQC and applies ZNE at inference time.

    Usage
    -----
    mitigator = ZNEMitigator(scale_factors=[1, 3, 5], noise_config=config["noise"])
    y_pred_mitigated = mitigator.mitigated_predict(vqc, X_test, base_noise_cfg=config)
    """

    def __init__(self, scale_factors: list = None, noise_config: dict = None):
        self.scale_factors = scale_factors or [1, 3, 5]
        self.noise_config = noise_config or {}

    @property
    def overhead_factor(self) -> int:
        """Number of extra circuit runs per sample (= number of scale factors)."""
        return len(self.scale_factors)

    def mitigated_predict(self, vqc, X: np.ndarray) -> np.ndarray:
        """
        Run prediction at each noise scale and Richardson-extrapolate to zero noise.

        Returns predicted class labels, shape (n_samples,).
        """
        from noise.noise_model import build_noise_model, build_noisy_sampler_and_pm
        from models.vqc_model import create_vqc

        all_probas = []  # list of (n_samples, n_classes) arrays, one per scale

        for scale in self.scale_factors:
            # Build a scaled noise model: multiply depolarizing rates by scale
            dep_1q = self.noise_config.get("depolarizing_1q", 0.001) * scale
            dep_2q = self.noise_config.get("depolarizing_2q", 0.01) * scale
            # Clamp to valid probability range
            dep_1q = min(dep_1q, 0.75)
            dep_2q = min(dep_2q, 0.75)

            scaled_nm = build_noise_model(
                depolarizing_1q=dep_1q,
                depolarizing_2q=dep_2q,
                t1_us=self.noise_config.get("t1_us", 100.0),
                t2_us=self.noise_config.get("t2_us", 80.0),
                readout_error=self.noise_config.get("readout_error", 0.02),
            )
            scaled_sampler, scaled_pm = build_noisy_sampler_and_pm(scaled_nm, shots=1024)

            # Transfer trained weights to a fresh VQC with the scaled sampler
            trained_vqc = _clone_vqc_with_sampler(vqc, scaled_sampler, scaled_pm)
            probas = trained_vqc.predict_proba(X)  # (n_samples, n_classes)
            all_probas.append(probas)

        # Extrapolate each (sample, class) probability to zero noise
        all_probas = np.array(all_probas)  # (n_scales, n_samples, n_classes)
        n_samples, n_classes = all_probas.shape[1], all_probas.shape[2]
        mitigated_probas = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for c in range(n_classes):
                values_at_scales = all_probas[:, i, c].tolist()
                mitigated_probas[i, c] = richardson_extrapolate(
                    self.scale_factors, values_at_scales
                )

        # Clip to [0,1] and renormalise (extrapolation can give small negatives)
        mitigated_probas = np.clip(mitigated_probas, 0.0, None)
        row_sums = mitigated_probas.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        mitigated_probas /= row_sums

        return np.argmax(mitigated_probas, axis=1)

    def mitigated_predict_proba(self, vqc, X: np.ndarray) -> np.ndarray:
        """Same as mitigated_predict but returns probability array."""
        from noise.noise_model import build_noise_model, build_noisy_sampler_and_pm

        all_probas = []

        for scale in self.scale_factors:
            dep_1q = min(self.noise_config.get("depolarizing_1q", 0.001) * scale, 0.75)
            dep_2q = min(self.noise_config.get("depolarizing_2q", 0.01) * scale, 0.75)

            scaled_nm = build_noise_model(
                depolarizing_1q=dep_1q,
                depolarizing_2q=dep_2q,
                t1_us=self.noise_config.get("t1_us", 100.0),
                t2_us=self.noise_config.get("t2_us", 80.0),
                readout_error=self.noise_config.get("readout_error", 0.02),
            )
            scaled_sampler, scaled_pm = build_noisy_sampler_and_pm(scaled_nm, shots=1024)
            trained_vqc = _clone_vqc_with_sampler(vqc, scaled_sampler, scaled_pm)
            all_probas.append(trained_vqc.predict_proba(X))

        all_probas = np.array(all_probas)
        n_samples, n_classes = all_probas.shape[1], all_probas.shape[2]
        out = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for c in range(n_classes):
                out[i, c] = richardson_extrapolate(
                    self.scale_factors, all_probas[:, i, c].tolist()
                )

        out = np.clip(out, 0.0, None)
        out /= np.where(out.sum(1, keepdims=True) == 0, 1.0, out.sum(1, keepdims=True))
        return out


def _clone_vqc_with_sampler(fitted_vqc, new_sampler, new_pass_manager=None):
    """
    Returns a VQC with the same trained weights but a different sampler.
    Uses the public .weights attribute (set after .fit()).
    """
    from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
    from qiskit_machine_learning.algorithms import VQC

    try:
        from qiskit_algorithms.optimizers import COBYLA
    except ImportError:
        from qiskit.algorithms.optimizers import COBYLA

    fm = ZZFeatureMap(
        feature_dimension=fitted_vqc.feature_map.num_qubits, reps=2
    )
    ans = EfficientSU2(
        num_qubits=fitted_vqc.ansatz.num_qubits,
        reps=fitted_vqc.ansatz.reps,
        entanglement="linear",
        insert_barriers=False,
    )

    vqc_kwargs = dict(
        feature_map=fm,
        ansatz=ans,
        optimizer=COBYLA(maxiter=1),  # never called at inference
        sampler=new_sampler,
        initial_point=fitted_vqc.weights,
    )
    if new_pass_manager is not None:
        vqc_kwargs["pass_manager"] = new_pass_manager

    cloned = VQC(**vqc_kwargs)
    # Mark as fitted by setting internal weights directly
    cloned._fit_result = type("_FitResult", (), {"x": fitted_vqc.weights})()
    return cloned
