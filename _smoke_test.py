"""Minimal smoke test — 4 qubits, 1 rep, 15 SPSA iters."""
import sys, time, numpy as np
sys.path.insert(0, ".")

print("Starting minimal pipeline test...")
sys.stdout.flush()

from data.generate_data import generate_sensor_data
from models.vqc_model import create_vqc
from noise.noise_model import build_ideal_sampler_and_pm, build_noisy_sampler_and_pm, noise_model_from_config
from evaluation.metrics import compute_metrics
from sklearn.model_selection import train_test_split
import yaml

print("Imports OK")
sys.stdout.flush()

with open("config/base_config.yaml") as f:
    config = yaml.safe_load(f)

X, y = generate_sensor_data(n_samples=200, n_features=4, snr=3.0, seed=42)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data: {X_tr.shape} train, {X_val.shape} val")
sys.stdout.flush()

# ── Test 1: ideal sampler ──────────────────────────────────────────────────────
print("\nTest 1: Ideal sampler (StatevectorSampler)")
sampler, pm = build_ideal_sampler_and_pm(shots=256, seed=42)
history = []
def cb(nfev, x, fx, dx, accept=None):
    history.append(float(fx))

t0 = time.time()
vqc = create_vqc(n_qubits=4, reps=1, max_iter=15, sampler=sampler, pass_manager=pm, seed=42, callback=cb)
vqc.fit(X_tr, y_tr)
elapsed = time.time() - t0
y_pred = vqc.predict(X_val)
m = compute_metrics(y_val, y_pred, None)
print(f"  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  time={elapsed:.1f}s  iters={len(history)}")
sys.stdout.flush()

# ── Test 2: noisy sampler ─────────────────────────────────────────────────────
print("\nTest 2: Noisy sampler (AerSamplerV2 + PassManager)")
nm = noise_model_from_config(config)
sampler2, pm2 = build_noisy_sampler_and_pm(nm, shots=256, seed=42)
history2 = []
def cb2(nfev, x, fx, dx, accept=None):
    history2.append(float(fx))

t0 = time.time()
vqc2 = create_vqc(n_qubits=4, reps=1, max_iter=15, sampler=sampler2, pass_manager=pm2, seed=42, callback=cb2)
vqc2.fit(X_tr, y_tr)
elapsed2 = time.time() - t0
y_pred2 = vqc2.predict(X_val)
m2 = compute_metrics(y_val, y_pred2, None)
print(f"  acc={m2['accuracy']:.4f}  f1={m2['f1']:.4f}  time={elapsed2:.1f}s  iters={len(history2)}")
sys.stdout.flush()

print("\nSMOKE TEST PASSED — both ideal and noisy samplers work correctly")
