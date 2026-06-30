"""Quantum Sensor Fault Classifier - deployable package.

A 4-qubit variational quantum classifier (plus a classical baseline) that labels
a vibration-sensor signal as `normal` or `fault`, served behind a REST API.

Layout:
    signals.py    raw sensor-signal generation
    quantum.py    VQC circuit + EstimatorQNN (shared by training and serving)
    training.py   fit preprocessing, train models, save an artifact bundle
    inference.py  load the bundle and classify a new signal
    api.py        FastAPI service
"""
__version__ = "1.0.0"
