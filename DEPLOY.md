# Deployment - Quantum Sensor Fault Classifier API

An end-to-end service that classifies a vibration-sensor signal as **normal** or
**fault** using a 4-qubit variational quantum circuit (VQC), with a fast
classical RBF-SVM fallback.

```
raw signal (8 samples)  ->  preprocess (scale -> PCA(4) -> scale)  ->  VQC <Z>  ->  normal / fault
```

## Project layout (serving)

```
qsensor/
  signals.py     sensor-signal generation
  quantum.py     VQC circuit + EstimatorQNN (shared by train & serve)
  training.py    fit preprocessing, train VQC + SVM, save artifacts/
  inference.py   load artifacts, classify a signal
  api.py         FastAPI service
artifacts/       saved model (preprocessor, vqc_weights, svm, metadata) - created by training
Dockerfile, docker-compose.yml, requirements-serve.txt
```

## 1. Train the model (creates `artifacts/`)

```bash
python -m qsensor.training          # full (~10-12 min), best accuracy
python -m qsensor.training --fast   # quick (~4 min), slightly lower accuracy
```

This writes `artifacts/preprocessor.joblib`, `vqc_weights.npy`,
`classical_svm.joblib`, and `metadata.json`.

## 2a. Run the API locally (no Docker)

```bash
uvicorn qsensor.api:app --host 0.0.0.0 --port 8000
```
Open <http://localhost:8000/docs> for the interactive UI.

## 2b. Run with Docker

```bash
docker build -t quantum-sensor-vqc:1.0.0 .
docker run -p 8000:8000 quantum-sensor-vqc:1.0.0
# or:
docker compose up --build
```
The image bundles the trained `artifacts/`, so the container is self-contained.
(Train the model once before building, so `artifacts/` exists.)

## 3. Use the API

**Health**
```bash
curl http://localhost:8000/health
```

**Get a demo signal** (so you can try it without real data)
```bash
curl "http://localhost:8000/sample?fault=true"
```

**Predict** (VQC by default; `?model=classical` for the fast baseline)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal":[0.97,0.41,0.05,-0.22,-0.88,-0.41,0.18,0.12]}'
```
Response:
```json
{"label":1,"prediction":"fault","confidence":0.83,"model":"vqc","z_expectation":-0.66}
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health`  | liveness + loaded-model metadata |
| POST | `/predict` | classify a signal (`?model=vqc\|classical`) |
| GET  | `/sample`  | generate a demo signal (`?fault=true`) |
| GET  | `/docs`    | interactive OpenAPI UI |

## Notes

- **Inference runs on a quantum simulator** (exact statevector), so the VQC
  prediction is a real quantum computation (~tens of ms per request). For
  high-throughput production you can serve `?model=classical` (instant) and keep
  the VQC as the quantum showcase.
- The model is trained on **synthetic, physics-inspired** signals. For a real
  deployment, retrain `qsensor.training` on measured sensor data with the same
  8-sample format.
