# Noise-Resilient Classification of Quantum Sensor Signals using Variational Quantum Circuits

QMET PARIMANA UG Summer Internship 2026, Indian Institute of Technology Patna
Author: Aaditya Sah (`aaditya_2312res02@iitp.ac.in`)
Supervisor: Dr. Nutan Kumar Tomar (`nktomar@iitp.ac.in`)

A four-qubit quantum machine-learning pipeline that classifies noisy sensor
signals (a healthy versus a faulty bearing) and measures how the models hold up
under hardware noise. It compares a quantum-kernel classifier (QSVC) and a
trainable variational quantum circuit (VQC) against classical baselines, then
stresses the trained VQC under a device-noise model with Zero-Noise
Extrapolation (ZNE).

The repository has two parts: the **research pipeline** (`experiments/`, the
paper) and a **deployable REST API** (`qsensor/`) that serves the trained model.

## Deployment (REST API)

The `qsensor/` package wraps the model in a FastAPI service. Train once, then
serve:

```bash
python -m qsensor.training            # trains VQC + SVM, saves artifacts/
uvicorn qsensor.api:app               # serves http://localhost:8000/docs
# or containerised:  docker compose up --build
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal":[0.05,-0.3,0.9,-0.6,0.7,-0.9,0.4,-0.2]}'
# -> {"prediction":"fault","confidence":0.56,"model":"vqc",...}
```

Full instructions and endpoint reference: [DEPLOY.md](DEPLOY.md).

## Results (ideal statevector simulator, seed 42)

| Model | Type | Accuracy | F1 |
|-------|------|----------|-----|
| SVM-RBF | Classical | 100.0% | 1.000 |
| Logistic Regression | Classical | 100.0% | 1.000 |
| Multi-Layer Perceptron | Classical | 95.8% | 0.957 |
| QSVC (quantum kernel) | Quantum | 97.2% | 0.972 |
| VQC (variational) | Quantum | 90.3% | 0.892 |

The VQC uses data re-uploading (three blocks) with an average-Z expectation
readout, trained by COBYLA on the exact simulator. Under increasing two-qubit
depolarizing error the trained VQC holds near 89-92% up to p2 = 0.05 and degrades
to 56.9% at p2 = 0.15. At p2 = 0.02 the unfolded accuracy is 91.7%.

The ideal-simulator results (classical, QSVC, VQC) are computed exactly and are
bit-reproducible run to run. The noisy-simulator points (noise sweep, ZNE) use a
fixed seed but carry a few-percent variation from the density-matrix backend;
multi-seed error bars are planned for the final report. The complete write-up is
in `paper/research_paper.pdf`.

## Architecture

```
quantum_sensor_vqc/
├── README.md                 this file
├── requirements.txt          Python dependencies
├── config/                   YAML experiment configs (base_config, fast_config)
├── data/                     sensor-signal dataset generator
├── models/                   VQC and classical model factories
├── noise/                    device noise models (depolarizing, T1/T2, readout)
├── mitigation/               Zero-Noise Extrapolation
├── evaluation/               metrics and plotting helpers
├── experiments/              runnable experiments
│   └── month1_experiment.py  self-contained run that produces every paper number
├── scripts/                  figure generators for the paper
│   ├── make_paper_figures.py pipeline, signals, PCA, circuits, ZNE figures
│   └── make_convergence.py   VQC training-convergence curve
├── results/                  CSV results and figures (PNG + PDF)
├── paper/                    research_paper.tex and compiled PDF
├── docs/                     Word progress report and its generator
├── saved_models/             trained model artifacts
└── train.py, evaluate.py, validate.py   original config-driven CV pipeline
```

Two pipelines live here. `experiments/month1_experiment.py` is the streamlined
four-qubit run behind the paper. The `train.py` / `evaluate.py` /
`experiments/run_experiments.py` framework is the original config-driven pipeline
with k-fold cross-validation and multiple seeds, kept for the planned
statistical study in the final report.

## How to run

```bash
# 1. environment
python -m venv venv
venv\Scripts\activate            # Windows  (use: source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt

# 2. reproduce the paper results (deterministic, ~10 min on CPU)
python -m experiments.month1_experiment

# 3. regenerate the paper figures
python scripts/make_paper_figures.py
python scripts/make_convergence.py

# 4. build the paper (needs a LaTeX engine, e.g. tectonic or pdflatex)
tectonic paper/research_paper.tex

# 5. build the Word progress report (needs Node.js and the docx package)
node docs/generate_report.js
```

## Notes

- The simulation runs on CPU. A four-qubit statevector is tiny, so a GPU gives
  no speed-up here; the cost is the number of small circuit evaluations.
- Single-threaded BLAS is set inside the experiment for reproducibility.
- Code and results: https://github.com/aadityasah1/Quantum_Sensor_VQC
