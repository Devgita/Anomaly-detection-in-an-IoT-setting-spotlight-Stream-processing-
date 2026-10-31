# IoT Anomaly Detection — Oral Exam Project

**Task 1: Stream processing and anomaly detection for wind turbine component manufacturing.**

## Reproduce locally (Python 3.10+)

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive REST documentation. In another terminal:

```bash
python simulate_stream.py --items 30 --interval 0.5
pytest -q
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"item_id":"ITEM-1","temperature_c":65,"humidity_pct":48,"sound_db":73}'
```

## What is and is not implemented

Implemented: deterministic synthetic measurements, offline training, holdout metrics, serialized model, typed REST scoring, request validation, health endpoint, repeatable event simulator, automated API tests. The simulator sends one *item-level aggregated reading per request*; it is a continuous sequence of events, **not** a real sensor message broker or raw high-frequency event-time windowing system.

Proposed production extensions (not implemented): MQTT/Kafka ingestion, Flink/stream windows keyed by item ID, durable storage, API authentication, observability/alerts, drift detection, model registry, retraining pipeline, autoscaling and idempotency.

## Model interpretation

Isolation Forest flags outliers based on three numeric features. `anomaly_score = -decision_function` means higher scores are more suspicious; **0 is the default decision boundary**, not a probability. Synthetic anomaly labels are reserved for evaluation and are not genuine faulty-part labels. Evaluate real production labels and false-alarm costs before deployment.

## Public code link for the exam

This ZIP is reproducible offline, but it has **not** been uploaded to a public repository. Upload the project folder to GitHub/GitLab and replace the placeholder link in the presentation before submitting. Never claim a repository URL that has not been created.
