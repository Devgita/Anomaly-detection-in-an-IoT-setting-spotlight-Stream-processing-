"""HTTP scoring service; item-level prediction from aggregated sensor measurements."""
from pathlib import Path
from datetime import datetime, timezone
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
app = FastAPI(title='Wind Turbine Factory Anomaly API', version='1.0')
ARTIFACT = Path(__file__).resolve().parents[1] / 'artifacts' / 'model.joblib'
loaded = joblib.load(ARTIFACT) if ARTIFACT.exists() else None

class Measurement(BaseModel):
    item_id: str = Field(min_length=1, max_length=100)
    temperature_c: float = Field(ge=-40, le=200)
    humidity_pct: float = Field(ge=0, le=100)
    sound_db: float = Field(ge=0, le=150)

@app.get('/health')
def health():
    return {'status': 'ok' if loaded else 'model_missing', 'model_version': loaded['version'] if loaded else None}

@app.post('/predict')
def predict(data: Measurement):
    if loaded is None:
        raise HTTPException(status_code=503, detail='Run python train.py before serving')
    row = pd.DataFrame([{f: getattr(data, f) for f in loaded['features']}])
    raw = float(loaded['model'].decision_function(row)[0])
    # IsolationForest score < 0 is anomalous. This score is NOT a calibrated probability.
    flag = bool(loaded['model'].predict(row)[0] == -1)
    result = {'item_id': data.item_id, 'anomaly': flag, 'anomaly_score': round(-raw, 5), 'model_version': loaded['version'], 'scored_at': datetime.now(timezone.utc).isoformat()}
    logging.info('item=%s anomaly=%s model=%s', data.item_id, flag, loaded['version'])
    return result
