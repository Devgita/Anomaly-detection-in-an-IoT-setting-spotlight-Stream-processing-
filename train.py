"""Train on reproducible fictional factory measurements (not real factory evidence)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import joblib

ROOT = Path(__file__).parent
FEATURES = ['temperature_c', 'humidity_pct', 'sound_db']

def sample(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    X = np.column_stack([rng.normal(65, 4, n), rng.normal(48, 7, n), rng.normal(73, 5, n)])
    for i in np.flatnonzero(y):
        f = rng.integers(0, 3)
        X[i, f] += rng.choice([-1, 1]) * [18, 29, 23][f]
    X[:, 1] = np.clip(X[:, 1], 0, 100)
    return pd.DataFrame(X, columns=FEATURES), y

if __name__ == '__main__':
    X, y = sample()
    cut = int(len(X) * 0.8)
    model = IsolationForest(n_estimators=150, contamination=0.08, random_state=42)
    model.fit(X.iloc[:cut])
    pred = (model.predict(X.iloc[cut:]) == -1).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y[cut:], pred, average='binary', zero_division=0)
    metrics = {'precision': round(float(p), 3), 'recall': round(float(r), 3), 'f1': round(float(f), 3), 'confusion_matrix': confusion_matrix(y[cut:], pred).tolist(), 'test_items': len(pred), 'note': 'Labels are synthetic and only validate the demonstration.'}
    ROOT.joinpath('artifacts').mkdir(exist_ok=True)
    joblib.dump({'model': model, 'features': FEATURES, 'version': '1.0'}, ROOT / 'artifacts' / 'model.joblib')
    (ROOT / 'artifacts' / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
