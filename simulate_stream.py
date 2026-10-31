"""Simulate streaming sensor events; aggregate one window into one item prediction."""
import argparse
import time
import requests
from train import sample

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--interval', type=float, default=0.5)
parser.add_argument('--items', type=int, default=30)
args = parser.parse_args()
X, _ = sample(n=args.items, seed=2026)
for i, row in X.iterrows():
    payload = {'item_id': f'ITEM-{i+1:05d}', **{k: float(v) for k, v in row.items()}}
    try:
        r = requests.post(args.url + '/predict', json=payload, timeout=5)
        r.raise_for_status()
        print(r.json(), flush=True)
    except requests.RequestException as e:
        print(f'HTTP request failed: {e}', flush=True)
    time.sleep(args.interval)
