from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_health():
    assert client.get('/health').status_code == 200

def test_validation():
    response = client.post('/predict', json={'item_id': 'x', 'temperature_c': 65, 'humidity_pct': 120, 'sound_db': 70})
    assert response.status_code == 422

def test_prediction_when_trained():
    if client.get('/health').json()['status'] == 'ok':
        response = client.post('/predict', json={'item_id': 'x', 'temperature_c': 65, 'humidity_pct': 48, 'sound_db': 73})
        assert response.status_code == 200
        assert isinstance(response.json()['anomaly'], bool)
