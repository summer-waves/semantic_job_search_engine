import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

def test_root_is_running(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["total_jobs"] == 43484

def test_search_returns_results(client):
    response = client.get("/search?q=machine+learning+engineer")
    assert response.status_code == 200
    assert len(response.json()["results"]) > 0

def test_search_top_k(client):
    response = client.get("/search?q=data+analyst&top_k=3")
    assert response.status_code == 200
    assert len(response.json()["results"]) == 3

def test_search_scores_between_zero_and_one(client):
    response = client.get("/search?q=software+engineer&top_k=5")
    results = response.json()["results"]
    for r in results:
        assert 0.0 <= r["score"] <= 1.0