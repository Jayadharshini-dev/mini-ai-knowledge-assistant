from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "config" in data
    config = data["config"]
    assert config["EMBEDDING_MODEL"] == "BAAI/bge-small-en-v1.5"
    assert config["VECTOR_DIM"] == 384
    assert config["INDEX_TYPE"] == "IndexFlatIP"
    assert "GEMINI_API_KEY" not in config
    assert "GEMINI_API_KEY_SET" in config
