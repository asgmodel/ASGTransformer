from fastapi.testclient import TestClient
import asg_transformer.api.main as api

class DummyService: pass

def test_root_and_health(monkeypatch):
    monkeypatch.setattr(api, "get_service", lambda: DummyService())
    with TestClient(api.app) as client:
        assert client.get("/").status_code == 200
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
