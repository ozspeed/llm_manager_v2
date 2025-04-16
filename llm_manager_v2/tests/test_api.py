"""
[V2-6.1] API endpoint tests for LLM Manager v2
"""
def test_huggingface_search(test_client):
    resp = test_client.get("/api/huggingface/search?query=llama")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "results" in data
    assert data["query"] == "llama"

def test_huggingface_trending(test_client):
    resp = test_client.get("/api/huggingface/trending")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "results" in data

def test_huggingface_recent_searches(test_client):
    resp = test_client.get("/api/huggingface/recent-searches")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)

def test_settings(test_client):
    resp = test_client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)
    assert "SECRET_KEY" in data

def test_models(test_client):
    resp = test_client.get("/api/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
