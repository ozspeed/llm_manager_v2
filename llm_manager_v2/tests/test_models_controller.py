import pytest
from llm_manager_v2.app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_get_models_empty(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.is_json
    assert resp.get_json() == []

def test_add_model_missing_fields(client):
    resp = client.post("/api/models", json={})
    assert resp.status_code in (400, 422)

def test_add_and_get_model(client):
    # Create a dummy file for upload
    import io
    file_content = b"test model content"
    data = {
        "name": "testmodel",
        "framework": "pytorch",
        "file": (io.BytesIO(file_content), "model.bin")
    }
    resp = client.post("/api/models", data=data, content_type='multipart/form-data')
    assert resp.status_code in (200, 201)
    get_resp = client.get("/api/models")
    models = get_resp.get_json()
    assert any(m.get("name") == "testmodel" for m in models)

def test_delete_nonexistent_model(client):
    resp = client.delete("/api/models/9999")
    assert resp.status_code in (404, 400)
