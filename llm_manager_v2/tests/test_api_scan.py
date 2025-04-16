import pytest
from llm_manager_v2.app import create_app
from llm_manager_v2.models.settings_service import SettingsService
import tempfile
import os

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_scan_library_api(client):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup fake library
        pub = os.path.join(tmpdir, "TestPublisher")
        mdl = os.path.join(pub, "TestModel")
        os.makedirs(mdl)
        model_file = os.path.join(mdl, "model.bin")
        with open(model_file, "w") as f:
            f.write("fake content")
        # Set library_root
        settings = SettingsService()
        settings.set("library_root", tmpdir)
        # POST /api/models/scan
        rv = client.post("/api/models/scan")
        assert rv.status_code == 200
        data = rv.get_json()
        assert isinstance(data, list)
        assert any(
            m['publisher'] == "TestPublisher" and m['model_name'] == "TestModel"
            for m in data
        )
