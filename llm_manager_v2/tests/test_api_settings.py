import pytest
from llm_manager_v2.app import create_app
from llm_manager_v2.models.settings_service import SettingsService

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_settings_get_put(client):
    # PUT a value
    rv = client.put('/api/settings/testkey', json={"value": "abc"})
    assert rv.status_code == 200
    # GET the value
    rv = client.get('/api/settings/testkey')
    assert rv.status_code == 200
    data = rv.get_json()
    assert data['value'] == "abc"
    # GET a missing key
    rv = client.get('/api/settings/doesnotexist')
    assert rv.status_code == 404
