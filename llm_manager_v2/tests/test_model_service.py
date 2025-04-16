import pytest
from llm_manager_v2.services import model_service
import tempfile
import os

# These tests assume the underlying add_model, get_all_models, delete_model are properly mocked or use a test database.

def test_list_models_runs():
    result = model_service.list_models()
    assert isinstance(result, tuple)
    assert isinstance(result[0], list)

def test_add_model_service_saves_file(tmp_path):
    class DummyFile:
        def __init__(self, filename):
            self.filename = filename
            self.saved = False
        def save(self, dest):
            self.saved = True
            with open(dest, 'w') as f:
                f.write('test')
    dummy = DummyFile("foo.bin")
    name = "test"
    framework = "pytorch"
    result, status = model_service.add_model_service(name, framework, dummy)
    assert status in (200, 201)
    assert dummy.saved
    assert os.path.exists(os.path.join("models_uploads", "foo.bin"))
    os.remove(os.path.join("models_uploads", "foo.bin"))

def test_get_model_details_not_found():
    resp, code = model_service.get_model_details(999999)
    assert code == 404
    assert "not found" in resp["error"].lower()

def test_delete_model_service_runs():
    # Should return a tuple (dict, int), even if model doesn't exist
    resp, code = model_service.delete_model_service(999999)
    assert isinstance(resp, dict)
    assert isinstance(code, int)
