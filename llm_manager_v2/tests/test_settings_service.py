from llm_manager_v2.models.settings_service import SettingsService
import os
import tempfile

def test_settings_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "settings.db")
        service = SettingsService(db_path)
        assert service.get("foo") is None
        service.set("foo", "bar")
        assert service.get("foo") == "bar"
        service.set("foo", "baz")
        assert service.get("foo") == "baz"
