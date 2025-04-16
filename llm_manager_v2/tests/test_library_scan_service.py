import os
import tempfile
from llm_manager_v2.models.model_record import ModelRecord
from llm_manager_v2.models.settings_service import SettingsService
from llm_manager_v2.services.library_scan_service import LibraryScanService

def test_scan_library_root_discovers_models():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup fake library: Publisher/ModelName/model.bin
        pub = os.path.join(tmpdir, "TestPublisher")
        mdl = os.path.join(pub, "TestModel")
        os.makedirs(mdl)
        model_file = os.path.join(mdl, "model.bin")
        with open(model_file, "w") as f:
            f.write("fake content")
        # SettingsService points to tmpdir
        db_path = os.path.join(tmpdir, "settings.db")
        settings = SettingsService(db_path)
        settings.set("library_root", tmpdir)
        scan_service = LibraryScanService(settings)
        records = scan_service.scan_library_root()
        assert len(records) == 1
        rec = records[0]
        assert rec.publisher == "TestPublisher"
        assert rec.model_name == "TestModel"
        assert "TestPublisher/TestModel/model.bin" in rec.file_paths
