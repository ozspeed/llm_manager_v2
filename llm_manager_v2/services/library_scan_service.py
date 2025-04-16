import os
from datetime import datetime, timezone
from typing import List
from llm_manager_v2.models.model_record import ModelRecord
from llm_manager_v2.models.settings_service import SettingsService

class LibraryScanService:
    """
    Scans the model library root directory and returns ModelRecord instances for each discovered model.
    """
    def __init__(self, settings_service: SettingsService):
        self.settings = settings_service

    def scan_library_root(self) -> List[ModelRecord]:
        library_root = self.settings.get("library_root")
        if not library_root or not os.path.isdir(library_root):
            raise ValueError(f"Invalid or unset library_root: {library_root}")
        model_records = []
        for publisher in os.listdir(library_root):
            publisher_path = os.path.join(library_root, publisher)
            if not os.path.isdir(publisher_path):
                continue
            for model_name in os.listdir(publisher_path):
                model_path = os.path.join(publisher_path, model_name)
                if not os.path.isdir(model_path):
                    continue
                files = [
                    os.path.join(publisher, model_name, f)
                    for f in os.listdir(model_path)
                    if os.path.isfile(os.path.join(model_path, f))
                ]
                if files:
                    record = ModelRecord(
                        id=None,
                        publisher=publisher,
                        model_name=model_name,
                        file_paths=files,
                        framework="unknown",  # Optionally infer from extension
                        metadata={},
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                        is_sharded=len(files) > 1
                    )
                    model_records.append(record)
        return model_records
