from llm_manager_v2.models.model_record import ModelRecord
from datetime import datetime, timezone

def test_model_record_serialization():
    now = datetime.now(timezone.utc)
    record = ModelRecord(
        id=1,
        publisher="testpub",
        model_name="testmodel",
        file_paths=["testpub/testmodel/model.bin"],
        framework="pytorch",
        metadata={"param": "val"},
        created_at=now,
        updated_at=now,
        is_sharded=False,
    )
    d = record.to_dict()
    assert d["publisher"] == "testpub"
    r2 = ModelRecord.from_dict(d)
    assert r2.publisher == record.publisher
    assert r2.file_paths == record.file_paths
    assert r2.framework == record.framework
    assert r2.metadata == record.metadata
    assert r2.is_sharded == record.is_sharded
