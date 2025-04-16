from typing import List, Dict, Optional
from datetime import datetime, timezone

class ModelRecord:
    """
    Represents a model in the LLM Model Library, including its file system location and metadata.
    This class is suitable for use with ORM or as a DTO for service/controller layers.
    """
    def __init__(
        self,
        id: Optional[int],
        publisher: str,
        model_name: str,
        file_paths: List[str],  # Paths relative to library root
        framework: str,
        metadata: Dict,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        is_sharded: bool = False,
    ):
        self.id = id
        self.publisher = publisher
        self.model_name = model_name
        self.file_paths = file_paths
        self.framework = framework
        self.metadata = metadata
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.is_sharded = is_sharded

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "publisher": self.publisher,
            "model_name": self.model_name,
            "file_paths": self.file_paths,
            "framework": self.framework,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_sharded": self.is_sharded,
        }

    @staticmethod
    def from_dict(data: Dict) -> 'ModelRecord':
        return ModelRecord(
            id=data.get("id"),
            publisher=data["publisher"],
            model_name=data["model_name"],
            file_paths=data["file_paths"],
            framework=data["framework"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            is_sharded=data.get("is_sharded", False),
        )
