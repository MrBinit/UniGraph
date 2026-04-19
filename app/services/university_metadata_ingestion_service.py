import json
from pathlib import Path
from app.core.config import get_settings
from app.core.paths import resolve_project_path
from app.schemas.university_metadata_schema import UniversityMetadataIngestionPayload

settings = get_settings()


def _load_payload(path: Path) -> UniversityMetadataIngestionPayload:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Metadata payload must be a JSON object: {path}")
    return UniversityMetadataIngestionPayload.model_validate(raw)


def ingest_university_metadata_file(path_value: str | Path) -> dict[str, int]:
    """Load one metadata JSON payload and upsert into normalized Postgres tables."""
    if not settings.postgres.enabled:
        return {"universities": 0, "programs": 0, "source_records": 0}
    from app.repositories.university_metadata_repository import ingest_university_metadata_payload

    payload_path = resolve_project_path(path_value)
    payload = _load_payload(payload_path)
    return ingest_university_metadata_payload(payload)
