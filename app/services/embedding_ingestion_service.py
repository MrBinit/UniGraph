import json
from pathlib import Path
from app.core.config import get_settings
from app.core.paths import resolve_project_path

settings = get_settings()


def _resolve_path(path_value: str) -> Path:
    """Resolve an ingestion path relative to the project root when needed."""
    return resolve_project_path(path_value)


def _load_embedding_manifest(path: Path) -> dict:
    """Read one embedding manifest from disk and validate its base structure."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding manifest must be a JSON object: {path}")
    return payload


def ingest_embedding_manifest_file(path: Path) -> int:
    """Load one embedding manifest file and ingest its chunks into Postgres."""
    if not settings.postgres.enabled:
        return 0
    from app.repositories.document_chunk_repository import ingest_embedding_manifest

    payload = _load_embedding_manifest(path)
    return ingest_embedding_manifest(payload)


def ingest_configured_embedding_manifests() -> dict:
    """Ingest all configured embedding manifest files into Postgres."""
    input_dir = _resolve_path(settings.embedding.output_dir)
    processed_files = 0
    processed_chunks = 0

    for manifest_path in sorted(input_dir.glob("*.embeddings.json")):
        if not manifest_path.is_file():
            continue
        processed_chunks += ingest_embedding_manifest_file(manifest_path)
        processed_files += 1

    return {
        "processed_files": processed_files,
        "processed_chunks": processed_chunks,
    }
