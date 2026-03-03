import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.core.paths import resolve_project_path
from app.infra.bedrock_client import get_bedrock_runtime_client

settings = get_settings()


def _resolve_path(path_value: str) -> Path:
    """Resolve an embedding path relative to the project root when needed."""
    return resolve_project_path(path_value)


def _load_chunk_manifest(chunk_manifest_path: Path) -> dict:
    """Read one chunk manifest from disk and validate its basic shape."""
    payload = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Chunk manifest must be a JSON object: {chunk_manifest_path}")
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"Chunk manifest 'chunks' must be a list: {chunk_manifest_path}")
    return payload


def _embedding_output_path(chunk_manifest_path: Path, output_dir: Path) -> Path:
    """Map a chunk manifest path to its embedding output file path."""
    base_name = chunk_manifest_path.name.replace(".chunks.json", ".embeddings.json")
    if base_name == chunk_manifest_path.name:
        base_name = f"{chunk_manifest_path.stem}.embeddings.json"
    return output_dir / base_name


def _coerce_embedding(response_payload: dict) -> list[float]:
    """Extract and validate the embedding vector returned by Bedrock."""
    embedding = response_payload.get("embedding", [])
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("Bedrock response did not include a valid embedding list.")
    return [float(value) for value in embedding]


def embed_text(text: str) -> list[float]:
    """Generate a Bedrock embedding vector for one text input."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text to embed must be a non-empty string.")

    truncated = text.strip()[: settings.embedding.max_text_chars]
    client = get_bedrock_runtime_client()
    response = client.invoke_model(
        modelId=settings.embedding.model_id,
        body=json.dumps({"inputText": truncated}),
        contentType="application/json",
        accept="application/json",
    )
    response_payload = json.loads(response["body"].read())
    if not isinstance(response_payload, dict):
        raise ValueError("Bedrock response body must decode to a JSON object.")
    return _coerce_embedding(response_payload)


async def aembed_text(text: str) -> list[float]:
    """Generate a Bedrock embedding vector without blocking the event loop."""
    return await asyncio.to_thread(embed_text, text)


def embed_chunk_manifest(chunk_manifest_path: Path, output_dir: Path | None = None) -> Path:
    """Embed every chunk in one chunk manifest and persist an embedding manifest."""
    payload = _load_chunk_manifest(chunk_manifest_path)
    destination_dir = output_dir or _resolve_path(settings.embedding.output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    embedded_chunks = []
    embedding_dimensions = 0
    for chunk in payload.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        content = chunk.get("content", "")
        if not isinstance(content, str):
            continue
        vector = embed_text(content)
        embedding_dimensions = len(vector)
        embedded_chunk = dict(chunk)
        embedded_chunk["embedding"] = vector
        embedded_chunks.append(embedded_chunk)

    output_payload = dict(payload)
    output_payload["embedding_provider"] = settings.embedding.provider
    output_payload["embedding_region"] = settings.embedding.region_name
    output_payload["embedding_model"] = settings.embedding.model_id
    output_payload["embedding_dimensions"] = embedding_dimensions
    output_payload["embedded_at"] = datetime.now(timezone.utc).isoformat()
    output_payload["chunks"] = embedded_chunks

    output_path = _embedding_output_path(chunk_manifest_path, destination_dir)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return output_path


async def aembed_chunk_manifest(chunk_manifest_path: Path, output_dir: Path | None = None) -> Path:
    """Embed one chunk manifest without blocking the event loop."""
    return await asyncio.to_thread(embed_chunk_manifest, chunk_manifest_path, output_dir)


def embed_configured_chunk_manifests() -> list[Path]:
    """Embed all configured chunk manifests and return the output file paths."""
    if not settings.embedding.enabled:
        return []

    input_dir = _resolve_path(settings.embedding.input_dir)
    output_dir = _resolve_path(settings.embedding.output_dir)
    output_paths = []
    for chunk_manifest_path in sorted(input_dir.glob(settings.embedding.glob_pattern)):
        if chunk_manifest_path.is_file():
            output_paths.append(embed_chunk_manifest(chunk_manifest_path, output_dir))
    return output_paths


async def aembed_configured_chunk_manifests() -> list[Path]:
    """Embed all configured chunk manifests without blocking the event loop."""
    if not settings.embedding.enabled:
        return []

    input_dir = _resolve_path(settings.embedding.input_dir)
    output_dir = _resolve_path(settings.embedding.output_dir)
    manifest_paths = [
        chunk_manifest_path
        for chunk_manifest_path in sorted(input_dir.glob(settings.embedding.glob_pattern))
        if chunk_manifest_path.is_file()
    ]
    return await asyncio.gather(
        *(aembed_chunk_manifest(chunk_manifest_path, output_dir) for chunk_manifest_path in manifest_paths)
    )
