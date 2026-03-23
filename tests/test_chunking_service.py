import json
from pathlib import Path

from app.services.chunking_service import (
    build_chunk_records,
    recursive_chunk_text,
    write_chunk_file,
)


def test_recursive_chunk_text_uses_recursive_boundaries_and_limits():
    text = (
        "Paragraph one explains the university research profile in detail.\n\n"
        "Paragraph two covers labs, professors, and coursework in artificial intelligence.\n\n"
        "Paragraph three covers funding, admissions, and program structure for graduate study."
    )

    chunks = recursive_chunk_text(
        text,
        chunk_size_chars=90,
        chunk_overlap_chars=15,
        separators=["\n\n", "\n", ". ", " ", ""],
        min_chunk_chars=20,
    )

    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 90 for chunk in chunks)


def test_recursive_chunk_text_respects_structural_section_boundaries():
    text = (
        "Bachelor of Science in Computer Science\n\n"
        "This section describes the bachelor program in detail and its admission rules.\n\n"
        "Master of Science in Artificial Intelligence\n\n"
        "This section describes the master program in detail and its research focus."
    )

    chunks = recursive_chunk_text(
        text,
        chunk_size_chars=500,
        chunk_overlap_chars=50,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        min_chunk_chars=20,
    )

    assert len(chunks) == 2
    assert "Bachelor of Science" in chunks[0]
    assert "Master of Science" not in chunks[0]
    assert "Master of Science" in chunks[1]


def test_recursive_chunk_text_avoids_mid_word_hard_cuts():
    text = " ".join(f"token{i}" for i in range(200))

    chunks = recursive_chunk_text(
        text,
        chunk_size_chars=80,
        chunk_overlap_chars=20,
        separators=[""],
        min_chunk_chars=10,
    )

    assert len(chunks) >= 2
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert not chunks[1].startswith("ken")


def test_recursive_chunk_text_merges_tiny_heading_chunk_into_next_chunk():
    text = (
        "Master of Science in Artificial Intelligence\n\n"
        "This paragraph contains the actual content for the program "
        "and should stay with its heading."
    )

    chunks = recursive_chunk_text(
        text,
        chunk_size_chars=300,
        chunk_overlap_chars=40,
        separators=["\n\n", "\n", ". ", " ", ""],
        min_chunk_chars=300,
        merge_forward_below_chars=250,
    )

    assert len(chunks) == 1
    assert "Master of Science in Artificial Intelligence" in chunks[0]
    assert "actual content for the program" in chunks[0]


def test_write_chunk_file_persists_chunk_manifest(tmp_path: Path):
    source_path = tmp_path / "university_test.md"
    source_path.write_text(
        "This university focuses on AI systems.\n\n"
        "It offers strong labs in robotics, ML infrastructure, and security.",
        encoding="utf-8",
    )

    chunk_records = build_chunk_records(
        source_path,
        source_path.read_text(encoding="utf-8"),
        chunk_size_chars=70,
        chunk_overlap_chars=10,
        separators=["\n\n", "\n", ". ", " ", ""],
        min_chunk_chars=10,
        merge_forward_below_chars=5,
    )
    output_path = write_chunk_file(
        source_path,
        chunk_records,
        tmp_path / "chunks",
        chunk_size_chars=70,
        chunk_overlap_chars=10,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.name == "university_test.chunks.json"
    assert payload["source_file"] == "university_test.md"
    assert payload["chunk_count"] == len(chunk_records)
    assert "document_metadata" in payload
    assert payload["chunks"][0]["chunk_id"].startswith("university_test:")


def test_build_chunk_records_attaches_document_and_chunk_metadata(tmp_path: Path):
    source_path = tmp_path / "sample_university.md"
    source_path.write_text(
        "Sample University of AI\n\n"
        "Location: Boston, United States\n"
        "Founded: 1970\n"
        "Type: Private Research University\n\n"
        "Master of Science in Artificial Intelligence\n\n"
        "Program Overview\n\n"
        "This program focuses on language models, retrieval, and systems engineering.",
        encoding="utf-8",
    )

    chunk_records = build_chunk_records(
        source_path,
        source_path.read_text(encoding="utf-8"),
        chunk_size_chars=900,
        chunk_overlap_chars=120,
        separators=["\n\n", "\n", ". ", " ", ""],
        min_chunk_chars=300,
        merge_forward_below_chars=250,
    )

    assert chunk_records
    metadata = chunk_records[0]["metadata"]
    assert metadata["document_title"] == "Sample University of AI"
    assert metadata["university"] == "Sample University of AI"
    assert metadata["city"] == "Boston"
    assert metadata["country"] == "United States"

    assert any(record["metadata"]["degree_level"] == "masters" for record in chunk_records)
    assert any(record["metadata"]["entity_type"] == "program" for record in chunk_records)
