from app.services.embedding_service import embed_configured_chunk_manifests


def main():
    """Embed all configured chunk manifests and print the written output files."""
    output_paths = embed_configured_chunk_manifests()
    print(f"Embedded {len(output_paths)} chunk manifest(s).")
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
