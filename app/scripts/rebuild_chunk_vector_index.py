from app.repositories.document_chunk_repository import rebuild_document_chunk_vector_index


def main():
    """Rebuild the document chunk vector index using the configured ANN type."""
    index_type = rebuild_document_chunk_vector_index()
    print(f"Rebuilt document chunk vector index using '{index_type}'.")


if __name__ == "__main__":
    main()
