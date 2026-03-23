import argparse
from app.services.university_metadata_ingestion_service import ingest_university_metadata_file


def main():
    """Ingest one university metadata payload JSON file into Postgres."""
    parser = argparse.ArgumentParser(
        description="Ingest university metadata payload into Postgres."
    )
    parser.add_argument("payload_path", help="Path to the metadata JSON payload.")
    args = parser.parse_args()

    result = ingest_university_metadata_file(args.payload_path)
    total = sum(result.values())
    print(f"Ingested {total} row payload items.")
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
