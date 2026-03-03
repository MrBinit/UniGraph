import argparse
import json

from app.services.retrieval_service import retrieve_document_chunks


def _parse_metadata_filters(raw_filters: list[str]) -> dict[str, str]:
    """Parse repeated key=value CLI arguments into a metadata filter mapping."""
    filters: dict[str, str] = {}
    for raw_filter in raw_filters:
        if "=" not in raw_filter:
            raise ValueError(f"Invalid filter '{raw_filter}'. Expected key=value.")
        key, value = raw_filter.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid filter '{raw_filter}'. Expected key=value.")
        filters[key] = value
    return filters


def main():
    """Run one retrieval query and print latency plus the top-k chunk results."""
    parser = argparse.ArgumentParser(description="Evaluate vector retrieval over embedded chunks.")
    parser.add_argument("--query", required=True, help="Query text to embed and search.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest chunks to return.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Optional metadata filter in key=value form. Repeat for multiple filters.",
    )
    args = parser.parse_args()

    filters = _parse_metadata_filters(args.filter)
    result = retrieve_document_chunks(
        args.query,
        top_k=args.top_k,
        metadata_filters=filters or None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
