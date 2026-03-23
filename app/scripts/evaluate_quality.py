import argparse
import asyncio
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.services.llm_service import generate_response  # noqa: E402
from app.services.quality_metrics_service import (  # noqa: E402
    aggregate_metric_rows,
    generation_metrics,
    retrieval_metrics,
)
from app.services.retrieval_service import retrieve_document_chunks  # noqa: E402


def _load_eval_cases(dataset_path: Path) -> list[dict]:
    """Load eval cases from a JSON array file or JSONL file."""
    if dataset_path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("JSON dataset must be an array of case objects.")
    return data


def _validate_case(case: dict, index: int) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"Case #{index} must be a JSON object.")
    query = case.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Case #{index} is missing required 'query' string.")


async def _evaluate_case(
    case: dict,
    *,
    default_top_k: int,
    evaluate_generation: bool,
    default_user_id: str,
) -> dict:
    query = case["query"]
    top_k = int(case.get("top_k", default_top_k))
    metadata_filters = case.get("metadata_filters")
    if metadata_filters is not None and not isinstance(metadata_filters, dict):
        raise ValueError("'metadata_filters' must be an object when provided.")

    retrieval_result = retrieve_document_chunks(
        query,
        top_k=top_k,
        metadata_filters=metadata_filters,
    )
    retrieved_results = retrieval_result.get("results", [])
    relevant_ids = case.get("relevant_chunk_ids", [])
    if relevant_ids is None:
        relevant_ids = []
    if not isinstance(relevant_ids, list):
        raise ValueError("'relevant_chunk_ids' must be a list when provided.")

    retrieval_row = retrieval_metrics(retrieved_results, relevant_ids)

    generation_row: dict[str, float] = {}
    answer: str | None = None
    if evaluate_generation:
        if isinstance(case.get("generated_answer"), str):
            answer = case["generated_answer"]
        else:
            user_id = case.get("user_id", default_user_id)
            answer = await generate_response(str(user_id), query)
        generation_row = generation_metrics(
            query=query,
            answer=answer,
            expected_answer=case.get("expected_answer"),
            retrieved_results=retrieved_results,
        )

    return {
        "id": case.get("id"),
        "query": query,
        "retrieval": retrieval_row,
        "generation": generation_row,
        "answer": answer,
        "top_k": top_k,
        "retrieval_strategy": retrieval_result.get("retrieval_strategy"),
        "timings_ms": retrieval_result.get("timings_ms", {}),
    }


async def _run(args: argparse.Namespace) -> dict:
    dataset_path = Path(args.dataset).resolve()
    cases = _load_eval_cases(dataset_path)
    for index, case in enumerate(cases, start=1):
        _validate_case(case, index)

    evaluated_cases = []
    retrieval_rows = []
    generation_rows = []

    for case in cases:
        evaluated = await _evaluate_case(
            case,
            default_top_k=args.top_k,
            evaluate_generation=args.with_generation,
            default_user_id=args.user_id,
        )
        evaluated_cases.append(evaluated)
        retrieval_rows.append(evaluated["retrieval"])
        if evaluated["generation"]:
            generation_rows.append(evaluated["generation"])

    summary = {
        "dataset": str(dataset_path),
        "total_cases": len(evaluated_cases),
        "retrieval_metrics": aggregate_metric_rows(retrieval_rows),
        "generation_metrics": aggregate_metric_rows(generation_rows),
    }
    return {
        "summary": summary,
        "cases": evaluated_cases,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and generation quality over a dataset.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSON array or JSONL file of evaluation cases.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Default top-k to use for retrieval when per-case top_k is not set.",
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Evaluate generation metrics. Uses case.generated_answer or calls the model.",
    )
    parser.add_argument(
        "--user-id",
        default="eval-user",
        help="Default user_id for generation calls when not specified per case.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path. If omitted, prints JSON to stdout.",
    )
    args = parser.parse_args()

    result = asyncio.run(_run(args))
    rendered = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote quality evaluation report: {output_path}")
        return
    print(rendered)


if __name__ == "__main__":
    main()
