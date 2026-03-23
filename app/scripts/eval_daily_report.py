import argparse
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
import boto3
from boto3.dynamodb.types import TypeDeserializer
from app.core.config import get_settings
from app.core.paths import resolve_project_path

settings = get_settings()
_deserializer = TypeDeserializer()
_SCAN_PROJECTION = (
    "request_id,#ts,user_id,session_id,clarity_score,relevance_score,"
    "evidence_similarity_score,hallucination_score,answered_question,"
    "failure_reason,overall_score,question,answer"
)


def _region_name() -> str | None:
    return (
        os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
        or os.getenv("AWS_SECRETS_MANAGER_REGION", "").strip()
        or None
    )


def _dynamodb_client():
    kwargs = {"region_name": _region_name()} if _region_name() else {}
    return boto3.client("dynamodb", **kwargs)


def _deserialize_item(item: dict) -> dict:
    return {key: _deserializer.deserialize(value) for key, value in item.items()}


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def _load_eval_rows(hours: int) -> list[dict]:
    table = settings.evaluation.dynamodb_table.strip()
    if not table:
        raise RuntimeError("EVALUATION_DYNAMODB_TABLE is not configured.")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    ddb = _dynamodb_client()
    rows: list[dict] = []
    last_key = None
    while True:
        kwargs = {
            "TableName": table,
            "ProjectionExpression": _SCAN_PROJECTION,
            "ExpressionAttributeNames": {"#ts": "timestamp"},
            "Limit": 200,
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        response = ddb.scan(**kwargs)
        items = response.get("Items", [])
        rows.extend(_deserialize_item(item) for item in items)
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break

    filtered: list[dict] = []
    for row in rows:
        ts = str(row.get("timestamp", ""))
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed >= cutoff:
            filtered.append(row)
    return filtered


def _build_report(rows: list[dict], top_bad: int, window_hours: int) -> dict:
    clarity = [float(row.get("clarity_score", 0.0) or 0.0) for row in rows]
    relevance = [float(row.get("relevance_score", 0.0) or 0.0) for row in rows]
    evidence_similarity = [float(row.get("evidence_similarity_score", 0.0) or 0.0) for row in rows]
    hallucination = [float(row.get("hallucination_score", 0.0) or 0.0) for row in rows]
    score = [float(row.get("overall_score", 0.0) or 0.0) for row in rows]
    reasons = Counter(str(row.get("failure_reason", "none")) for row in rows)

    worst_rows = sorted(rows, key=lambda r: float(r.get("overall_score", 0.0) or 0.0))[
        : max(0, top_bad)
    ]
    top_bad_examples = [
        {
            "request_id": row.get("request_id", ""),
            "timestamp": row.get("timestamp", ""),
            "overall_score": float(row.get("overall_score", 0.0) or 0.0),
            "failure_reason": row.get("failure_reason", "none"),
            "question": row.get("question", ""),
            "answer_preview": str(row.get("answer", ""))[:300],
        }
        for row in worst_rows
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": window_hours,
        "evaluated_count": len(rows),
        "scores": {
            "clarity_p50": round(_percentile(clarity, 50), 4),
            "clarity_p95": round(_percentile(clarity, 95), 4),
            "relevance_p50": round(_percentile(relevance, 50), 4),
            "relevance_p95": round(_percentile(relevance, 95), 4),
            "evidence_similarity_p50": round(_percentile(evidence_similarity, 50), 4),
            "evidence_similarity_p95": round(_percentile(evidence_similarity, 95), 4),
            "hallucination_p50": round(_percentile(hallucination, 50), 4),
            "hallucination_p95": round(_percentile(hallucination, 95), 4),
            "overall_p50": round(_percentile(score, 50), 4),
            "overall_p95": round(_percentile(score, 95), 4),
        },
        "failure_reasons": dict(reasons),
        "top_bad_examples": top_bad_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate daily evaluation report from DynamoDB evaluation table."
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=settings.evaluation.lookback_hours,
        help="Lookback window in hours.",
    )
    parser.add_argument(
        "--top-bad", type=int, default=10, help="Number of low-score examples to include."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reports/eval_daily_report.json",
        help="Report output path.",
    )
    args = parser.parse_args()

    rows = _load_eval_rows(hours=args.hours)
    report = _build_report(rows, top_bad=args.top_bad, window_hours=args.hours)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote daily eval report: {output_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
