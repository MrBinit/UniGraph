import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
import boto3
from boto3.dynamodb.types import TypeDeserializer
from app.core.config import get_evaluation_prompts, get_settings
from app.infra.bedrock_chat_client import client

settings = get_settings()
evaluation_prompts = get_evaluation_prompts()
_deserializer = TypeDeserializer()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def _from_dynamo_item(item: dict) -> dict:
    payload = {}
    for key, value in item.items():
        payload[key] = _deserializer.deserialize(value)
    return payload


def _request_status_attr() -> str:
    return settings.evaluation.request_status_attribute.strip() or "eval_status"


def _request_status_index() -> str:
    return settings.evaluation.request_status_index_name.strip()


def _request_pending_value() -> str:
    return settings.evaluation.request_pending_value.strip() or "pending"


def _request_in_progress_value() -> str:
    return settings.evaluation.request_in_progress_value.strip() or "in_progress"


def _request_completed_value() -> str:
    return settings.evaluation.request_completed_value.strip() or "completed"


def _eval_status_attr() -> str:
    return settings.evaluation.eval_status_attribute.strip() or "status"


def _eval_completed_value() -> str:
    return settings.evaluation.eval_completed_value.strip() or "completed"


def _safe_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _ttl_epoch(ttl_days: int) -> int:
    if ttl_days <= 0:
        return 0
    return int((_utc_now() + timedelta(days=ttl_days)).timestamp())


def _extract_json_block(text: str) -> dict:
    if not isinstance(text, str):
        return {}
    text = text.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _judge_prompt(system_prompt: str, payload: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def _normalize_score(value, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _normalize_reason(value: str) -> str:
    normalized = str(value or "").strip().lower()
    allowed = {"none", "unclear", "irrelevant", "hallucination", "incomplete"}
    return normalized if normalized in allowed else "none"


async def _evaluate_one(record: dict) -> dict:
    question = str(record.get("question", ""))
    answer = str(record.get("answer", ""))
    evidence = record.get("retrieval_evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    prompts = evaluation_prompts.get("evaluation_judge", {})
    clarity_prompt = str(prompts.get("clarity_system_prompt", "")).strip()
    relevance_prompt = str(prompts.get("relevance_system_prompt", "")).strip()
    hallucination_prompt = str(prompts.get("hallucination_system_prompt", "")).strip()
    if not clarity_prompt:
        clarity_prompt = (
            "You are a strict evaluator for answer clarity. "
            "Return only JSON with keys: clarity_score, notes."
        )
    if not relevance_prompt:
        relevance_prompt = (
            "You are a strict evaluator for answer relevance. "
            "Return only JSON with keys: relevance_score, answered_question, notes."
        )
    if not hallucination_prompt:
        hallucination_prompt = (
            "You are a strict evaluator for hallucination risk using retrieval evidence. "
            "Return only JSON with keys: evidence_similarity_score, hallucination_score, notes."
        )

    async def _judge_once(system_prompt: str, payload: dict) -> tuple[dict, dict]:
        response = await client.chat.completions.create(
            model=settings.evaluation.judge_model_id,
            messages=_judge_prompt(system_prompt, payload),
            timeout=settings.bedrock.timeout,
        )
        content = response.choices[0].message.content if response.choices else ""
        judged = _extract_json_block(content)
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        return judged, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    base_payload = {
        "question": question,
        "answer": answer,
    }
    clarity_raw, clarity_usage = await _judge_once(clarity_prompt, base_payload)
    relevance_raw, relevance_usage = await _judge_once(relevance_prompt, base_payload)
    hallucination_payload = {
        "question": question,
        "answer": answer,
        "retrieval_evidence": evidence,
    }
    hallucination_raw, hallucination_usage = await _judge_once(
        hallucination_prompt, hallucination_payload
    )

    clarity = _normalize_score(
        clarity_raw.get("clarity_score", clarity_raw.get("score")),
        0.0,
    )
    relevance = _normalize_score(
        relevance_raw.get("relevance_score", relevance_raw.get("score")),
        0.0,
    )
    evidence_similarity = _normalize_score(
        hallucination_raw.get(
            "evidence_similarity_score",
            hallucination_raw.get("score"),
        ),
        0.0,
    )
    hallucination = _normalize_score(hallucination_raw.get("hallucination_score"), 1.0)
    if "hallucination_score" not in hallucination_raw:
        hallucination = _normalize_score(1.0 - evidence_similarity, 1.0)
    answered_question = bool(relevance_raw.get("answered_question", False))
    notes = " | ".join(
        note
        for note in [
            str(clarity_raw.get("notes", "")).strip(),
            str(relevance_raw.get("notes", "")).strip(),
            str(hallucination_raw.get("notes", "")).strip(),
        ]
        if note
    )[:600]

    # Deterministic failure reason from independent judges.
    if hallucination >= 0.6:
        failure_reason = "hallucination"
    elif not answered_question:
        failure_reason = "incomplete"
    elif relevance < 0.5:
        failure_reason = "irrelevant"
    elif clarity < 0.5:
        failure_reason = "unclear"
    else:
        failure_reason = "none"

    prompt_tokens = (
        clarity_usage["prompt_tokens"]
        + relevance_usage["prompt_tokens"]
        + hallucination_usage["prompt_tokens"]
    )
    completion_tokens = (
        clarity_usage["completion_tokens"]
        + relevance_usage["completion_tokens"]
        + hallucination_usage["completion_tokens"]
    )
    total_tokens = (
        clarity_usage["total_tokens"]
        + relevance_usage["total_tokens"]
        + hallucination_usage["total_tokens"]
    )
    return {
        "clarity_score": clarity,
        "relevance_score": relevance,
        "hallucination_score": hallucination,
        "evidence_similarity_score": evidence_similarity,
        "answered_question": answered_question,
        "failure_reason": _normalize_reason(failure_reason),
        "notes": notes,
        "judge_prompt_tokens": prompt_tokens,
        "judge_completion_tokens": completion_tokens,
        "judge_total_tokens": total_tokens,
    }


def _extract_retrieval_evidence(item: dict) -> list[dict]:
    raw = item.get("retrieval_evidence_json", [])
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]

    raw_record = item.get("record_json")
    if isinstance(raw_record, str):
        try:
            payload = json.loads(raw_record)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            retrieval = payload.get("retrieval", {})
            if isinstance(retrieval, dict):
                evidence = retrieval.get("evidence", [])
                if isinstance(evidence, list):
                    return [row for row in evidence if isinstance(row, dict)]
    return []


def _normalize_request_for_eval(item: dict) -> dict | None:
    if str(item.get("outcome", "")) != "success":
        return None
    if not str(item.get("question", "")).strip():
        item["question"] = str(item.get("query", "")).strip()
    item["retrieval_evidence"] = _extract_retrieval_evidence(item)
    return item


def _load_request_for_eval(request_id: str) -> dict | None:
    requests_table = settings.app.metrics_dynamodb_requests_table.strip()
    request_id = str(request_id).strip()
    if not requests_table or not request_id:
        return None

    ddb = _dynamodb_client()
    response = ddb.get_item(
        TableName=requests_table,
        Key={"request_id": {"S": request_id}},
    )
    raw = response.get("Item")
    if not isinstance(raw, dict):
        return None
    item = _from_dynamo_item(raw)
    return _normalize_request_for_eval(item)


def _load_requests_for_eval(max_items: int, lookback_hours: int) -> list[dict]:
    requests_table = settings.app.metrics_dynamodb_requests_table.strip()
    if not requests_table:
        raise RuntimeError("APP_METRICS_DYNAMODB_REQUESTS_TABLE is not configured.")

    index_name = _request_status_index()
    if not index_name:
        raise RuntimeError("EVALUATION_REQUEST_STATUS_INDEX_NAME is not configured.")

    cutoff_iso = _safe_iso(_utc_now() - timedelta(hours=lookback_hours))
    status_attr = _request_status_attr()
    pending_value = _request_pending_value()
    ddb = _dynamodb_client()
    items: list[dict] = []
    last_key: dict | None = None
    while len(items) < max_items:
        kwargs = {
            "TableName": requests_table,
            "IndexName": index_name,
            "KeyConditionExpression": "#status = :status_value AND #ts >= :cutoff_iso",
            "ExpressionAttributeNames": {
                "#status": status_attr,
                "#ts": "timestamp",
            },
            "ExpressionAttributeValues": {
                ":status_value": {"S": pending_value},
                ":cutoff_iso": {"S": cutoff_iso},
            },
            "ProjectionExpression": (
                "request_id,#ts,user_id,session_id,outcome,question,query,answer,"
                "retrieval_evidence_json,record_json,#status"
            ),
            "Limit": min(100, max_items - len(items)),
            "ScanIndexForward": False,
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        response = ddb.query(**kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break

    normalized_items = [_from_dynamo_item(item) for item in items]
    filtered: list[dict] = []
    for item in normalized_items:
        normalized = _normalize_request_for_eval(item)
        if normalized:
            filtered.append(normalized)
    return filtered


def _claim_request_for_eval(request_id: str) -> bool:
    requests_table = settings.app.metrics_dynamodb_requests_table.strip()
    if not requests_table or not request_id:
        return False

    status_attr = _request_status_attr()
    status_updated_attr = f"{status_attr}_updated_at"
    ddb = _dynamodb_client()
    try:
        ddb.update_item(
            TableName=requests_table,
            Key={"request_id": {"S": request_id}},
            UpdateExpression="SET #status = :in_progress, #status_updated = :now_iso",
            ConditionExpression=(
                "attribute_exists(request_id) "
                "AND #outcome = :success "
                "AND (attribute_not_exists(#status) OR #status = :pending)"
            ),
            ExpressionAttributeNames={
                "#outcome": "outcome",
                "#status": status_attr,
                "#status_updated": status_updated_attr,
            },
            ExpressionAttributeValues={
                ":success": {"S": "success"},
                ":pending": {"S": _request_pending_value()},
                ":in_progress": {"S": _request_in_progress_value()},
                ":now_iso": {"S": _safe_iso(_utc_now())},
            },
        )
        return True
    except ddb.exceptions.ConditionalCheckFailedException:
        return False


def _mark_request_eval_status(request_id: str, status: str) -> None:
    requests_table = settings.app.metrics_dynamodb_requests_table.strip()
    if not requests_table or not request_id:
        return
    status_attr = _request_status_attr()
    status_updated_attr = f"{status_attr}_updated_at"
    _dynamodb_client().update_item(
        TableName=requests_table,
        Key={"request_id": {"S": request_id}},
        UpdateExpression="SET #status = :status, #status_updated = :now_iso",
        ExpressionAttributeNames={
            "#status": status_attr,
            "#status_updated": status_updated_attr,
        },
        ExpressionAttributeValues={
            ":status": {"S": str(status)},
            ":now_iso": {"S": _safe_iso(_utc_now())},
        },
    )


def _persist_eval(request_record: dict, judged: dict) -> bool:
    table = settings.evaluation.dynamodb_table.strip()
    request_id = str(request_record.get("request_id", "")).strip()
    if not request_id:
        return False
    expires_at = _ttl_epoch(settings.evaluation.ttl_days)
    score = round(
        (
            float(judged["clarity_score"])
            + float(judged["relevance_score"])
            + (1.0 - float(judged["hallucination_score"]))
        )
        / 3.0,
        4,
    )
    item = {
        "request_id": {"S": request_id},
        "timestamp": {"S": str(request_record.get("timestamp", _safe_iso(_utc_now())))},
        "user_id": {"S": str(request_record.get("user_id", ""))},
        "session_id": {"S": str(request_record.get("session_id", ""))},
        "judge_model_id": {"S": settings.evaluation.judge_model_id},
        "clarity_score": {"N": str(judged["clarity_score"])},
        "relevance_score": {"N": str(judged["relevance_score"])},
        "hallucination_score": {"N": str(judged["hallucination_score"])},
        "evidence_similarity_score": {"N": str(judged["evidence_similarity_score"])},
        "answered_question": {"BOOL": bool(judged["answered_question"])},
        "failure_reason": {"S": str(judged["failure_reason"])},
        "overall_score": {"N": str(score)},
        "judge_prompt_tokens": {"N": str(int(judged["judge_prompt_tokens"]))},
        "judge_completion_tokens": {"N": str(int(judged["judge_completion_tokens"]))},
        "judge_total_tokens": {"N": str(int(judged["judge_total_tokens"]))},
        "notes": {"S": str(judged["notes"])},
        "question": {"S": str(request_record.get("question", ""))},
        "answer": {"S": str(request_record.get("answer", ""))},
        _eval_status_attr(): {"S": _eval_completed_value()},
    }
    if expires_at > 0:
        item["expires_at"] = {"N": str(expires_at)}
    ddb = _dynamodb_client()
    try:
        ddb.put_item(
            TableName=table,
            Item=item,
            ConditionExpression="attribute_not_exists(request_id)",
        )
        return True
    except ddb.exceptions.ConditionalCheckFailedException:
        return False


async def run_request_eval(request_id: str) -> dict:
    """Evaluate exactly one successful request and persist scores."""
    request_id = str(request_id).strip()
    if not settings.evaluation.enabled:
        return {
            "request_id": request_id,
            "evaluated": False,
            "skipped": True,
            "reason": "evaluation disabled",
        }
    if not request_id:
        return {
            "request_id": request_id,
            "evaluated": False,
            "skipped": True,
            "reason": "missing request_id",
        }

    if not _claim_request_for_eval(request_id):
        return {
            "request_id": request_id,
            "evaluated": False,
            "skipped": True,
            "reason": "already evaluated or in progress",
        }

    request = _load_request_for_eval(request_id)
    if not request:
        _mark_request_eval_status(request_id, _request_pending_value())
        return {
            "request_id": request_id,
            "evaluated": False,
            "skipped": True,
            "reason": "request not found or not successful",
        }

    try:
        judged = await _evaluate_one(request)
        inserted = _persist_eval(request, judged)
        _mark_request_eval_status(request_id, _request_completed_value())
        if not inserted:
            return {
                "request_id": request_id,
                "evaluated": False,
                "skipped": True,
                "reason": "already evaluated",
            }
        return {
            "request_id": request_id,
            "evaluated": True,
            "skipped": False,
            "reason": "ok",
        }
    except Exception:
        _mark_request_eval_status(request_id, _request_pending_value())
        raise


async def run(limit: int | None = None) -> dict:
    if not settings.evaluation.enabled:
        return {"evaluated": 0, "skipped": 0, "reason": "evaluation disabled"}

    max_items = min(
        limit or settings.evaluation.max_items_per_run, settings.evaluation.max_items_per_run
    )
    requests = _load_requests_for_eval(
        max_items=max_items, lookback_hours=settings.evaluation.lookback_hours
    )
    evaluated = 0
    skipped = 0
    for request in requests:
        request_id = str(request.get("request_id", "")).strip()
        if not request_id:
            skipped += 1
            continue
        if not _claim_request_for_eval(request_id):
            skipped += 1
            continue
        try:
            judged = await _evaluate_one(request)
            inserted = _persist_eval(request, judged)
            _mark_request_eval_status(request_id, _request_completed_value())
            if inserted:
                evaluated += 1
            else:
                skipped += 1
        except Exception:
            _mark_request_eval_status(request_id, _request_pending_value())
            raise
    return {"evaluated": evaluated, "skipped": skipped, "scanned": len(requests)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate successful chat requests and write scores to DynamoDB."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max items to evaluate in this run."
    )
    args = parser.parse_args()
    result = asyncio.run(run(limit=args.limit))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
