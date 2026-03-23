from __future__ import annotations
import re
from collections import Counter
from statistics import mean
from urllib.parse import urlparse

_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
_URL_PATTERN = re.compile(r"https?://[^\s<>\")\]]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric word units."""
    if not isinstance(text, str):
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def exact_match_score(prediction: str, reference: str) -> float:
    """Return 1.0 when normalized strings match exactly, else 0.0."""
    pred = " ".join(_tokenize(prediction))
    ref = " ".join(_tokenize(reference))
    if not pred and not ref:
        return 1.0
    return 1.0 if pred == ref else 0.0


def token_precision_recall_f1(prediction: str, reference: str) -> dict[str, float]:
    """Compute token-level precision, recall, and F1."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts)

    precision = _safe_divide(overlap, len(pred_tokens))
    recall = _safe_divide(overlap, len(ref_tokens))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def query_relevance_score(answer: str, query: str) -> float:
    """Approximate relevance with token-overlap recall over the query."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    answer_token_set = set(_tokenize(answer))
    overlap = sum(1 for token in query_tokens if token in answer_token_set)
    return _safe_divide(overlap, len(query_tokens))


def context_coverage_score(answer: str, contexts: list[str]) -> float:
    """
    Approximate groundedness as answer-token coverage by retrieved context tokens.

    A higher value means more of the answer is represented in the retrieved context.
    """
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 1.0
    context_token_set: set[str] = set()
    for context in contexts:
        context_token_set.update(_tokenize(context))
    if not context_token_set:
        return 0.0
    supported = sum(1 for token in answer_tokens if token in context_token_set)
    return _safe_divide(supported, len(answer_tokens))


def hallucination_proxy_score(answer: str, contexts: list[str]) -> float:
    """Proxy hallucination score: 1.0 - context coverage."""
    return 1.0 - context_coverage_score(answer, contexts)


def citation_accuracy_score(answer: str, evidence_urls: list[str] | None) -> float:
    """Return ratio of cited URLs that match the allowed evidence URL hosts."""
    if not isinstance(answer, str) or not answer.strip():
        return 0.0
    allowed_hosts = {
        str(urlparse(str(url)).netloc or "").strip().lower()
        for url in (evidence_urls or [])
        if isinstance(url, str) and str(url).strip()
    }
    allowed_hosts = {host for host in allowed_hosts if host}
    if not allowed_hosts:
        return 0.0

    cited_urls = _URL_PATTERN.findall(answer)
    if not cited_urls:
        return 0.0
    cited_hosts = [
        str(urlparse(url).netloc or "").strip().lower()
        for url in cited_urls
        if isinstance(url, str)
    ]
    cited_hosts = [host for host in cited_hosts if host]
    if not cited_hosts:
        return 0.0
    matching = sum(1 for host in cited_hosts if host in allowed_hosts)
    return _safe_divide(matching, len(cited_hosts))


def _extract_result_identifier(result: dict) -> str | None:
    """Extract a stable identifier from a retrieved result payload."""
    if not isinstance(result, dict):
        return None
    for key in ("chunk_id", "document_id", "id"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def retrieval_metrics(
    retrieved_results: list[dict],
    relevant_ids: list[str],
) -> dict[str, float | int]:
    """
    Compute basic retrieval ranking metrics.

    - hit_at_k: at least one relevant item is present in the top-k results
    - recall_at_k: fraction of relevant ids recovered in top-k
    - mrr: reciprocal rank of first relevant hit
    """
    ranked_ids: list[str] = []
    for result in retrieved_results:
        identifier = _extract_result_identifier(result)
        if identifier:
            ranked_ids.append(identifier)

    relevant = {item.strip() for item in relevant_ids if isinstance(item, str) and item.strip()}
    if not relevant:
        return {
            "hit_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "retrieved_relevant_count": 0,
        }

    retrieved_relevant = [identifier for identifier in ranked_ids if identifier in relevant]
    hit_at_k = 1.0 if retrieved_relevant else 0.0
    recall_at_k = _safe_divide(len(set(retrieved_relevant)), len(relevant))

    reciprocal_rank = 0.0
    for index, identifier in enumerate(ranked_ids, start=1):
        if identifier in relevant:
            reciprocal_rank = 1.0 / index
            break

    return {
        "hit_at_k": hit_at_k,
        "recall_at_k": recall_at_k,
        "mrr": reciprocal_rank,
        "retrieved_relevant_count": len(set(retrieved_relevant)),
    }


def generation_metrics(
    *,
    query: str,
    answer: str,
    expected_answer: str | None = None,
    retrieved_results: list[dict] | None = None,
) -> dict[str, float]:
    """Compute generation quality metrics from answer text and optional references."""
    metrics: dict[str, float] = {
        "query_relevance": query_relevance_score(answer, query),
    }

    contexts = []
    for result in retrieved_results or []:
        if not isinstance(result, dict):
            continue
        content = result.get("content")
        if isinstance(content, str) and content.strip():
            contexts.append(content.strip())

    metrics["context_coverage"] = context_coverage_score(answer, contexts)
    metrics["groundedness"] = metrics["context_coverage"]
    metrics["hallucination_proxy"] = hallucination_proxy_score(answer, contexts)

    if expected_answer is not None:
        prf = token_precision_recall_f1(answer, expected_answer)
        metrics["exact_match"] = exact_match_score(answer, expected_answer)
        metrics["token_precision"] = prf["precision"]
        metrics["token_recall"] = prf["recall"]
        metrics["token_f1"] = prf["f1"]
    return metrics


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric key across all provided rows."""
    if not rows:
        return {}
    summary: dict[str, float] = {}
    keys = sorted({key for row in rows for key in row.keys()})
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[key] = mean(values)
    return summary
