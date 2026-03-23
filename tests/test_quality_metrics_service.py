from app.services.quality_metrics_service import (
    aggregate_metric_rows,
    citation_accuracy_score,
    generation_metrics,
    retrieval_metrics,
    token_precision_recall_f1,
)


def test_retrieval_metrics_hit_recall_and_mrr():
    retrieved = [
        {"chunk_id": "chunk-A"},
        {"chunk_id": "chunk-B"},
        {"chunk_id": "chunk-C"},
    ]
    relevant = ["chunk-C", "chunk-X"]

    metrics = retrieval_metrics(retrieved, relevant)

    assert metrics["hit_at_k"] == 1.0
    assert metrics["recall_at_k"] == 0.5
    assert metrics["mrr"] == 1 / 3
    assert metrics["retrieved_relevant_count"] == 1


def test_retrieval_metrics_no_relevant_ids_returns_zeroes():
    metrics = retrieval_metrics([{"chunk_id": "chunk-A"}], [])

    assert metrics["hit_at_k"] == 0.0
    assert metrics["recall_at_k"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["retrieved_relevant_count"] == 0


def test_token_precision_recall_f1_partial_overlap():
    metrics = token_precision_recall_f1(
        "Berlin has strong AI systems labs",
        "Berlin has AI labs",
    )

    assert round(metrics["precision"], 4) == 0.6667
    assert round(metrics["recall"], 4) == 1.0
    assert round(metrics["f1"], 4) == 0.8


def test_generation_metrics_with_expected_answer_and_contexts():
    retrieved = [
        {"content": "Berlin University offers AI systems labs and security research."},
        {"content": "Munich has robotics labs."},
    ]
    metrics = generation_metrics(
        query="Which city has AI systems labs?",
        answer="Berlin has AI systems labs.",
        expected_answer="Berlin has AI systems labs.",
        retrieved_results=retrieved,
    )

    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["context_coverage"] == 1.0
    assert metrics["groundedness"] == 1.0
    assert metrics["hallucination_proxy"] == 0.0
    assert metrics["query_relevance"] > 0.5


def test_citation_accuracy_score_counts_allowed_hosts():
    answer = "See https://www.ox.ac.uk/admissions and " "https://www.example.com/post for details."
    score = citation_accuracy_score(answer, ["https://www.ox.ac.uk/admissions"])
    assert score == 0.5


def test_aggregate_metric_rows_computes_per_key_means():
    summary = aggregate_metric_rows(
        [
            {"hit_at_k": 1.0, "mrr": 0.5},
            {"hit_at_k": 0.0, "mrr": 0.25},
        ]
    )

    assert summary["hit_at_k"] == 0.5
    assert summary["mrr"] == 0.375
