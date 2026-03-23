# Website Search (SerpAPI + Evidence-Grounded Answers)

## 1) Purpose
Enable live website fallback when vector retrieval is weak, while keeping answers grounded to evidence URLs.

## 2) Runtime Flow
`User question`
-> `Vector retrieval (pgvector)`
-> `Confidence gate (top similarity threshold)`
-> `SerpAPI fallback (only when needed)`
-> `Top pages fetch (async)`
-> `HTML cleanup + boilerplate stripping`
-> `Chunking + ranking + near-duplicate removal`
-> `Reranking (optional Bedrock reranker)`
-> `Citation-grounded prompt + allowed URLs`
-> `Model answer`
-> `Citation validation`
-> `Abstain if weak evidence or missing citations`

Abstain message:
`Sorry, no relevant information is found.`

## 3) Search + Retrieval Components
- SerpAPI client:
  - `app/services/serpapi_search_service.py`
  - Async single and batch query support.
- Web retrieval pipeline:
  - `app/services/web_retrieval_service.py`
  - Multi-query search variants.
  - Domain allowlist filtering (for example `.de`, `.eu`).
  - Async top-page fetch.
  - Published date extraction from SerpAPI rows and HTML metadata.
  - Clean-text chunking and relevance scoring.
  - Near-duplicate chunk filtering.

## 4) Citation-Grounded Answering
- Prompt policy:
  - `app/config/prompt.yaml`
  - Requires evidence-only answers and URL citations.
- Runtime enforcement:
  - `app/services/llm_service.py`
  - Injects citation policy + allowed URLs into system messages.
  - Blocks/abstains when:
    - no evidence,
    - no evidence URLs,
    - answer does not cite allowed URLs.

## 5) Configuration
Main config:
- `app/config/serpapi_config.yaml`

Important knobs:
- `enabled`
- `fallback_enabled`
- `fallback_similarity_threshold`
- `multi_query_enabled`
- `max_query_variants`
- `allowed_domain_suffixes`
- `max_pages_to_fetch`
- `page_fetch_timeout_seconds`
- `strip_boilerplate`
- `page_chunk_chars`
- `page_chunk_overlap_chars`
- `max_chunks_per_page`
- `chunk_dedupe_similarity`

## 6) Evaluation and Tracking
Tracked for web fallback quality:
- retrieval strategy
- source count
- groundedness
- citation accuracy
- user feedback

Where:
- Request metrics:
  - `app/services/llm_service.py`
  - `app/services/metrics_dynamodb_service.py`
- Evaluation traces and report summary:
  - `app/services/evaluation_service.py`
  - `GET /api/v1/eval/report`
- Feedback labeling:
  - `POST /api/v1/eval/conversations/{conversation_id}/label`
  - fields: `user_feedback`, `user_feedback_score` (-1/0/1)

## 7) Quick Validation
Local search fetch:
```bash
./venv/bin/python -m app.scripts.fetch_serpapi_google \
  --query "Oxford AI masters admission" \
  --query "TU Munich AI lab"
```

Focused tests:
```bash
./venv/bin/pytest -q \
  tests/test_web_retrieval_service.py \
  tests/test_llm_service.py \
  tests/test_evaluation_service.py \
  tests/test_quality_metrics_service.py
```
