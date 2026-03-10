# Evaluation Pipeline

## Goal

Evaluate generated answers asynchronously (outside live request latency) with separate scores for:

- `clarity_score`
- `relevance_score`
- `hallucination_score`

and an evidence-grounding support metric:

- `evidence_similarity_score`

## End-to-End Flow

`User request`
-> `retrieval from Postgres/pgvector`
-> `answer generation`
-> `request metrics persisted to DynamoDB requests table (includes retrieval evidence snapshot)`
-> `optional SQS evaluation event enqueue (success-only request_id)`
-> `offline evaluator (queue worker by request_id OR scan worker)`
-> `3 independent LLM judge calls (clarity, relevance, hallucination)`
-> `evaluation item written to DynamoDB evaluations table by request_id`
-> `daily/adhoc report computes p50/p95 + failure reasons + worst examples`

## Data Sources

### 1) Request metrics table

Table: `unigraph-chat-metrics-requests`

Used fields:

- `request_id`
- `timestamp`
- `user_id`
- `session_id`
- `question` (fallback to `query`)
- `answer`
- `retrieval_evidence_json` (snapshot captured at request time)

### 2) Evaluations table

Table: `unigraph-chat-evaluations`

Primary key:

- `request_id` (String)

Stored output:

- `clarity_score`
- `relevance_score`
- `hallucination_score`
- `evidence_similarity_score`
- `answered_question`
- `failure_reason`
- `overall_score`
- `judge_prompt_tokens`, `judge_completion_tokens`, `judge_total_tokens`
- `notes`

## Judge Models And Prompts

Judge model config:

- `app/config/evaluation_config.yaml`
- key: `evaluation.judge_model_id`
- current default: `us.amazon.nova-2-lite-v1:0`

Judge prompts:

- file: `app/config/evaluation_prompt.yaml`
- keys:
  - `evaluation_judge.clarity_system_prompt`
  - `evaluation_judge.relevance_system_prompt`
  - `evaluation_judge.hallucination_system_prompt`

Implementation:

- `app/scripts/eval_dynamodb_worker.py`
- `app/scripts/eval_queue_worker.py`
- each metric is scored by a separate LLM judge call

## Scoring Logic

### Clarity

How readable, structured, and coherent the answer is.

### Relevance

How directly the answer addresses the user question.
Also returns `answered_question` boolean.

### Hallucination

Judged against `retrieval_evidence` snapshot:

- compare answer claims with retrieved evidence
- penalize unsupported entities/facts/numbers
- produce `evidence_similarity_score`
- produce `hallucination_score` (high means more hallucination)

### Aggregate

`overall_score = average(clarity_score, relevance_score, 1 - hallucination_score)`

Failure reason rules:

- hallucination if hallucination score is high
- incomplete if question not answered
- irrelevant if relevance is low
- unclear if clarity is low
- otherwise none

## Runtime Modes

### Manual

- `POST /api/v1/eval/offline/run?force=true`

### Guarded run

- `POST /api/v1/eval/offline/run`
- runs only when interval + new-data conditions pass

### Scheduled

- background scheduler checks every `evaluation.schedule_interval_hours` (default 24)
- executes only if new successful requests exist since last evaluated timestamp

### Queue-driven (event mode)

- set:
  - `EVALUATION_QUEUE_ENABLED=true`
  - `EVALUATION_QUEUE_URL=<sqs-url>`
- run worker:
  - `python -m app.scripts.eval_queue_worker`
- behavior:
  - successful requests publish one event keyed by `request_id`
  - worker evaluates that request immediately and stores one eval row
  - scan scheduler can remain enabled as safety net/backfill

## Reporting

Report endpoint:

- `GET /api/v1/eval/offline/report?hours=24&top_bad=10`

Report includes:

- p50/p95 for clarity/relevance/hallucination/overall
- p50/p95 for evidence similarity
- failure reason distribution
- top low-score examples
