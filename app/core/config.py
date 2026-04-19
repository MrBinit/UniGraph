import json
import logging
import os
import yaml
from functools import lru_cache
from pathlib import Path
import boto3

from app.core.paths import APP_CONFIG_DIR
from app.schemas.settings_schema import Settings

logger = logging.getLogger(__name__)


def _secrets_region() -> str | None:
    """Resolve AWS region used for Secrets Manager access."""
    region = (
        os.getenv("AWS_SECRETS_MANAGER_REGION", "").strip()
        or os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
    )
    return region or None


def _load_secret_string(secret_id: str) -> str:
    """Fetch one Secrets Manager SecretString payload."""
    region = _secrets_region()
    client_kwargs = {"region_name": region} if region else {}
    try:
        client = boto3.client("secretsmanager", **client_kwargs)
        response = client.get_secret_value(SecretId=secret_id)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load AWS Secrets Manager secret '{secret_id}'. {exc}"
        ) from exc

    secret_string = response.get("SecretString", "")
    if not isinstance(secret_string, str) or not secret_string.strip():
        raise RuntimeError(f"AWS secret '{secret_id}' is empty or not a SecretString payload.")
    return secret_string


def _parse_secret_payload(secret_id: str, secret_string: str) -> dict:
    """Parse one Secrets Manager JSON payload into a mapping."""
    try:
        payload = json.loads(secret_string)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"AWS secret '{secret_id}' is not valid JSON. {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"AWS secret '{secret_id}' must be a JSON object.")
    return payload


def _apply_secret_payload(payload: dict) -> int:
    """Load missing env variables from one parsed secret payload."""
    loaded_keys = 0
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip() or value is None:
            continue
        key = key.strip()
        existing = os.getenv(key)
        if existing is not None and existing != "":
            continue
        if isinstance(value, (str, int, float, bool)):
            os.environ[key] = str(value)
            loaded_keys += 1
    return loaded_keys


def _load_yaml_file(file_path: Path) -> dict:
    """Load one YAML file and enforce a top-level mapping structure."""
    with open(file_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at top-level: {file_path}")
    return data


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay dict into base dict."""
    for key, value in overlay.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_aws_secrets_manager_env():
    """Load environment variables from one AWS Secrets Manager JSON secret."""
    secret_id = os.getenv("AWS_SECRETS_MANAGER_SECRET_ID", "").strip()
    if not secret_id:
        return

    secret_string = _load_secret_string(secret_id)
    payload = _parse_secret_payload(secret_id, secret_string)
    loaded_keys = _apply_secret_payload(payload)

    logger.info("AWSSecretLoaded | secret_id=%s | loaded_keys=%s", secret_id, loaded_keys)


def _apply_env_overrides(data: dict) -> dict:
    """Apply environment overrides on top of YAML configuration for deployments."""
    config = data
    yaml_only_prefixes = ("WEB_SEARCH_", "BEDROCK_", "AZURE_OPENAI_")

    def _set(path: list[str], env_name: str, cast=str):
        if any(str(env_name).startswith(prefix) for prefix in yaml_only_prefixes):
            # Keep model/search behavior controlled by YAML config, not env flags.
            return
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            return

        if cast is bool:
            value = raw.strip().lower() in {"1", "true", "yes", "on"}
        elif cast is int:
            value = int(raw)
        else:
            value = cast(raw)

        node = config
        for key in path[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[path[-1]] = value

    _set(["app", "log_level"], "APP_LOG_LEVEL")
    _set(["app", "docs_enabled"], "APP_DOCS_ENABLED", bool)
    _set(["app", "metrics_json_enabled"], "APP_METRICS_JSON_ENABLED", bool)
    _set(["app", "metrics_json_dir"], "APP_METRICS_JSON_DIR")
    _set(["app", "metrics_dynamodb_enabled"], "APP_METRICS_DYNAMODB_ENABLED", bool)
    _set(["app", "metrics_dynamodb_requests_table"], "APP_METRICS_DYNAMODB_REQUESTS_TABLE")
    _set(["app", "metrics_dynamodb_aggregate_table"], "APP_METRICS_DYNAMODB_AGGREGATE_TABLE")
    _set(["app", "metrics_dynamodb_ttl_days"], "APP_METRICS_DYNAMODB_TTL_DAYS", int)
    _set(["evaluation", "enabled"], "EVALUATION_ENABLED", bool)
    _set(["evaluation", "dynamodb_table"], "EVALUATION_DYNAMODB_TABLE")
    _set(["evaluation", "judge_model_id"], "EVALUATION_JUDGE_MODEL_ID")
    _set(["evaluation", "batch_size"], "EVALUATION_BATCH_SIZE", int)
    _set(["evaluation", "max_items_per_run"], "EVALUATION_MAX_ITEMS_PER_RUN", int)
    _set(["evaluation", "lookback_hours"], "EVALUATION_LOOKBACK_HOURS", int)
    _set(["evaluation", "ttl_days"], "EVALUATION_TTL_DAYS", int)
    _set(["evaluation", "schedule_enabled"], "EVALUATION_SCHEDULE_ENABLED", bool)
    _set(["evaluation", "schedule_interval_hours"], "EVALUATION_SCHEDULE_INTERVAL_HOURS", int)
    _set(["evaluation", "request_status_attribute"], "EVALUATION_REQUEST_STATUS_ATTRIBUTE")
    _set(["evaluation", "request_status_index_name"], "EVALUATION_REQUEST_STATUS_INDEX_NAME")
    _set(["evaluation", "request_pending_value"], "EVALUATION_REQUEST_PENDING_VALUE")
    _set(["evaluation", "request_in_progress_value"], "EVALUATION_REQUEST_IN_PROGRESS_VALUE")
    _set(["evaluation", "request_completed_value"], "EVALUATION_REQUEST_COMPLETED_VALUE")
    _set(["evaluation", "request_not_applicable_value"], "EVALUATION_REQUEST_NOT_APPLICABLE_VALUE")
    _set(["evaluation", "eval_status_attribute"], "EVALUATION_EVAL_STATUS_ATTRIBUTE")
    _set(["evaluation", "eval_status_index_name"], "EVALUATION_EVAL_STATUS_INDEX_NAME")
    _set(["evaluation", "eval_completed_value"], "EVALUATION_EVAL_COMPLETED_VALUE")

    _set(["security", "auth_enabled"], "SECURITY_AUTH_ENABLED", bool)
    _set(["security", "jwt_secret"], "SECURITY_JWT_SECRET")
    _set(["security", "jwt_issuer"], "SECURITY_JWT_ISSUER")
    _set(["security", "jwt_audience"], "SECURITY_JWT_AUDIENCE")
    _set(["security", "jwt_exp_minutes"], "SECURITY_JWT_EXP_MINUTES", int)

    _set(["web_search", "enabled"], "WEB_SEARCH_ENABLED", bool)
    _set(["web_search", "google_search_url"], "WEB_SEARCH_GOOGLE_SEARCH_URL")
    _set(["web_search", "engine"], "WEB_SEARCH_ENGINE")
    _set(["web_search", "search_depth"], "WEB_SEARCH_SEARCH_DEPTH")
    _set(["web_search", "api_key_env_name"], "WEB_SEARCH_API_KEY_ENV_NAME")
    _set(["web_search", "api_key_env_name"], "TAVILY_API_KEY_ENV_NAME")
    _set(["web_search", "default_gl"], "WEB_SEARCH_DEFAULT_GL")
    _set(["web_search", "default_hl"], "WEB_SEARCH_DEFAULT_HL")
    _set(["web_search", "default_num"], "WEB_SEARCH_DEFAULT_NUM", int)
    _set(["web_search", "timeout_seconds"], "WEB_SEARCH_TIMEOUT_SECONDS", float)
    _set(["web_search", "max_concurrency"], "WEB_SEARCH_MAX_CONCURRENCY", int)
    _set(["web_search", "queue_workers"], "WEB_SEARCH_QUEUE_WORKERS", int)
    _set(["web_search", "queue_max_size"], "WEB_SEARCH_QUEUE_MAX_SIZE", int)
    _set(
        ["web_search", "always_web_retrieval_enabled"],
        "WEB_SEARCH_ALWAYS_WEB_RETRIEVAL_ENABLED",
        bool,
    )
    _set(
        ["web_search", "retrieval_fanout_enabled"],
        "WEB_SEARCH_RETRIEVAL_FANOUT_ENABLED",
        bool,
    )
    _set(["web_search", "fallback_enabled"], "WEB_SEARCH_FALLBACK_ENABLED", bool)
    _set(
        ["web_search", "fallback_similarity_threshold"],
        "WEB_SEARCH_FALLBACK_SIMILARITY_THRESHOLD",
        float,
    )
    _set(
        ["web_search", "expansion_similarity_threshold"],
        "WEB_SEARCH_EXPANSION_SIMILARITY_THRESHOLD",
        float,
    )
    _set(["web_search", "query_planner_enabled"], "WEB_SEARCH_QUERY_PLANNER_ENABLED", bool)
    _set(["web_search", "query_planner_use_llm"], "WEB_SEARCH_QUERY_PLANNER_USE_LLM", bool)
    _set(["web_search", "query_planner_model_id"], "WEB_SEARCH_QUERY_PLANNER_MODEL_ID")
    _set(
        ["web_search", "query_planner_acquire_timeout_seconds"],
        "WEB_SEARCH_QUERY_PLANNER_ACQUIRE_TIMEOUT_SECONDS",
        float,
    )
    _set(["web_search", "query_planner_cache_enabled"], "WEB_SEARCH_QUERY_PLANNER_CACHE_ENABLED", bool)
    _set(
        ["web_search", "query_planner_cache_ttl_seconds"],
        "WEB_SEARCH_QUERY_PLANNER_CACHE_TTL_SECONDS",
        int,
    )
    _set(["web_search", "query_planner_max_queries"], "WEB_SEARCH_QUERY_PLANNER_MAX_QUERIES", int)
    _set(
        ["web_search", "query_planner_max_subquestions"],
        "WEB_SEARCH_QUERY_PLANNER_MAX_SUBQUESTIONS",
        int,
    )
    _set(["web_search", "retrieval_loop_enabled"], "WEB_SEARCH_RETRIEVAL_LOOP_ENABLED", bool)
    _set(["web_search", "retrieval_loop_use_llm"], "WEB_SEARCH_RETRIEVAL_LOOP_USE_LLM", bool)
    _set(["web_search", "retrieval_loop_model_id"], "WEB_SEARCH_RETRIEVAL_LOOP_MODEL_ID")
    _set(
        ["web_search", "retrieval_loop_acquire_timeout_seconds"],
        "WEB_SEARCH_RETRIEVAL_LOOP_ACQUIRE_TIMEOUT_SECONDS",
        float,
    )
    _set(["web_search", "retrieval_loop_cache_enabled"], "WEB_SEARCH_RETRIEVAL_LOOP_CACHE_ENABLED", bool)
    _set(
        ["web_search", "retrieval_loop_cache_ttl_seconds"],
        "WEB_SEARCH_RETRIEVAL_LOOP_CACHE_TTL_SECONDS",
        int,
    )
    _set(["web_search", "retrieval_loop_max_steps"], "WEB_SEARCH_RETRIEVAL_LOOP_MAX_STEPS", int)
    _set(
        ["web_search", "retrieval_loop_max_gap_queries"],
        "WEB_SEARCH_RETRIEVAL_LOOP_MAX_GAP_QUERIES",
        int,
    )
    _set(
        ["web_search", "agentic_required_field_rescue_max_rounds"],
        "WEB_SEARCH_AGENTIC_REQUIRED_FIELD_RESCUE_MAX_ROUNDS",
        int,
    )
    _set(
        ["web_search", "retrieval_min_unique_domains"],
        "WEB_SEARCH_RETRIEVAL_MIN_UNIQUE_DOMAINS",
        int,
    )
    _set(
        ["web_search", "deep_min_unique_domains"],
        "WEB_SEARCH_DEEP_MIN_UNIQUE_DOMAINS",
        int,
    )
    _set(
        ["web_search", "retrieval_gap_min_token_coverage"],
        "WEB_SEARCH_RETRIEVAL_GAP_MIN_TOKEN_COVERAGE",
        float,
    )
    _set(["web_search", "multi_query_enabled"], "WEB_SEARCH_MULTI_QUERY_ENABLED", bool)
    _set(["web_search", "max_query_variants"], "WEB_SEARCH_MAX_QUERY_VARIANTS", int)
    _set(
        ["web_search", "allowed_domain_suffixes"],
        "WEB_SEARCH_ALLOWED_DOMAIN_SUFFIXES",
        lambda raw: [entry.strip() for entry in raw.split(",") if entry.strip()],
    )
    _set(
        ["web_search", "official_source_filter_enabled"],
        "WEB_SEARCH_OFFICIAL_SOURCE_FILTER_ENABLED",
        bool,
    )
    _set(
        ["web_search", "official_source_allowlist"],
        "WEB_SEARCH_OFFICIAL_SOURCE_ALLOWLIST",
        lambda raw: [entry.strip().lower() for entry in raw.split(",") if entry.strip()],
    )
    _set(["web_search", "max_context_results"], "WEB_SEARCH_MAX_CONTEXT_RESULTS", int)
    _set(["web_search", "fetch_page_content"], "WEB_SEARCH_FETCH_PAGE_CONTENT", bool)
    _set(["web_search", "max_pages_to_fetch"], "WEB_SEARCH_MAX_PAGES_TO_FETCH", int)
    _set(
        ["web_search", "page_fetch_timeout_seconds"],
        "WEB_SEARCH_PAGE_FETCH_TIMEOUT_SECONDS",
        float,
    )
    _set(["web_search", "max_page_chars"], "WEB_SEARCH_MAX_PAGE_CHARS", int)
    _set(["web_search", "strip_boilerplate"], "WEB_SEARCH_STRIP_BOILERPLATE", bool)
    _set(["web_search", "min_clean_line_chars"], "WEB_SEARCH_MIN_CLEAN_LINE_CHARS", int)
    _set(["web_search", "page_chunk_chars"], "WEB_SEARCH_PAGE_CHUNK_CHARS", int)
    _set(
        ["web_search", "page_chunk_overlap_chars"],
        "WEB_SEARCH_PAGE_CHUNK_OVERLAP_CHARS",
        int,
    )
    _set(["web_search", "max_chunks_per_page"], "WEB_SEARCH_MAX_CHUNKS_PER_PAGE", int)
    _set(["web_search", "min_chunk_chars"], "WEB_SEARCH_MIN_CHUNK_CHARS", int)
    _set(
        ["web_search", "chunk_dedupe_similarity"],
        "WEB_SEARCH_CHUNK_DEDUPE_SIMILARITY",
        float,
    )
    _set(["web_search", "trust_relevance_weight"], "WEB_SEARCH_TRUST_RELEVANCE_WEIGHT", float)
    _set(["web_search", "trust_authority_weight"], "WEB_SEARCH_TRUST_AUTHORITY_WEIGHT", float)
    _set(["web_search", "trust_recency_weight"], "WEB_SEARCH_TRUST_RECENCY_WEIGHT", float)
    _set(["web_search", "trust_agreement_weight"], "WEB_SEARCH_TRUST_AGREEMENT_WEIGHT", float)
    _set(["web_search", "retry_max_attempts"], "WEB_SEARCH_RETRY_MAX_ATTEMPTS", int)
    _set(
        ["web_search", "retry_base_backoff_seconds"],
        "WEB_SEARCH_RETRY_BASE_BACKOFF_SECONDS",
        float,
    )

    # Backward-compatible aliases: old AZURE_OPENAI_* vars still map into bedrock config.
    _set(["bedrock", "primary_model_id"], "AZURE_OPENAI_PRIMARY_DEPLOYMENT")
    _set(["bedrock", "fallback_model_id"], "AZURE_OPENAI_FALLBACK_DEPLOYMENT")
    _set(["bedrock", "timeout"], "AZURE_OPENAI_TIMEOUT", int)
    _set(["bedrock", "max_concurrency"], "AZURE_OPENAI_MAX_CONCURRENCY", int)
    _set(["bedrock", "primary_model_id"], "BEDROCK_PRIMARY_MODEL_ID")
    _set(["bedrock", "fallback_model_id"], "BEDROCK_FALLBACK_MODEL_ID")
    _set(["bedrock", "timeout"], "BEDROCK_TIMEOUT", int)
    _set(["bedrock", "max_concurrency"], "BEDROCK_MAX_CONCURRENCY", int)
    _set(["bedrock", "web_grounding_enabled"], "BEDROCK_WEB_GROUNDING_ENABLED", bool)
    _set(
        ["bedrock", "web_grounding_include_sources"],
        "BEDROCK_WEB_GROUNDING_INCLUDE_SOURCES",
        bool,
    )
    _set(
        ["bedrock", "throttle_retry_max_attempts"],
        "BEDROCK_THROTTLE_RETRY_MAX_ATTEMPTS",
        int,
    )
    _set(
        ["bedrock", "throttle_retry_base_backoff_seconds"],
        "BEDROCK_THROTTLE_RETRY_BASE_BACKOFF_SECONDS",
        float,
    )
    _set(
        ["bedrock", "throttle_retry_max_backoff_seconds"],
        "BEDROCK_THROTTLE_RETRY_MAX_BACKOFF_SECONDS",
        float,
    )
    _set(
        ["bedrock", "throttle_retry_jitter_seconds"],
        "BEDROCK_THROTTLE_RETRY_JITTER_SECONDS",
        float,
    )
    _set(["bedrock", "answer_rate_limit_rps"], "BEDROCK_ANSWER_RATE_LIMIT_RPS", float)
    _set(["bedrock", "answer_rate_limit_burst"], "BEDROCK_ANSWER_RATE_LIMIT_BURST", int)
    _set(["bedrock", "planner_rate_limit_rps"], "BEDROCK_PLANNER_RATE_LIMIT_RPS", float)
    _set(["bedrock", "planner_rate_limit_burst"], "BEDROCK_PLANNER_RATE_LIMIT_BURST", int)
    _set(["bedrock", "reranker_enabled"], "BEDROCK_RERANKER_ENABLED", bool)
    _set(["bedrock", "reranker_model_id"], "BEDROCK_RERANKER_MODEL_ID")
    _set(["bedrock", "reranker_api_version"], "BEDROCK_RERANKER_API_VERSION", int)
    _set(["bedrock", "reranker_top_n"], "BEDROCK_RERANKER_TOP_N", int)
    _set(["bedrock", "reranker_min_documents"], "BEDROCK_RERANKER_MIN_DOCUMENTS", int)
    _set(["bedrock", "reranker_max_documents"], "BEDROCK_RERANKER_MAX_DOCUMENTS", int)
    _set(["bedrock", "reranker_max_query_chars"], "BEDROCK_RERANKER_MAX_QUERY_CHARS", int)
    _set(
        ["bedrock", "reranker_max_document_chars"],
        "BEDROCK_RERANKER_MAX_DOCUMENT_CHARS",
        int,
    )

    _set(["redis", "local_tunnel_enabled"], "REDIS_LOCAL_TUNNEL_ENABLED", bool)
    _set(["redis", "tunnel_local_port"], "REDIS_TUNNEL_LOCAL_PORT", int)
    _set(["redis", "tunnel_instance_id"], "REDIS_TUNNEL_INSTANCE_ID")
    _set(["redis", "app", "host"], "REDIS_APP_HOST")
    _set(["redis", "app", "port"], "REDIS_APP_PORT", int)
    _set(["redis", "app", "db"], "REDIS_APP_DB", int)
    _set(["redis", "app", "username"], "REDIS_APP_USERNAME")
    _set(["redis", "app", "password"], "REDIS_APP_PASSWORD")
    _set(["redis", "app", "tls"], "REDIS_APP_TLS", bool)
    _set(["redis", "app", "ssl_cert_reqs"], "REDIS_APP_SSL_CERT_REQS")
    _set(["redis", "app", "ssl_ca_certs"], "REDIS_APP_SSL_CA_CERTS")
    _set(
        ["redis", "app", "socket_connect_timeout_seconds"],
        "REDIS_APP_SOCKET_CONNECT_TIMEOUT_SECONDS",
        float,
    )
    _set(["redis", "app", "socket_timeout_seconds"], "REDIS_APP_SOCKET_TIMEOUT_SECONDS", float)
    _set(["redis", "app", "namespace"], "REDIS_APP_NAMESPACE")

    _set(["redis", "worker", "host"], "REDIS_WORKER_HOST")
    _set(["redis", "worker", "port"], "REDIS_WORKER_PORT", int)
    _set(["redis", "worker", "db"], "REDIS_WORKER_DB", int)
    _set(["redis", "worker", "username"], "REDIS_WORKER_USERNAME")
    _set(["redis", "worker", "password"], "REDIS_WORKER_PASSWORD")
    _set(["redis", "worker", "tls"], "REDIS_WORKER_TLS", bool)
    _set(["redis", "worker", "ssl_cert_reqs"], "REDIS_WORKER_SSL_CERT_REQS")
    _set(["redis", "worker", "ssl_ca_certs"], "REDIS_WORKER_SSL_CA_CERTS")
    _set(
        ["redis", "worker", "socket_connect_timeout_seconds"],
        "REDIS_WORKER_SOCKET_CONNECT_TIMEOUT_SECONDS",
        float,
    )
    _set(
        ["redis", "worker", "socket_timeout_seconds"],
        "REDIS_WORKER_SOCKET_TIMEOUT_SECONDS",
        float,
    )
    _set(["redis", "worker", "namespace"], "REDIS_WORKER_NAMESPACE")

    _set(["postgres", "enabled"], "POSTGRES_ENABLED", bool)
    _set(["postgres", "host"], "POSTGRES_HOST")
    _set(["postgres", "port"], "POSTGRES_PORT", int)
    _set(["postgres", "database"], "POSTGRES_DATABASE")
    _set(["postgres", "username"], "POSTGRES_USERNAME")
    _set(["postgres", "ssl_mode"], "POSTGRES_SSL_MODE")
    _set(["postgres", "min_pool_size"], "POSTGRES_MIN_POOL_SIZE", int)
    _set(["postgres", "max_pool_size"], "POSTGRES_MAX_POOL_SIZE", int)
    _set(["postgres", "connect_timeout_seconds"], "POSTGRES_CONNECT_TIMEOUT_SECONDS", int)
    _set(["postgres", "app_name"], "POSTGRES_APP_NAME")
    _set(["postgres", "schema_name"], "POSTGRES_SCHEMA_NAME")
    _set(["postgres", "memory_table"], "POSTGRES_MEMORY_TABLE")
    _set(["postgres", "chunk_table"], "POSTGRES_CHUNK_TABLE")
    _set(["postgres", "evaluation_table"], "POSTGRES_EVALUATION_TABLE")
    _set(["postgres", "default_top_k"], "POSTGRES_DEFAULT_TOP_K", int)
    _set(["postgres", "vector_index_type"], "POSTGRES_VECTOR_INDEX_TYPE")
    _set(["postgres", "embedding_dimensions"], "POSTGRES_EMBEDDING_DIMENSIONS", int)

    _set(["memory", "summary_queue_claim_idle_ms"], "MEMORY_SUMMARY_QUEUE_CLAIM_IDLE_MS", int)
    _set(["memory", "summary_queue_claim_batch_size"], "MEMORY_SUMMARY_QUEUE_CLAIM_BATCH_SIZE", int)

    _set(["middleware", "timeout_seconds"], "MIDDLEWARE_TIMEOUT_SECONDS", int)
    _set(["middleware", "max_in_flight_requests"], "MIDDLEWARE_MAX_IN_FLIGHT_REQUESTS", int)
    _set(["middleware", "rate_limit_requests"], "MIDDLEWARE_RATE_LIMIT_REQUESTS", int)
    _set(["middleware", "rate_limit_window_seconds"], "MIDDLEWARE_RATE_LIMIT_WINDOW_SECONDS", int)
    _set(
        ["middleware", "enable_distributed_rate_limit"],
        "MIDDLEWARE_ENABLE_DISTRIBUTED_RATE_LIMIT",
        bool,
    )
    _set(
        ["middleware", "distributed_rate_limit_prefix"], "MIDDLEWARE_DISTRIBUTED_RATE_LIMIT_PREFIX"
    )
    _set(
        ["middleware", "trusted_proxy_cidrs"],
        "MIDDLEWARE_TRUSTED_PROXY_CIDRS",
        lambda raw: [entry.strip() for entry in raw.split(",") if entry.strip()],
    )
    _set(
        ["middleware", "enable_distributed_backpressure"],
        "MIDDLEWARE_ENABLE_DISTRIBUTED_BACKPRESSURE",
        bool,
    )
    _set(["middleware", "distributed_backpressure_key"], "MIDDLEWARE_DISTRIBUTED_BACKPRESSURE_KEY")
    _set(
        ["middleware", "distributed_backpressure_lease_seconds"],
        "MIDDLEWARE_DISTRIBUTED_BACKPRESSURE_LEASE_SECONDS",
        int,
    )
    _set(["middleware", "enable_request_logging"], "MIDDLEWARE_ENABLE_REQUEST_LOGGING", bool)
    _set(["middleware", "enable_cors"], "MIDDLEWARE_ENABLE_CORS", bool)
    _set(
        ["middleware", "cors_allow_origins"],
        "MIDDLEWARE_CORS_ALLOW_ORIGINS",
        lambda raw: [entry.strip() for entry in raw.split(",") if entry.strip()],
    )
    _set(
        ["middleware", "cors_allow_methods"],
        "MIDDLEWARE_CORS_ALLOW_METHODS",
        lambda raw: [entry.strip().upper() for entry in raw.split(",") if entry.strip()],
    )
    _set(
        ["middleware", "cors_allow_headers"],
        "MIDDLEWARE_CORS_ALLOW_HEADERS",
        lambda raw: [entry.strip() for entry in raw.split(",") if entry.strip()],
    )
    _set(
        ["middleware", "cors_allow_credentials"],
        "MIDDLEWARE_CORS_ALLOW_CREDENTIALS",
        bool,
    )
    _set(["middleware", "enable_rate_limit"], "MIDDLEWARE_ENABLE_RATE_LIMIT", bool)
    _set(["middleware", "enable_timeout"], "MIDDLEWARE_ENABLE_TIMEOUT", bool)
    _set(["middleware", "enable_backpressure"], "MIDDLEWARE_ENABLE_BACKPRESSURE", bool)
    _set(["middleware", "enable_route_matching"], "MIDDLEWARE_ENABLE_ROUTE_MATCHING", bool)

    _set(["queue", "llm_async_enabled"], "LLM_ASYNC_ENABLED", bool)
    _set(["queue", "llm_queue_url"], "LLM_QUEUE_URL")
    _set(["queue", "llm_dlq_url"], "LLM_DLQ_URL")
    _set(["queue", "llm_result_table"], "LLM_RESULT_TABLE")
    _set(["queue", "llm_result_ttl_days"], "LLM_RESULT_TTL_DAYS", int)
    _set(["queue", "llm_receive_wait_seconds"], "LLM_RECEIVE_WAIT_SECONDS", int)
    _set(["queue", "llm_max_messages_per_poll"], "LLM_MAX_MESSAGES_PER_POLL", int)
    _set(
        ["queue", "llm_visibility_timeout_seconds"],
        "LLM_VISIBILITY_TIMEOUT_SECONDS",
        int,
    )
    _set(["queue", "llm_poll_sleep_seconds"], "LLM_POLL_SLEEP_SECONDS", float)
    _set(
        ["queue", "metrics_aggregation_queue_enabled"],
        "METRICS_AGGREGATION_QUEUE_ENABLED",
        bool,
    )
    _set(["queue", "metrics_aggregation_queue_url"], "METRICS_AGGREGATION_QUEUE_URL")
    _set(
        ["queue", "metrics_aggregation_receive_wait_seconds"],
        "METRICS_AGGREGATION_RECEIVE_WAIT_SECONDS",
        int,
    )
    _set(
        ["queue", "metrics_aggregation_max_messages_per_poll"],
        "METRICS_AGGREGATION_MAX_MESSAGES_PER_POLL",
        int,
    )
    _set(
        ["queue", "metrics_aggregation_visibility_timeout_seconds"],
        "METRICS_AGGREGATION_VISIBILITY_TIMEOUT_SECONDS",
        int,
    )
    _set(
        ["queue", "metrics_aggregation_poll_sleep_seconds"],
        "METRICS_AGGREGATION_POLL_SLEEP_SECONDS",
        float,
    )
    _set(["queue", "evaluation_queue_enabled"], "EVALUATION_QUEUE_ENABLED", bool)
    _set(["queue", "evaluation_queue_url"], "EVALUATION_QUEUE_URL")
    _set(["queue", "evaluation_receive_wait_seconds"], "EVALUATION_RECEIVE_WAIT_SECONDS", int)
    _set(
        ["queue", "evaluation_max_messages_per_poll"],
        "EVALUATION_MAX_MESSAGES_PER_POLL",
        int,
    )
    _set(
        ["queue", "evaluation_visibility_timeout_seconds"],
        "EVALUATION_VISIBILITY_TIMEOUT_SECONDS",
        int,
    )
    _set(
        ["queue", "evaluation_poll_sleep_seconds"],
        "EVALUATION_POLL_SLEEP_SECONDS",
        float,
    )
    _set(["queue", "summary_queue_enabled"], "SUMMARY_QUEUE_ENABLED", bool)
    _set(["queue", "summary_queue_url"], "SUMMARY_QUEUE_URL")
    _set(["queue", "summary_dlq_url"], "SUMMARY_DLQ_URL")
    _set(
        ["queue", "summary_receive_wait_seconds"],
        "SUMMARY_RECEIVE_WAIT_SECONDS",
        int,
    )
    _set(
        ["queue", "summary_max_messages_per_poll"],
        "SUMMARY_MAX_MESSAGES_PER_POLL",
        int,
    )
    _set(
        ["queue", "summary_visibility_timeout_seconds"],
        "SUMMARY_VISIBILITY_TIMEOUT_SECONDS",
        int,
    )
    _set(
        ["queue", "summary_poll_sleep_seconds"],
        "SUMMARY_POLL_SLEEP_SECONDS",
        float,
    )
    _set(["io", "llm_max_concurrency"], "IO_LLM_MAX_CONCURRENCY", int)
    _set(["io", "llm_answer_max_concurrency"], "IO_LLM_ANSWER_MAX_CONCURRENCY", int)
    _set(["io", "llm_planner_max_concurrency"], "IO_LLM_PLANNER_MAX_CONCURRENCY", int)
    _set(["io", "embedding_max_concurrency"], "IO_EMBEDDING_MAX_CONCURRENCY", int)
    _set(["io", "retrieval_max_concurrency"], "IO_RETRIEVAL_MAX_CONCURRENCY", int)
    _set(["io", "reranker_max_concurrency"], "IO_RERANKER_MAX_CONCURRENCY", int)
    _set(["io", "redis_max_concurrency"], "IO_REDIS_MAX_CONCURRENCY", int)
    _set(["io", "bedrock_executor_workers"], "IO_BEDROCK_EXECUTOR_WORKERS", int)

    return config


@lru_cache()
def get_settings() -> Settings:
    """Load and cache merged application settings from the config directory."""
    config_files = sorted(APP_CONFIG_DIR.glob("*_config.yaml"))
    if not config_files:
        raise FileNotFoundError(f"No config files found in {APP_CONFIG_DIR}")

    data = {}
    for config_file in config_files:
        _deep_merge(data, _load_yaml_file(config_file))

    override_dir_raw = os.getenv("APP_CONFIG_OVERRIDE_DIR", "").strip()
    if override_dir_raw:
        override_dir = Path(override_dir_raw).expanduser().resolve()
        if not override_dir.exists():
            raise FileNotFoundError(f"APP_CONFIG_OVERRIDE_DIR does not exist: {override_dir}")
        override_files = sorted(override_dir.glob("*_config.yaml"))
        for override_file in override_files:
            _deep_merge(data, _load_yaml_file(override_file))

    _load_aws_secrets_manager_env()
    data = _apply_env_overrides(data)
    return Settings(**data)


@lru_cache()
def get_prompts() -> dict:
    """Load and cache prompt definitions from the prompt config file."""
    prompt_path = APP_CONFIG_DIR / "prompt.yaml"
    return _load_yaml_file(prompt_path)


@lru_cache()
def get_evaluation_prompts() -> dict:
    """Load and cache offline evaluator (LLM-as-judge) prompts."""
    prompt_path = APP_CONFIG_DIR / "evaluation_prompt.yaml"
    return _load_yaml_file(prompt_path)
