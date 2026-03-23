# Security

## Security Layers

UniGraph currently applies multiple security layers:

- JWT-based authentication
- user-level authorization with admin override
- domain-scoped guardrails
- prompt injection filtering
- distributed rate limiting (Redis-backed with local fallback)
- distributed backpressure admission control (Redis-backed with local fallback)
- short-term memory encryption at rest in Redis
- Redis TLS for in-transit encryption
- Redis client and namespace separation for app and worker

## Authentication

Authentication uses bearer tokens backed by JWT.

For local/dev UI flows, the app can issue JWTs from a username/password login endpoint:

- `POST /api/v1/auth/login`
- user records come from `SECURITY_LOGIN_USERS_JSON` (no built-in fallback user)

Token claims:

- `sub`: user id
- `roles`: role list
- `iss`: issuer
- `aud`: audience
- `iat`: issued-at timestamp
- `exp`: expiration timestamp

Config:

- `jwt_secret`
- `jwt_algorithm`
- `jwt_issuer`
- `jwt_audience`
- `jwt_exp_minutes`

Secret source:

- prefers `SECURITY_JWT_SECRET` (then `JWT_SECRET`) from environment
- falls back to `app/config/security_config.yaml`

Operational note:

- production startup validates secret strength and rejects placeholder/default values
- token validation enforces both issuer (`iss`) and audience (`aud`) checks
- in production, use environment secrets or a secret manager

## Authorization

Two authorization checks are implemented:

- user access: a user can access only their own resources
- admin access: admin roles can access protected operational routes

Current admin-only route:

- `GET /api/v1/ops/status`

## Guardrails

Guardrails enforce that UniGraph stays in its academic domain.

Input protections:

- blocks empty input
- blocks oversized input
- blocks configured harmful regex patterns
- blocks out-of-scope prompts
- redacts emails, phone numbers, API keys, and card-like data

Context protections:

- strips malformed messages
- prepends a policy system message
- removes likely prompt-injection content
- removes out-of-scope user content
- limits context message count

Output protections:

- blocks configured dangerous output patterns
- redacts sensitive values
- truncates oversized output

Guardrail goal:

- protect tokens from being burned on irrelevant requests
- reduce prompt-injection risk
- keep the assistant limited to university, professor, research-lab, and course discovery

## Abuse Protection

Request-level protections:

- Redis-backed distributed rate limiting
- timeout enforcement
- Redis-backed distributed backpressure
- trusted-proxy CIDR enforcement before accepting `X-Forwarded-For`

Fallback model:

- if Redis is unavailable, middleware falls back to local in-memory controls
- this avoids total traffic outage while preserving some protection
- local backpressure fallback uses an atomic lock+counter gate (race-safe admission/rejection), not private semaphore internals

Rate limiting key:

- `user_id + client_ip + request_path`

This hybrid key reduces abuse by:

- limiting anonymous IP bursts
- limiting authenticated user bursts
- avoiding cross-route interference

Proxy trust note:

- `X-Forwarded-For` is used only when the immediate peer IP belongs to `MIDDLEWARE_TRUSTED_PROXY_CIDRS`
- this prevents direct client spoofing of source IP headers

## Data Protection

### Short-Term Memory Encryption

Short-term memory payloads are encrypted before being written to Redis.

Current implementation:

- standard AEAD encryption with AES-GCM
- stored payload prefix: `enc:v2:`
- authenticated decryption rejects tampered payloads

Key source:

- requires `MEMORY_ENCRYPTION_KEY` in environment
- startup rejects missing, weak, or reused key material
- key must be different from JWT signing secret

Compatibility note:

- `enc:v1:` payloads remain readable during migration
- legacy plaintext JSON payloads remain readable for backward compatibility
- all new writes use `enc:v2:`

### Evaluation Trace Encryption

Evaluation traces (including prompt/answer content) are encrypted before storage in Redis using the same `enc:v2:` envelope.

Compatibility:

- legacy plaintext traces remain readable for backward compatibility
- all new trace writes are encrypted by default

### Redis Isolation

Redis is split by role:

- app client
- worker client

Each role has its own:

- host/port/db credentials
- username and password fields
- TLS settings (`tls`, `ssl_cert_reqs`, `ssl_ca_certs`)
- namespace prefix

This reduces blast radius if one runtime is compromised.

## Runtime Hardening

- API docs/OpenAPI exposure is configurable via `APP_DOCS_ENABLED`
- production health checks use `/healthz` instead of docs endpoints
- `.env` should be owner-readable only (`chmod 600 .env`)

## Security Gaps To Track

Remaining hardening items:

- move secrets out of YAML in all production deployments
- add key rotation/versioning policy for memory encryption keys
- add external audit logging and security event aggregation
- add stricter Redis ACL command restrictions per role
