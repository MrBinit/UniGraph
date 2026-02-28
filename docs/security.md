# Security

## Security Layers

UniGraph currently applies multiple security layers:

- JWT-based authentication
- user-level authorization with admin override
- domain-scoped guardrails
- prompt injection filtering
- rate limiting
- timeouts and backpressure
- short-term memory encryption at rest in Redis
- Redis client and namespace separation for app and worker

## Authentication

Authentication uses bearer tokens backed by JWT.

Token claims:

- `sub`: user id
- `roles`: role list
- `iss`: issuer
- `iat`: issued-at timestamp
- `exp`: expiration timestamp

Config:

- `jwt_secret`
- `jwt_algorithm`
- `jwt_issuer`
- `jwt_exp_minutes`

Secret source:

- prefers `JWT_SECRET` environment variable
- falls back to `app/config/security_config.yaml`

Operational note:

- the config fallback is acceptable for development
- in production, use an environment secret or a secret manager

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

- rate limiting
- timeout enforcement
- backpressure

Rate limiting key:

- `user_id + client_ip + request_path`

This hybrid key reduces abuse by:

- limiting anonymous IP bursts
- limiting authenticated user bursts
- avoiding cross-route interference

## Data Protection

### Short-Term Memory Encryption

Short-term memory payloads are encrypted before being written to Redis.

Current implementation:

- a custom stream-style encryption layer built from HMAC-SHA256-derived keystream blocks
- HMAC-SHA256 authentication tag
- stored payload prefix: `enc:v1:`

Key source:

- prefers `MEMORY_ENCRYPTION_KEY`
- falls back to `jwt_secret`

Important note:

- this protects Redis contents from casual inspection
- for stronger production-grade cryptography, replace this with a standard AEAD library such as AES-GCM or ChaCha20-Poly1305 when dependency installation is available

### Redis Isolation

Redis is split by role:

- app client
- worker client

Each role has its own:

- host/port/db credentials
- username and password fields
- namespace prefix

This reduces blast radius if one runtime is compromised.

## Security Gaps To Track

The system is in a good intermediate state, but these are still future hardening items:

- move secrets out of YAML in all production deployments
- replace custom memory crypto with standard AEAD
- add transport security and TLS for Redis if deployed remotely
- add external audit logging and security event aggregation
- add stricter Redis ACL command restrictions per role
