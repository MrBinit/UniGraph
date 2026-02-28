# Caching

## Purpose

UniGraph caches completed LLM responses to reduce:

- repeat model calls
- token cost
- latency for repeated identical prompts

## What Is Cached

The app caches the final chat response after:

- input guardrails
- context building
- model response generation
- output guardrails

The cached value is the final text returned to the client.

## Cache Key

Current cache key format:

- `app:cache:chat:{user_id}:{sanitized_prompt}`

Notes:

- the prompt is the sanitized prompt after input guardrails and redaction
- the cache is user-scoped
- cache matches are exact string matches

## Cache TTL

The cache uses:

- `memory.redis_ttl_seconds`

At the moment, the same TTL is shared by:

- short-term memory
- response cache
- some user-scoped memory metrics

## Cache Flow

On each chat request:

1. check cache
2. if hit, return cached response immediately
3. if miss, build context and call the model
4. store the final response in cache

Latency metrics mark cache hits separately with:

- `last_outcome = "cache_hit"`

## Benefits

- faster repeat responses
- lower cost for repeated prompts
- lower pressure on the primary and fallback model

## Current Limitations

- exact-match only; semantically similar prompts do not hit
- cache key includes the full sanitized prompt, which can become long
- no explicit cache invalidation beyond TTL expiration
- no cache versioning tied to model or prompt version

## Recommended Future Improvements

- include model version in the cache key
- include prompt policy version in the cache key
- hash very long prompt segments instead of storing raw text in the key
- add targeted invalidation when system prompt or guardrail policy changes
