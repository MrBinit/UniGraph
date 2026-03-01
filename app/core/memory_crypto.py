import base64
import hashlib
import hmac
import json
import os
import secrets

from app.core.config import get_settings

settings = get_settings()

_ENC_PREFIX = "enc:v1:"
_NONCE_LEN = 16
_TAG_LEN = 32

def _master_secret() -> bytes:
    """Return the base secret used to derive memory encryption keys."""
    configured = os.getenv("MEMORY_ENCRYPTION_KEY", "").strip()
    if configured:
        return configured.encode("utf-8")
    return settings.security.jwt_secret.encode("utf-8")


def _derive_key(label: bytes) -> bytes:
    """Derive a stable sub-key for a specific memory crypto purpose."""
    return hmac.new(_master_secret(), label, hashlib.sha256).digest()

_ENC_KEY = _derive_key(b"memory-encryption-v1")
_MAC_KEY = _derive_key(b"memory-auth-v1")

def _xor_stream(data: bytes, nonce: bytes) -> bytes:
    """Encrypt or decrypt bytes with the internal HMAC-derived stream cipher."""
    out = bytearray(len(data))
    counter = 0
    offset = 0
    while offset < len(data):
        block = hmac.new(
            _ENC_KEY,
            nonce + counter.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        chunk_len = min(len(block), len(data) - offset)
        for i in range(chunk_len):
            out[offset + i] = data[offset + i] ^ block[i]
        offset += chunk_len
        counter += 1
    return bytes(out)

def encrypt_memory_payload(payload: dict) -> str:
    """Encrypt and authenticate a memory payload for Redis storage."""
    plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    nonce = secrets.token_bytes(_NONCE_LEN)
    ciphertext = _xor_stream(plaintext, nonce)
    tag = hmac.new(_MAC_KEY, nonce + ciphertext, hashlib.sha256).digest()
    token = base64.urlsafe_b64encode(nonce + ciphertext + tag).decode("ascii")
    return f"{_ENC_PREFIX}{token}"

def decrypt_memory_payload(raw: str) -> dict | None:
    """Decrypt a Redis memory payload and fall back to legacy plaintext parsing."""
    if not isinstance(raw, str) or not raw:
        return None

    if not raw.startswith(_ENC_PREFIX):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    encoded = raw[len(_ENC_PREFIX) :]
    try:
        blob = base64.urlsafe_b64decode(encoded.encode("ascii"))
    except Exception:
        return None

    if len(blob) <= _NONCE_LEN + _TAG_LEN:
        return None

    nonce = blob[:_NONCE_LEN]
    tag = blob[-_TAG_LEN:]
    ciphertext = blob[_NONCE_LEN:-_TAG_LEN]

    expected = hmac.new(_MAC_KEY, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        return None

    plaintext = _xor_stream(ciphertext, nonce)
    try:
        parsed = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None
