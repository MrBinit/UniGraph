import base64
import hashlib
import hmac
import json
import os
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_ENC_PREFIX = "enc:v2:"
_AEAD_NONCE_LEN = 12

_LEGACY_ENC_PREFIX = "enc:v1:"
_LEGACY_NONCE_LEN = 16
_LEGACY_TAG_LEN = 32


def _master_secret() -> bytes:
    """Return the base secret used to derive memory encryption keys."""
    configured = os.getenv("MEMORY_ENCRYPTION_KEY", "").strip()
    if not configured:
        raise RuntimeError("MEMORY_ENCRYPTION_KEY is required for memory encryption.")
    return configured.encode("utf-8")


def _derive_aead_key() -> bytes:
    """Derive a fixed-length AES-GCM key from the configured master secret."""
    return hashlib.sha256(_master_secret()).digest()


def _legacy_derive_key(label: bytes) -> bytes:
    """Derive legacy v1 keys for backward-compatible decryption."""
    return hmac.new(_master_secret(), label, hashlib.sha256).digest()


def _legacy_xor_stream(data: bytes, nonce: bytes) -> bytes:
    """Decrypt legacy v1 ciphertext using the original stream transform."""
    enc_key = _legacy_derive_key(b"memory-encryption-v1")
    out = bytearray(len(data))
    counter = 0
    offset = 0
    while offset < len(data):
        block = hmac.new(
            enc_key,
            nonce + counter.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        chunk_len = min(len(block), len(data) - offset)
        for i in range(chunk_len):
            out[offset + i] = data[offset + i] ^ block[i]
        offset += chunk_len
        counter += 1
    return bytes(out)


def _parse_json_object(raw: str | bytes) -> dict | None:
    """Parse and validate a JSON object payload."""
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def encrypt_memory_payload(payload: dict) -> str:
    """Encrypt and authenticate a memory payload using AES-GCM."""
    plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    nonce = secrets.token_bytes(_AEAD_NONCE_LEN)
    ciphertext = AESGCM(_derive_aead_key()).encrypt(nonce, plaintext, associated_data=None)
    token = base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")
    return f"{_ENC_PREFIX}{token}"


def _decrypt_v2_payload(encoded: str) -> dict | None:
    """Decrypt an `enc:v2` AES-GCM payload."""
    try:
        blob = base64.urlsafe_b64decode(encoded.encode("ascii"))
    except Exception:
        return None

    if len(blob) <= _AEAD_NONCE_LEN:
        return None

    nonce = blob[:_AEAD_NONCE_LEN]
    ciphertext = blob[_AEAD_NONCE_LEN:]
    aead_key = _derive_aead_key()
    try:
        plaintext = AESGCM(aead_key).decrypt(nonce, ciphertext, associated_data=None)
    except Exception:
        return None
    try:
        decoded = plaintext.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return _parse_json_object(decoded)


def _decrypt_v1_payload(encoded: str) -> dict | None:
    """Decrypt and verify a legacy `enc:v1` payload."""
    try:
        blob = base64.urlsafe_b64decode(encoded.encode("ascii"))
    except Exception:
        return None

    if len(blob) <= _LEGACY_NONCE_LEN + _LEGACY_TAG_LEN:
        return None

    nonce = blob[:_LEGACY_NONCE_LEN]
    tag = blob[-_LEGACY_TAG_LEN:]
    ciphertext = blob[_LEGACY_NONCE_LEN:-_LEGACY_TAG_LEN]

    mac_key = _legacy_derive_key(b"memory-auth-v1")
    expected = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        return None

    plaintext = _legacy_xor_stream(ciphertext, nonce)
    try:
        decoded = plaintext.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return _parse_json_object(decoded)


def decrypt_memory_payload(raw: str) -> dict | None:
    """Decrypt a Redis memory payload with v2, v1, and plaintext compatibility."""
    if not isinstance(raw, str) or not raw:
        return None

    if raw.startswith(_ENC_PREFIX):
        return _decrypt_v2_payload(raw[len(_ENC_PREFIX) :])

    if raw.startswith(_LEGACY_ENC_PREFIX):
        return _decrypt_v1_payload(raw[len(_LEGACY_ENC_PREFIX) :])

    return _parse_json_object(raw)
