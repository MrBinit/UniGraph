import base64
import hashlib
import hmac
import json

import pytest

from app.core import memory_crypto


@pytest.fixture(autouse=True)
def _set_memory_encryption_key(monkeypatch):
    monkeypatch.setenv("MEMORY_ENCRYPTION_KEY", "m" * 64)


def _legacy_v1_payload(payload: dict) -> str:
    plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    nonce = b"0" * memory_crypto._LEGACY_NONCE_LEN
    ciphertext = memory_crypto._legacy_xor_stream(plaintext, nonce)
    mac_key = memory_crypto._legacy_derive_key(b"memory-auth-v1")
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    token = base64.urlsafe_b64encode(nonce + ciphertext + tag).decode("ascii")
    return f"{memory_crypto._LEGACY_ENC_PREFIX}{token}"


def test_encrypt_decrypt_round_trip_uses_v2_aead():
    payload = {"summary": "hello", "messages": [{"role": "user", "content": "hi"}]}

    encrypted = memory_crypto.encrypt_memory_payload(payload)
    decrypted = memory_crypto.decrypt_memory_payload(encrypted)

    assert encrypted.startswith("enc:v2:")
    assert decrypted == payload


def test_decrypt_rejects_tampered_v2_payload():
    encrypted = memory_crypto.encrypt_memory_payload({"summary": "hello"})
    tampered = encrypted[:-2] + ("A" if encrypted[-2] != "A" else "B") + encrypted[-1]

    assert memory_crypto.decrypt_memory_payload(tampered) is None


def test_decrypt_supports_legacy_v1_payload():
    payload = {"conversation_id": "abc123", "prompt": "hello"}
    legacy = _legacy_v1_payload(payload)

    decrypted = memory_crypto.decrypt_memory_payload(legacy)

    assert decrypted == payload


def test_decrypt_supports_plaintext_json_payload():
    raw = json.dumps({"summary": "legacy-plaintext"})

    decrypted = memory_crypto.decrypt_memory_payload(raw)

    assert decrypted == {"summary": "legacy-plaintext"}


def test_encrypt_requires_memory_encryption_key(monkeypatch):
    monkeypatch.delenv("MEMORY_ENCRYPTION_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MEMORY_ENCRYPTION_KEY is required"):
        memory_crypto.encrypt_memory_payload({"summary": "hello"})
