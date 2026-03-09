import os

import pytest

from app.core import config as config_module


class _FakeSecretsClient:
    def __init__(self, payload: str):
        self._payload = payload
        self.requested_secret_id = None

    def get_secret_value(self, SecretId: str):
        self.requested_secret_id = SecretId
        return {"SecretString": self._payload}


def test_load_aws_secret_populates_missing_env(monkeypatch):
    monkeypatch.setenv("AWS_SECRETS_MANAGER_SECRET_ID", "unigraph/prod/app")
    monkeypatch.setenv("AWS_SECRETS_MANAGER_REGION", "us-east-1")
    monkeypatch.delenv("SECURITY_JWT_SECRET", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    fake_client = _FakeSecretsClient(
        '{"SECURITY_JWT_SECRET":"jwt-from-secret","POSTGRES_PASSWORD":"pg-from-secret"}'
    )

    captured = {}

    def fake_boto3_client(service_name: str, **kwargs):
        captured["service_name"] = service_name
        captured["kwargs"] = kwargs
        return fake_client

    monkeypatch.setattr(config_module.boto3, "client", fake_boto3_client)

    config_module._load_aws_secrets_manager_env()

    assert captured["service_name"] == "secretsmanager"
    assert captured["kwargs"] == {"region_name": "us-east-1"}
    assert fake_client.requested_secret_id == "unigraph/prod/app"
    assert os.getenv("SECURITY_JWT_SECRET") == "jwt-from-secret"
    assert os.getenv("POSTGRES_PASSWORD") == "pg-from-secret"


def test_load_aws_secret_does_not_override_existing_env(monkeypatch):
    monkeypatch.setenv("AWS_SECRETS_MANAGER_SECRET_ID", "unigraph/prod/app")
    monkeypatch.setenv("SECURITY_JWT_SECRET", "already-set")

    fake_client = _FakeSecretsClient('{"SECURITY_JWT_SECRET":"jwt-from-secret"}')

    def fake_boto3_client(service_name: str, **kwargs):
        return fake_client

    monkeypatch.setattr(config_module.boto3, "client", fake_boto3_client)

    config_module._load_aws_secrets_manager_env()

    assert os.getenv("SECURITY_JWT_SECRET") == "already-set"


def test_load_aws_secret_raises_for_invalid_json(monkeypatch):
    monkeypatch.setenv("AWS_SECRETS_MANAGER_SECRET_ID", "unigraph/prod/app")

    fake_client = _FakeSecretsClient("not-json")

    def fake_boto3_client(service_name: str, **kwargs):
        return fake_client

    monkeypatch.setattr(config_module.boto3, "client", fake_boto3_client)

    with pytest.raises(RuntimeError, match="not valid JSON"):
        config_module._load_aws_secrets_manager_env()
