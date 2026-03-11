from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.infra.bedrock_client import aconverse, aconverse_stream_text
from app.infra.io_limiters import dependency_limiter


@dataclass
class _CompatMessage:
    content: str


@dataclass
class _CompatChoice:
    message: _CompatMessage


@dataclass
class _CompatUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _CompatResponse:
    choices: list[_CompatChoice]
    usage: _CompatUsage


def _to_bedrock_payload(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Convert role/content chat messages into Bedrock Converse payload fields."""
    system_blocks: list[dict[str, str]] = []
    convo_messages: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue

        if role == "system":
            system_blocks.append({"text": text})
            continue
        if role not in {"user", "assistant"}:
            continue

        convo_messages.append(
            {
                "role": role,
                "content": [{"text": text}],
            }
        )

    if not convo_messages:
        convo_messages = [{"role": "user", "content": [{"text": " "}]}]

    return system_blocks, convo_messages


def _from_bedrock_response(response: dict[str, Any]) -> _CompatResponse:
    """Normalize Bedrock Converse response into the subset used by services."""
    content_blocks = response.get("output", {}).get("message", {}).get("content", [])
    texts: list[str] = []
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
    output_text = "\n".join(texts)

    usage = response.get("usage", {})
    prompt_tokens = int(usage.get("inputTokens") or 0)
    completion_tokens = int(usage.get("outputTokens") or 0)
    total_tokens = int(usage.get("totalTokens") or (prompt_tokens + completion_tokens))

    return _CompatResponse(
        choices=[_CompatChoice(message=_CompatMessage(content=output_text))],
        usage=_CompatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


class _BedrockCompatCompletions:
    async def create(
        self, *, model: str, messages: list[dict[str, Any]], timeout: int | None = None
    ):
        """Async chat completion entrypoint backed by Bedrock Converse."""
        system_blocks, convo_messages = _to_bedrock_payload(messages)
        payload: dict[str, Any] = {
            "modelId": model,
            "messages": convo_messages,
        }
        if system_blocks:
            payload["system"] = system_blocks

        async with dependency_limiter("llm"):
            response = await aconverse(payload, timeout=timeout)
        return _from_bedrock_response(response)

    async def stream(
        self, *, model: str, messages: list[dict[str, Any]], timeout: int | None = None
    ) -> AsyncIterator[str]:
        """Yield true Bedrock token deltas for chat responses."""
        system_blocks, convo_messages = _to_bedrock_payload(messages)
        payload: dict[str, Any] = {
            "modelId": model,
            "messages": convo_messages,
        }
        if system_blocks:
            payload["system"] = system_blocks

        async with dependency_limiter("llm"):
            async for delta in aconverse_stream_text(payload, timeout=timeout):
                if delta:
                    yield delta


class _BedrockCompatChat:
    def __init__(self):
        self.completions = _BedrockCompatCompletions()


class _BedrockCompatClient:
    def __init__(self):
        self.chat = _BedrockCompatChat()


client = _BedrockCompatClient()
