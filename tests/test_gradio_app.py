import pytest

from app import gradio_app


def test_resolve_session_user_id_reuses_existing_id():
    existing = "gradio-session-existing"
    resolved = gradio_app._resolve_session_user_id(existing)
    assert resolved == existing


@pytest.mark.asyncio
async def test_answer_question_stream_assigns_and_reuses_session_user_id(monkeypatch):
    calls = []

    async def fake_generate_response_stream(
        user_id: str,
        question: str,
        *,
        chunk_size: int,
        chunk_delay_ms: int,
    ):
        calls.append((user_id, question, chunk_size, chunk_delay_ms))
        yield "partial-1"
        yield "partial-2"

    monkeypatch.setattr(gradio_app, "generate_response_stream", fake_generate_response_stream)

    first_outputs = []
    async for answer, session_user_id in gradio_app.answer_question_stream("hello", None):
        first_outputs.append((answer, session_user_id))

    assert [item[0] for item in first_outputs] == ["partial-1", "partial-2"]
    first_session_id = first_outputs[0][1]
    assert first_session_id.startswith(f"{gradio_app.SESSION_USER_ID_PREFIX}-")
    assert all(session_id == first_session_id for _, session_id in first_outputs)
    assert calls[0][0] == first_session_id

    second_outputs = []
    async for answer, session_user_id in gradio_app.answer_question_stream(
        "again", first_session_id
    ):
        second_outputs.append((answer, session_user_id))

    assert [item[0] for item in second_outputs] == ["partial-1", "partial-2"]
    assert all(session_id == first_session_id for _, session_id in second_outputs)
    assert calls[1][0] == first_session_id


@pytest.mark.asyncio
async def test_answer_question_stream_empty_prompt_returns_empty_and_keeps_session(monkeypatch):
    calls = []

    async def fake_generate_response_stream(*_args, **_kwargs):
        calls.append("called")
        yield "should-not-happen"

    monkeypatch.setattr(gradio_app, "generate_response_stream", fake_generate_response_stream)

    outputs = []
    async for answer, session_user_id in gradio_app.answer_question_stream("   ", None):
        outputs.append((answer, session_user_id))

    assert len(outputs) == 1
    assert outputs[0][0] == ""
    assert outputs[0][1].startswith(f"{gradio_app.SESSION_USER_ID_PREFIX}-")
    assert calls == []
