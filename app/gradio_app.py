import os
from pathlib import Path
import sys
from uuid import uuid4

try:
    import gradio as gr
except Exception as exc:  # pragma: no cover - environment dependent import safety
    gr = None
    _GRADIO_IMPORT_ERROR = exc
else:
    _GRADIO_IMPORT_ERROR = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.llm_service import generate_response_stream  # noqa: E402

SESSION_USER_ID_PREFIX = "gradio-session"
STREAM_CHUNK_SIZE = int(os.getenv("GRADIO_STREAM_CHUNK_SIZE", "120"))
STREAM_CHUNK_DELAY_MS = int(os.getenv("GRADIO_STREAM_CHUNK_DELAY_MS", "12"))


def _resolve_session_user_id(session_user_id: str | None) -> str:
    candidate = str(session_user_id or "").strip()
    if candidate:
        return candidate
    return f"{SESSION_USER_ID_PREFIX}-{uuid4().hex}"


async def answer_question_stream(question: str, session_user_id: str | None):
    resolved_user_id = _resolve_session_user_id(session_user_id)
    question = (question or "").strip()
    if not question:
        yield "", resolved_user_id
        return
    async for partial in generate_response_stream(
        resolved_user_id,
        question,
        chunk_size=STREAM_CHUNK_SIZE,
        chunk_delay_ms=STREAM_CHUNK_DELAY_MS,
    ):
        yield partial, resolved_user_id


def _build_demo():
    if gr is None:
        return None

    with gr.Blocks(title="Simple Q&A") as gradio_demo:
        session_user_id_state = gr.State(value=None)
        question_input = gr.Textbox(
            label="Question",
            placeholder="Ask a question...",
            lines=2,
        )
        answer_output = gr.Textbox(
            label="Answer",
            lines=8,
            interactive=False,
        )
        ask_button = gr.Button("Ask")

        ask_button.click(
            fn=answer_question_stream,
            inputs=[question_input, session_user_id_state],
            outputs=[answer_output, session_user_id_state],
        )
        question_input.submit(
            fn=answer_question_stream,
            inputs=[question_input, session_user_id_state],
            outputs=[answer_output, session_user_id_state],
        )
    return gradio_demo


demo = _build_demo()


if __name__ == "__main__":
    if demo is None:
        raise RuntimeError(f"Gradio UI dependencies are unavailable: {_GRADIO_IMPORT_ERROR}")
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
