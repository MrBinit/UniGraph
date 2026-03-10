import os
from pathlib import Path
import sys

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.llm_service import generate_response_stream

DEFAULT_USER_ID = "gradio-user"
STREAM_CHUNK_SIZE = int(os.getenv("GRADIO_STREAM_CHUNK_SIZE", "120"))
STREAM_CHUNK_DELAY_MS = int(os.getenv("GRADIO_STREAM_CHUNK_DELAY_MS", "12"))


async def answer_question_stream(question: str):
    question = (question or "").strip()
    if not question:
        yield ""
        return
    async for partial in generate_response_stream(
        DEFAULT_USER_ID,
        question,
        chunk_size=STREAM_CHUNK_SIZE,
        chunk_delay_ms=STREAM_CHUNK_DELAY_MS,
    ):
        yield partial


with gr.Blocks(title="Simple Q&A") as demo:
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
        inputs=question_input,
        outputs=answer_output,
    )
    question_input.submit(
        fn=answer_question_stream,
        inputs=question_input,
        outputs=answer_output,
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
