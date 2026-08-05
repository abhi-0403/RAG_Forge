"""
chat_tab.py

Chat interface for RAG_Forge Studio.
"""

import gradio as gr

from src.ragforge.ui.callbacks import (
    ask_question,
    clear_chat,
    get_vector_store_info,
)


def build_chat_tab():
    """
    Build the Chat tab.
    """

    gr.Markdown("## 💬 Chat with your Documents")

    with gr.Row():

        # ======================================================
        # Left Column
        # ======================================================

        with gr.Column(scale=3):

            question = gr.Textbox(
                label="Ask a Question",
                placeholder="Example: Explain Unit II of Computer Networks...",
                lines=3,
            )

            with gr.Row():

                ask_btn = gr.Button(
                    "🚀 Ask",
                    variant="primary",
                )

                clear_btn = gr.Button(
                    "🗑 Clear",
                )

        # ======================================================
        # Right Column
        # ======================================================

        with gr.Column(scale=1):

            vector_info = gr.Markdown(
                value=get_vector_store_info(),
            )

    # ==========================================================
    # Answer
    # ==========================================================

    answer = gr.Markdown(
        label="Answer",
    )

    # ==========================================================
    # Events
    # ==========================================================

    ask_btn.click(
        fn=ask_question,
        inputs=question,
        outputs=answer,
    )

    question.submit(
        fn=ask_question,
        inputs=question,
        outputs=answer,
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[
            question,
            answer,
        ],
    )