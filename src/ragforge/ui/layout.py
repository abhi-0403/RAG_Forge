"""
layout.py

Main layout for RAG_Forge Studio.
"""

import gradio as gr

from src.ragforge.ui.theme import get_theme
from src.ragforge.ui.chat_tab import build_chat_tab
from src.ragforge.ui.ingestion_tab import build_ingestion_tab
from src.ragforge.ui.documents_tab import build_documents_tab
from src.ragforge.ui.debug_tab import build_debug_tab
from src.ragforge.ui.settings import build_settings_tab


def create_app():
    """
    Create the complete Gradio application.
    """

    with gr.Blocks(
        theme=get_theme(),
        title="RAG_Forge Studio",
        fill_height=True,
    ) as app:

        # =====================================================
        # Header
        # =====================================================

        gr.Markdown(
            """
# 🔥 RAG_Forge Studio

### A Modular Retrieval-Augmented Generation Framework

---

Build • Ingest • Retrieve • Debug • Evaluate
"""
        )

        # =====================================================
        # Tabs
        # =====================================================

        with gr.Tabs():

            with gr.Tab("💬 Chat"):
                build_chat_tab()

            with gr.Tab("📥 Ingestion"):
                build_ingestion_tab()

            with gr.Tab("📄 Documents"):
                build_documents_tab()

            with gr.Tab("🐞 Debug"):
                build_debug_tab()

            with gr.Tab("⚙️ Settings"):
                build_settings_tab()

    return app