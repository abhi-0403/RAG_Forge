"""
studio.py

Entry point for RAG_Forge Studio.
"""

from src.ragforge.ui.layout import create_app


def main():
    """Launch RAG_Forge Studio."""

    app = create_app()

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()