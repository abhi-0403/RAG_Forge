"""
theme.py

Custom Gradio theme for RAG_Forge Studio.
"""

import gradio as gr


def get_theme():
    """
    Return the custom Gradio theme.
    """

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="lg",
        text_size="md",
    )

    theme.set(
        body_background_fill="#0F172A",
        body_text_color="#F8FAFC",

        block_background_fill="#1E293B",
        block_border_color="#334155",

        button_primary_background_fill="#2563EB",
        button_primary_background_fill_hover="#1D4ED8",

        button_secondary_background_fill="#334155",

        input_background_fill="#0F172A",
        input_border_color="#475569",

        checkbox_label_text_color="#F8FAFC",

        block_title_text_color="#F8FAFC",

        block_label_text_color="#CBD5E1",

        body_text_size="16px",
    )

    return theme