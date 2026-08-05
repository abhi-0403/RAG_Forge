"""
callbacks.py

Callbacks used by the Gradio interface.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.ragforge.embedding_model import EmbeddingManager
from src.ragforge.vector_database import VectorStore
from src.ragforge.retriever import RAGRetriever
from src.ragforge.pipeline import RAGPipeline


# ==========================================================
# Global Pipeline (Loaded Once)
# ==========================================================

load_dotenv()

embedding_manager = EmbeddingManager()

vector_store = VectorStore()

retriever = RAGRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager,
    debug=False,
)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024,
)

pipeline = RAGPipeline(
    retriever=retriever,
    llm=llm,
    debug=False,
)


# ==========================================================
# Chat Callback
# ==========================================================

def ask_question(
    question: str,
):
    """
    Answer a user question using the RAG pipeline.

    Args:
        question: User question.

    Returns:
        Generated answer.
    """

    question = question.strip()

    if not question:
        return "Please enter a question."

    try:

        answer = pipeline.rag_simple(
            query=question,
            top_k=20,
        )

        return answer

    except Exception as e:

        return f"Error:\n\n{e}"


# ==========================================================
# Vector Store Information
# ==========================================================

def get_vector_store_info():
    """
    Return vector database statistics.
    """

    try:

        count = vector_store.get_collection_count()

        return (
            "### Vector Store\n\n"
            f"Indexed Chunks : **{count}**"
        )

    except Exception as e:

        return f"Error : {e}"


# ==========================================================
# Clear Chat
# ==========================================================

def clear_chat():
    """
    Clear chat input and output.
    """

    return "", ""


# ==========================================================
# Health Check
# ==========================================================

def health_check():
    """
    Check whether the RAG backend is available.
    """

    try:

        count = vector_store.get_collection_count()

        return (
            f"✅ RAG Backend Ready\n\n"
            f"Indexed Chunks : {count}"
        )

    except Exception as e:

        return (
            "❌ Backend Error\n\n"
            f"{e}"
        )