"""
app.py

Query interface for RAG_Forge.

This script assumes the vector database has already been
created using ingest.py.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.ragforge.embedding_model import EmbeddingManager
from src.ragforge.vector_database import VectorStore
from src.ragforge.retriever import RAGRetriever
from src.ragforge.pipeline import RAGPipeline


def main():
    """Run the RAG query application."""

    load_dotenv()

    print("=" * 80)
    print("RAG_Forge")
    print("=" * 80)

    # ==========================================================
    # Embedding Model
    # ==========================================================

    embedding_manager = EmbeddingManager()

    # ==========================================================
    # Existing Vector Database
    # ==========================================================

    vector_store = VectorStore()

    print(
        f"\nVector Store Count : "
        f"{vector_store.get_collection_count()}"
    )

    # ==========================================================
    # Retriever
    # ==========================================================

    rag_retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        debug=False,
    )

    # ==========================================================
    # LLM
    # ==========================================================

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
    )

    # ==========================================================
    # Pipeline
    # ==========================================================

    rag = RAGPipeline(
        retriever=rag_retriever,
        llm=llm,
        debug=False,
    )

    # ==========================================================
    # Chat Loop
    # ==========================================================

    print("\nRAG_Forge is Ready!")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("Ask a Question: ").strip()

        if query.lower() == "exit":
            print("\nGoodbye!")
            break

        if not query:
            continue

        answer = rag.rag_simple(
            query=query,
            top_k=20,
        )

        print("\n" + "=" * 80)
        print(answer)
        print("=" * 80)


if __name__ == "__main__":
    main()