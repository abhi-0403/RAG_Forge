"""
app.py

Entry point for the RAG_Forge application.
"""

import os
from dotenv import load_dotenv

from src.ragforge.document_loader import DocumentLoader
from src.ragforge.text_splitter import TextSplitter
from src.ragforge.embedding_model import EmbeddingManager
from src.ragforge.vector_database import VectorStore
from src.ragforge.retriever import RAGRetriever
from src.ragforge.pipeline import RAGPipeline
from langchain_groq import ChatGroq


def main():
    """Run the complete RAG pipeline."""

    load_dotenv()

    print("=" * 60)
    print("RAG_Forge")
    print("=" * 60)

    # ----------------------------
    # Load Documents
    # ----------------------------
    loader = DocumentLoader("data/raw_pdfs")
    all_pdf_documents = loader.process_all_pdfs()

    # ----------------------------
    # Split Documents
    # ----------------------------
    splitter = TextSplitter()
    chunks = splitter.split_documents(all_pdf_documents)

    # ----------------------------
    # Generate Embeddings
    # ----------------------------
    embedding_manager = EmbeddingManager()

    texts = [doc.page_content for doc in chunks]

    embeddings = embedding_manager.generate_embeddings(texts)

    # ----------------------------
    # Store Embeddings
    # ----------------------------
    vectorstore = VectorStore()

    vectorstore.add_documents(
        chunks,
        embeddings,
    )

    # ----------------------------
    # Retriever
    # ----------------------------
    rag_retriever = RAGRetriever(
        vectorstore,
        embedding_manager,
    )

    # ----------------------------
    # LLM
    # ----------------------------
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
    )

    # ----------------------------
    # Pipeline
    # ----------------------------
    rag = RAGPipeline(
        rag_retriever,
        llm,
    )

    while True:

        query = input("\nAsk a Question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer = rag.rag_simple(query)

        print("\n")
        print("=" * 60)
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    main()