"""
ingest.py

Entry point for incremental document ingestion.

This script:

1. Scans data/raw_pdfs
2. Detects new/modified PDFs
3. Generates embeddings
4. Stores vectors into ChromaDB
5. Updates metadata.json
"""

from dotenv import load_dotenv

from src.ragforge.document_loader import DocumentLoader
from src.ragforge.text_splitter import TextSplitter
from src.ragforge.embedding_model import EmbeddingManager
from src.ragforge.vector_database import VectorStore
from src.ragforge.ingest_manager import IngestionManager


def main():
    """Run incremental document ingestion."""

    load_dotenv()

    print("=" * 80)
    print("RAG_Forge - Incremental Ingestion")
    print("=" * 80)

    # ----------------------------------------------------------
    # Initialize Components
    # ----------------------------------------------------------

    document_loader = DocumentLoader()

    text_splitter = TextSplitter()

    embedding_manager = EmbeddingManager()

    vector_store = VectorStore()

    ingestion_manager = IngestionManager()

    # ----------------------------------------------------------
    # Run Incremental Ingestion
    # ----------------------------------------------------------

    ingestion_manager.ingest(
        document_loader=document_loader,
        text_splitter=text_splitter,
        embedding_manager=embedding_manager,
        vector_store=vector_store,
    )

    print("\n")
    print("=" * 80)
    print("Ingestion Completed Successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()