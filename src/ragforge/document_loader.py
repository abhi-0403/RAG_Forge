"""
document_loader.py

Handles loading PDF documents from a directory.

This module is directly adapted from the original notebook implementation.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader


class DocumentLoader:
    """Handles loading PDF documents from a directory."""

    def __init__(self, pdf_directory: str):
        """
        Initialize the document loader.

        Args:
            pdf_directory: Path to the directory containing PDF files.
        """
        self.pdf_directory = pdf_directory

    def process_all_pdfs(self):
        """
        Process all PDF files in the specified directory.

        Returns:
            list: List of LangChain Document objects.
        """
        all_documents = []

        pdf_dir = Path(self.pdf_directory)

        # Find all PDF files recursively
        pdf_files = list(pdf_dir.glob("**/*.pdf"))

        print(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")

            try:
                # Same loader used in the notebook
                loader = PyPDFLoader(str(pdf_file))
                # If you want PyMuPDF instead, simply replace the above line with:
                # loader = PyMuPDFLoader(str(pdf_file))

                documents = loader.load()

                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["file_type"] = "pdf"

                all_documents.extend(documents)

                print(f"  ✓ Loaded {len(documents)} pages")

            except Exception as e:
                print(f"  ✗ Error: {e}")

        print(f"\nTotal documents loaded: {len(all_documents)}")

        return all_documents