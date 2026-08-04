"""
document_loader.py

Handles loading PDF documents for RAG_Forge.

Supports:
1. Loading a single PDF
2. Loading all PDFs from a directory

Designed to be easily extendable for future document types.
"""

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
)


class DocumentLoader:
    """Handles loading PDF documents."""

    def __init__(
        self,
        pdf_directory: str = "data/raw_pdfs",
    ):
        """
        Initialize the document loader.

        Args:
            pdf_directory: Directory containing PDF files.
        """

        self.pdf_directory = Path(pdf_directory)

    # ==========================================================
    # Load Single PDF
    # ==========================================================

    def load_pdf(
        self,
        pdf_path: str,
    ):
        """
        Load a single PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of LangChain Document objects.
        """

        pdf_path = Path(pdf_path)

        print(f"\nProcessing: {pdf_path.name}")

        try:

            loader = PyPDFLoader(str(pdf_path))

            documents = loader.load()

            for doc in documents:

                doc.metadata["source_file"] = pdf_path.name
                doc.metadata["file_type"] = "pdf"

            print(
                f"✓ Loaded {len(documents)} pages"
            )

            return documents

        except Exception as e:

            print(f"Error loading PDF: {e}")

            return []

    # ==========================================================
    # Load All PDFs
    # ==========================================================

    def process_all_pdfs(self):
        """
        Load every PDF inside the directory.

        Returns:
            List of LangChain Documents.
        """

        all_documents = []

        pdf_files = sorted(
            self.pdf_directory.glob("**/*.pdf")
        )

        print(
            f"Found {len(pdf_files)} PDF files to process"
        )

        for pdf_file in pdf_files:

            documents = self.load_pdf(
                str(pdf_file)
            )

            all_documents.extend(documents)

        print(
            f"\nTotal documents loaded: "
            f"{len(all_documents)}"
        )

        return all_documents

    # ==========================================================
    # List PDFs
    # ==========================================================

    def get_pdf_files(self):
        """
        Return all PDF files.

        Returns:
            List[Path]
        """

        return sorted(
            self.pdf_directory.glob("**/*.pdf")
        )