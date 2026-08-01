"""
text_splitter.py

Splits LangChain documents into smaller chunks for better
Retrieval-Augmented Generation (RAG) performance.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    """Handles document chunking for RAG."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Size of each chunk.
            chunk_overlap: Number of overlapping characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents):
        """
        Split documents into smaller chunks for better RAG performance.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of chunked Document objects.
        """

        split_docs = self.text_splitter.split_documents(documents)

        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

        # Show example chunk
        if split_docs:
            print("\nExample Chunk:")
            print(f"Content: {split_docs[0].page_content[:200]}...")
            print(f"Metadata: {split_docs[0].metadata}")

        return split_docs