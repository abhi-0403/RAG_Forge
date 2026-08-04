"""
vector_database.py

Manages document embeddings in a ChromaDB vector store.
"""

import os
import chromadb
import numpy as np

from typing import List, Any


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "vector_store/chroma",
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory where ChromaDB is stored.
        """

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection."""

        try:

            os.makedirs(
                self.persist_directory,
                exist_ok=True,
            )

            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG_Forge"
                },
            )

            print(
                f"Vector store initialized. Collection: {self.collection_name}"
            )

            print(
                f"Existing documents in collection: "
                f"{self.collection.count()}"
            )

        except Exception as e:

            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
        document_id: str,
    ):
        """
        Add chunked documents into ChromaDB.

        Args:
            documents: List of LangChain Documents.
            embeddings: Generated embeddings.
            document_id: Unique document identifier.
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings."
            )

        print(
            f"Adding {len(documents)} chunks to vector store..."
        )

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(
            zip(documents, embeddings)
        ):

            chunk_id = f"{document_id}_{i}"

            ids.append(chunk_id)

            metadata = dict(doc.metadata)

            metadata["document_id"] = document_id
            metadata["chunk_index"] = i
            metadata["content_length"] = len(doc.page_content)

            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(
                embedding.tolist()
            )

        try:

            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )

            print(
                f"Successfully added {len(documents)} chunks."
            )

            print(
                f"Total documents in collection: "
                f"{self.collection.count()}"
            )

        except Exception as e:

            print(f"Error adding documents: {e}")
            raise

    def delete_document(
        self,
        document_id: str,
    ):
        """
        Delete all chunks belonging to a document.

        Args:
            document_id: Unique document identifier.
        """

        try:

            print(
                f"Deleting document: {document_id}"
            )

            self.collection.delete(
                where={
                    "document_id": document_id
                }
            )

            print(
                "Document deleted successfully."
            )

        except Exception as e:

            print(
                f"Error deleting document: {e}"
            )
            raise

    def get_collection_count(self) -> int:
        """
        Return total number of vectors.

        Returns:
            Total vectors stored.
        """

        return self.collection.count()

    def reset_collection(self):
        """
        Delete all vectors from the collection.
        Useful during development/testing.
        """

        try:

            self.client.delete_collection(
                self.collection_name
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG_Forge"
                },
            )

            print("Collection reset successfully.")

        except Exception as e:

            print(f"Error resetting collection: {e}")
            raise