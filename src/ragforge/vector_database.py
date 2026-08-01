"""
vector_database.py

Manages document embeddings in a ChromaDB vector store.
"""

import os
import uuid
import chromadb
import numpy as np

from typing import List, Any


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "vectorstores/chroma"
    ):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""

        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG"
                }
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
        embeddings: np.ndarray
    ):
        """
        Add documents and embeddings into ChromaDB.
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings"
            )

        print(
            f"Adding {len(documents)} documents to vector store..."
        )

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(
            zip(documents, embeddings)
        ):

            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"

            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)

            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        try:

            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )

            print(
                f"Successfully added {len(documents)} documents "
                f"to vector store"
            )

            print(
                f"Total documents in collection: "
                f"{self.collection.count()}"
            )

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise