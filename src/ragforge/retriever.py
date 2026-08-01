"""
retriever.py

Handles query-based retrieval from the ChromaDB vector store.
"""

from typing import List, Dict, Any

from src.ragforge.vector_database import VectorStore
from src.ragforge.embedding_model import EmbeddingManager


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: Vector store containing document embeddings.
            embedding_manager: Manager for generating query embeddings.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.

        Returns:
            List of retrieved documents.
        """

        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:

                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (
                    doc_id,
                    document,
                    metadata,
                    distance,
                ) in enumerate(
                    zip(
                        ids,
                        documents,
                        metadatas,
                        distances,
                    )
                ):

                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1,
                            }
                        )

                print(
                    f"Retrieved {len(retrieved_docs)} documents (after filtering)"
                )

            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []