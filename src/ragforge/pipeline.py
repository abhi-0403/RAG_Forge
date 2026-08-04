"""
pipeline.py

RAG Pipeline integrating Retriever and Groq LLM.
"""

from typing import Dict, Any

from src.ragforge.prompt_builder import PromptBuilder


class RAGPipeline:
    """Handles end-to-end Retrieval Augmented Generation."""

    def __init__(
        self,
        retriever,
        llm,
        debug: bool = False,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            retriever: RAGRetriever instance.
            llm: ChatGroq instance.
            debug: Enable pipeline debugging.
        """

        self.retriever = retriever
        self.llm = llm
        self.debug = debug

    def rag_simple(
        self,
        query: str,
        top_k: int = 10,
    ) -> str:
        """
        Simple RAG Pipeline.
        """

        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        # ------------------------------------------------------
        # Debug: Retrieved Chunks
        # ------------------------------------------------------

        if self.debug:

            print("\n")
            print("=" * 100)
            print("RETRIEVED CHUNKS")
            print("=" * 100)

            for i, doc in enumerate(results):

                print(f"\nChunk {i+1}")
                print("-" * 100)

                print(
                    f"Similarity : {doc['similarity_score']:.4f}"
                )
                print(
                    f"Page       : {doc['metadata'].get('page')}"
                )
                print(
                    f"Source     : {doc['metadata'].get('source_file')}"
                )

                print("-" * 100)
                print(doc["content"])

        context = "\n\n".join(
            doc["content"] for doc in results
        ) if results else ""

        if not context:

            prompt = PromptBuilder.fallback_prompt(query)

            response = self.llm.invoke(prompt)

            return response.content

        prompt = PromptBuilder.simple_prompt(
            context=context,
            query=query,
        )

        response = self.llm.invoke(prompt)

        return response.content

    def rag_advanced(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.2,
        return_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Advanced RAG Pipeline.
        """

        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=min_score,
        )

        if not results:

            return {
                "answer": "No relevant context found.",
                "sources": [],
                "confidence": 0.0,
                "context": "",
            }

        if self.debug:

            print("\n")
            print("=" * 100)
            print("RETRIEVED CHUNKS")
            print("=" * 100)

            for i, doc in enumerate(results):

                print(f"\nChunk {i+1}")
                print("-" * 100)

                print(
                    f"Similarity : {doc['similarity_score']:.4f}"
                )
                print(
                    f"Page       : {doc['metadata'].get('page')}"
                )
                print(
                    f"Source     : {doc['metadata'].get('source_file')}"
                )

                print("-" * 100)
                print(doc["content"])

        context = "\n\n".join(
            doc["content"] for doc in results
        )

        sources = [
            {
                "source": doc["metadata"].get(
                    "source_file",
                    doc["metadata"].get(
                        "source",
                        "unknown",
                    ),
                ),
                "page": doc["metadata"].get(
                    "page",
                    "unknown",
                ),
                "score": doc["similarity_score"],
                "preview": doc["content"][:300] + "...",
            }
            for doc in results
        ]

        confidence = max(
            doc["similarity_score"]
            for doc in results
        )

        prompt = PromptBuilder.detailed_prompt(
            context=context,
            query=query,
        )

        response = self.llm.invoke(prompt)

        output = {
            "answer": response.content,
            "sources": sources,
            "confidence": confidence,
        }

        if return_context:
            output["context"] = context

        return output