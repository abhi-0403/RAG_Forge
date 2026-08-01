"""
pipeline.py

RAG Pipeline integrating Retriever and Groq LLM.
"""

from typing import Dict, Any

from src.ragforge.prompt_builder import PromptBuilder


class RAGPipeline:
    """Handles end-to-end Retrieval Augmented Generation."""

    def __init__(self, retriever, llm):
        """
        Initialize the RAG pipeline.

        Args:
            retriever: RAGRetriever instance.
            llm: ChatGroq instance.
        """
        self.retriever = retriever
        self.llm = llm

    def rag_simple(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        """
        Simple RAG Pipeline.

        Args:
            query: User query.
            top_k: Number of documents to retrieve.

        Returns:
            LLM response.
        """

        # Retrieve relevant documents
        results = self.retriever.retrieve(
            query,
            top_k=top_k,
        )

        context = "\n\n".join(
            [doc["content"] for doc in results]
        ) if results else ""

        # No relevant context found
        if not context:
            prompt = PromptBuilder.fallback_prompt(query)

            response = self.llm.invoke(prompt)

            return response.content

        # Build Prompt
        prompt = PromptBuilder.simple_prompt(
            context=context,
            query=query,
        )

        # Generate Response
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

        Args:
            query: User query.
            top_k: Number of retrieved documents.
            min_score: Minimum similarity score.
            return_context: Whether to include retrieved context.

        Returns:
            Dictionary containing answer, sources, confidence and context.
        """

        results = self.retriever.retrieve(
            query,
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

        context = "\n\n".join(
            [doc["content"] for doc in results]
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

        # Build Prompt
        prompt = PromptBuilder.detailed_prompt(
            context=context,
            query=query,
        )

        # Generate Response
        response = self.llm.invoke(prompt)

        output = {
            "answer": response.content,
            "sources": sources,
            "confidence": confidence,
        }

        if return_context:
            output["context"] = context

        return output