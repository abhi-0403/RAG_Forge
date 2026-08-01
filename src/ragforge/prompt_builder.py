"""
prompt_builder.py

Handles prompt generation for the RAG pipeline.
"""

from langchain_core.prompts import PromptTemplate


class PromptBuilder:
    """Builds prompts for different RAG use cases."""

    @staticmethod
    def simple_prompt(context: str, query: str) -> str:
        """
        Build a simple RAG prompt.

        Args:
            context: Retrieved context.
            query: User query.

        Returns:
            Formatted prompt string.
        """

        prompt = f"""Use the following context to answer the question concisely.

Strictly generate the answer in bullet points.

Context:
{context}

Question:
{query}

Answer:"""

        return prompt

    @staticmethod
    def detailed_prompt(context: str, query: str) -> str:
        """
        Build a detailed prompt using PromptTemplate.
        """

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful AI assistant.

Use the following context to answer the question accurately and concisely.

Context:
{context}

Question:
{question}

Answer:
Provide a clear and informative answer based on the context above.
If the context doesn't contain enough information to answer the question,
say so.
"""
        )

        return prompt_template.format(
            context=context,
            question=query
        )

    @staticmethod
    def fallback_prompt(query: str) -> str:
        """
        Build a prompt when no context is available.
        """

        return f"""
No relevant context was found.

Answer the following question using your general knowledge.

Question:
{query}

Answer:
"""