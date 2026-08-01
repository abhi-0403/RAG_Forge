"""
llm.py

Handles Groq LLM initialization and response generation.
"""

import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


class GroqLLM:
    """Handles response generation using Groq LLM."""

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: str = None,
    ):
        """
        Initialize Groq LLM.

        Args:
            model_name: Groq model name.
            api_key: Groq API key.
        """

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Groq API key is required. "
                "Set GROQ_API_KEY environment variable."
            )

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024,
        )

        print(
            f"Initialized Groq LLM with model: {self.model_name}"
        )

    def generate_response(
        self,
        query: str,
        context: str,
        max_length: int = 500,
    ) -> str:
        """
        Generate response using retrieved context.
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
""",
        )

        formatted_prompt = prompt_template.format(
            context=context,
            question=query,
        )

        try:
            messages = [
                HumanMessage(content=formatted_prompt)
            ]

            response = self.llm.invoke(messages)

            return response.content

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_response_simple(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Simple response generation.
        """

        simple_prompt = f"""
Based on this context:

{context}

Question:
{query}

Answer:
"""

        try:
            messages = [
                HumanMessage(content=simple_prompt)
            ]

            response = self.llm.invoke(messages)

            return response.content

        except Exception as e:
            return f"Error: {str(e)}"