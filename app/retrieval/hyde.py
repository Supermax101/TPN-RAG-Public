"""
HyDE (Hypothetical Document Embeddings) Retriever.

HyDE improves retrieval by generating a hypothetical answer to the query,
then using that answer's embedding for similarity search. This bridges
the semantic gap between questions and documents.

Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
       https://arxiv.org/abs/2212.10496

Example:
    >>> hyde = HyDERetriever(llm, retriever)
    >>> results = hyde.retrieve("What is the protein dose for neonates?")
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol, Any

logger = logging.getLogger(__name__)


class LLMProtocol(Protocol):
    """Protocol for LLM implementations."""

    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        ...


@dataclass
class HyDEConfig:
    """Configuration for HyDE retrieval."""

    # Number of hypothetical documents to generate
    num_hypotheticals: int = 1

    # Temperature for generation (lower = more deterministic)
    temperature: float = 0.0

    # Maximum tokens for hypothetical document
    max_tokens: int = 200

    # Whether to include original query in search
    include_original_query: bool = True

    # Prompt template for generating hypothetical documents
    prompt_template: str = """You are a clinical nutrition expert. Write a short, factual paragraph that would answer the following question about Total Parenteral Nutrition (TPN).

Question: {query}

Write a direct, informative answer as if it came from a clinical reference document. Include specific values, dosages, or recommendations if relevant. Keep it under 100 words.

Answer:"""


class HyDERetriever:
    """
    Hypothetical Document Embeddings (HyDE) retriever.

    HyDE works by:
    1. Taking a query (question)
    2. Using an LLM to generate a hypothetical document that would answer it
    3. Embedding the hypothetical document instead of the query
    4. Using that embedding for similarity search

    This helps because:
    - Questions and answers have different writing styles
    - The hypothetical answer is semantically closer to actual documents
    - Improves recall for complex or indirect questions

    Example:
        >>> hyde = HyDERetriever(llm_client, base_retriever)
        >>> results = hyde.retrieve("protein requirements preterm")

        # Internal flow:
        # Query: "protein requirements preterm"
        # Hypothetical: "Preterm infants require 3-4 g/kg/day of protein..."
        # Search uses hypothetical embedding -> better matches
    """

    def __init__(
        self,
        llm: LLMProtocol,
        base_retriever: Any,
        config: Optional[HyDEConfig] = None,
    ):
        """
        Initialize the HyDE retriever.

        Args:
            llm: LLM for generating hypothetical documents
            base_retriever: Underlying retriever (vector or hybrid)
            config: HyDE configuration
        """
        self.llm = llm
        self.base_retriever = base_retriever
        self.config = config or HyDEConfig()

    def generate_hypothetical(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.

        Args:
            query: User's question

        Returns:
            Hypothetical document text
        """
        prompt = self.config.prompt_template.format(query=query)

        try:
            hypothetical = self.llm.generate(prompt)
            logger.debug(f"HyDE generated: {hypothetical[:100]}...")
            return hypothetical.strip()
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            # Fall back to original query
            return query

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Any]:
        """
        Retrieve documents using HyDE.

        Args:
            query: Search query (question)
            top_k: Number of results to return

        Returns:
            List of retrieval results
        """
        # Generate hypothetical document
        hypothetical = self.generate_hypothetical(query)

        # Retrieve using hypothetical
        if self.config.include_original_query:
            # Combine original query and hypothetical
            combined_query = f"{query}\n\n{hypothetical}"
            results = self.base_retriever.retrieve(combined_query, top_k=top_k)
        else:
            results = self.base_retriever.retrieve(hypothetical, top_k=top_k)

        return results

    def retrieve_with_hypothetical(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[List[Any], str]:
        """
        Retrieve documents and return the generated hypothetical.

        Useful for debugging and understanding HyDE's behavior.

        Returns:
            Tuple of (results, hypothetical_document)
        """
        hypothetical = self.generate_hypothetical(query)

        if self.config.include_original_query:
            combined_query = f"{query}\n\n{hypothetical}"
            results = self.base_retriever.retrieve(combined_query, top_k=top_k)
        else:
            results = self.base_retriever.retrieve(hypothetical, top_k=top_k)

        return results, hypothetical


class MockLLM:
    """Mock LLM for testing HyDE."""

    def generate(self, prompt: str) -> str:
        """Generate a mock hypothetical document."""
        # Extract query from prompt
        if "Question:" in prompt:
            query = prompt.split("Question:")[-1].split("Answer:")[0].strip()
        else:
            query = prompt[:100]

        # Generate based on keywords
        if "protein" in query.lower():
            return "Protein requirements for preterm infants are typically 3-4 g/kg/day according to ASPEN guidelines. For term infants, the recommendation is 2.5-3 g/kg/day. Protein should be initiated early to prevent catabolism and support growth."
        elif "dextrose" in query.lower() or "glucose" in query.lower():
            return "Dextrose infusion for neonates should be initiated at 6-8 mg/kg/min and can be advanced to 10-14 mg/kg/min. Blood glucose should be monitored to prevent hyperglycemia."
        elif "lipid" in query.lower():
            return "Intravenous lipid emulsions provide essential fatty acids and concentrated calories. Start at 1-2 g/kg/day and advance to 3 g/kg/day. Monitor triglycerides weekly."
        else:
            return f"This clinical reference provides information about {query}. The recommended approach follows current ASPEN guidelines for pediatric and neonatal nutrition support."


def demo_hyde():
    """Demo function to test HyDE retrieval."""
    print("=" * 60)
    print("HyDE RETRIEVER DEMO")
    print("=" * 60)

    # Create mock components
    llm = MockLLM()
    config = HyDEConfig(include_original_query=True)

    # Test hypothetical generation
    queries = [
        "What is the protein requirement for preterm infants?",
        "How should dextrose be initiated in neonates?",
        "What are the guidelines for lipid emulsions?",
    ]

    print("\n--- Hypothetical Document Generation ---")
    for query in queries:
        hyde = HyDERetriever(llm, base_retriever=None, config=config)
        hypothetical = hyde.generate_hypothetical(query)
        print(f"\nQuery: {query}")
        print(f"Hypothetical: {hypothetical[:150]}...")


if __name__ == "__main__":
    demo_hyde()
