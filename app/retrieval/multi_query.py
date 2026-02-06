"""
Multi-Query Retriever for improved recall.

Generates multiple query variations to capture different phrasings
of the same information need, then combines results.

This helps with:
- Vocabulary mismatch (user says "dose", document says "requirement")
- Query ambiguity (clarifies intent through variations)
- Improved recall for complex queries

Example:
    >>> multi_query = MultiQueryRetriever(llm, retriever)
    >>> results = multi_query.retrieve("protein for neonates")
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Any, Set

logger = logging.getLogger(__name__)


class LLMProtocol(Protocol):
    """Protocol for LLM implementations."""

    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        ...


@dataclass
class MultiQueryConfig:
    """Configuration for Multi-Query retrieval."""

    # Number of query variations to generate
    num_queries: int = 3

    # Whether to include original query
    include_original: bool = True

    # Temperature for generation
    temperature: float = 0.3

    # Prompt template for generating query variations
    prompt_template: str = """You are an expert at generating search queries for clinical documents about Total Parenteral Nutrition (TPN).

Given the following question, generate {num_queries} alternative search queries that would help find relevant information. Each query should:
1. Use different clinical terminology
2. Focus on different aspects of the question
3. Be concise (under 15 words)

Original question: {query}

Generate {num_queries} alternative queries, one per line:"""


class MultiQueryRetriever:
    """
    Multi-Query retriever for improved recall.

    Works by:
    1. Taking the original query
    2. Using an LLM to generate alternative phrasings
    3. Running all queries through the base retriever
    4. Deduplicating and combining results

    Benefits:
    - Handles vocabulary mismatch
    - Improves recall without losing precision
    - Captures different semantic aspects of the query

    Example:
        >>> multi = MultiQueryRetriever(llm, base_retriever)
        >>> results = multi.retrieve("protein dose")

        # Generated queries might include:
        # - "amino acid requirements for TPN"
        # - "protein administration parenteral nutrition"
        # - "recommended protein intake pediatric PN"
    """

    def __init__(
        self,
        llm: LLMProtocol,
        base_retriever: Any,
        config: Optional[MultiQueryConfig] = None,
    ):
        """
        Initialize the Multi-Query retriever.

        Args:
            llm: LLM for generating query variations
            base_retriever: Underlying retriever
            config: Multi-Query configuration
        """
        self.llm = llm
        self.base_retriever = base_retriever
        self.config = config or MultiQueryConfig()

    def generate_queries(self, query: str) -> List[str]:
        """
        Generate alternative query variations.

        Args:
            query: Original user query

        Returns:
            List of query variations (including original if configured)
        """
        prompt = self.config.prompt_template.format(
            query=query,
            num_queries=self.config.num_queries,
        )

        try:
            response = self.llm.generate(prompt)
            # Parse generated queries (one per line)
            generated = self._parse_queries(response)
            logger.debug(f"Generated {len(generated)} query variations")
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            generated = []

        # Combine with original
        queries = []
        if self.config.include_original:
            queries.append(query)
        queries.extend(generated[:self.config.num_queries])

        return queries

    def _parse_queries(self, response: str) -> List[str]:
        """Parse LLM response into individual queries."""
        lines = response.strip().split("\n")
        queries = []

        for line in lines:
            # Clean up each line
            line = line.strip()

            # Remove numbering (1., 2., -, *, etc.)
            line = re.sub(r'^[\d\.\-\*\)]+\s*', '', line)

            # Remove quotes
            line = line.strip('"\'')

            # Skip empty or too short
            if len(line) > 5:
                queries.append(line)

        return queries

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Any]:
        """
        Retrieve documents using multiple query variations.

        Args:
            query: Original search query
            top_k: Number of results to return

        Returns:
            List of deduplicated retrieval results
        """
        # Generate query variations
        queries = self.generate_queries(query)
        logger.info(f"Retrieving with {len(queries)} queries")

        # Retrieve for each query
        all_results = []
        seen_content: Set[str] = set()

        for q in queries:
            try:
                results = self.base_retriever.retrieve(q, top_k=top_k)

                # Deduplicate based on content
                for result in results:
                    content_hash = hashlib.md5((result.content[:500] if hasattr(result, 'content') else str(result)[:500]).encode()).hexdigest()

                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_results.append(result)

            except Exception as e:
                logger.error(f"Retrieval failed for query '{q}': {e}")

        # Sort by score and return top-k
        if all_results and hasattr(all_results[0], 'score'):
            all_results.sort(key=lambda r: r.score, reverse=True)

        return all_results[:top_k]

    def retrieve_with_queries(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[List[Any], List[str]]:
        """
        Retrieve documents and return the generated queries.

        Useful for debugging and understanding query expansion.

        Returns:
            Tuple of (results, generated_queries)
        """
        queries = self.generate_queries(query)
        results = self.retrieve(query, top_k=top_k)
        return results, queries


class MockLLM:
    """Mock LLM for testing Multi-Query retrieval."""

    def generate(self, prompt: str) -> str:
        """Generate mock query variations."""
        # Extract original query from prompt
        if "Original question:" in prompt:
            query = prompt.split("Original question:")[-1].split("\n")[0].strip()
        else:
            query = "TPN question"

        # Generate variations based on keywords
        variations = []

        if "protein" in query.lower():
            variations = [
                "amino acid requirements parenteral nutrition pediatric",
                "protein dosing TPN neonates ASPEN guidelines",
                "recommended protein intake preterm infants PN",
            ]
        elif "dextrose" in query.lower() or "glucose" in query.lower():
            variations = [
                "glucose infusion rate neonatal TPN",
                "dextrose concentration parenteral nutrition",
                "carbohydrate requirements preterm infants",
            ]
        elif "lipid" in query.lower():
            variations = [
                "intravenous lipid emulsion dosing neonates",
                "fat requirements TPN pediatric",
                "essential fatty acids parenteral nutrition",
            ]
        else:
            variations = [
                f"clinical guidelines {query}",
                f"ASPEN recommendations {query}",
                f"pediatric TPN {query}",
            ]

        return "\n".join(f"{i+1}. {v}" for i, v in enumerate(variations))


def demo_multi_query():
    """Demo function to test Multi-Query retrieval."""
    print("=" * 60)
    print("MULTI-QUERY RETRIEVER DEMO")
    print("=" * 60)

    # Create mock components
    llm = MockLLM()
    config = MultiQueryConfig(num_queries=3, include_original=True)

    # Test query generation
    queries = [
        "protein requirements for preterm infants",
        "how to initiate dextrose in TPN",
        "lipid dosing guidelines",
    ]

    print("\n--- Query Expansion ---")
    for original in queries:
        multi = MultiQueryRetriever(llm, base_retriever=None, config=config)
        variations = multi.generate_queries(original)
        print(f"\nOriginal: {original}")
        print("Variations:")
        for i, v in enumerate(variations):
            print(f"  {i+1}. {v}")


if __name__ == "__main__":
    demo_multi_query()
