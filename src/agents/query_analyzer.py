"""
Query Analyzer Agent for understanding and processing user queries.
"""
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from config.config import config

logger = logging.getLogger(__name__)


class QueryAnalysis(BaseModel):
    """Structured output for query analysis."""
    intent: str = Field(description="The primary intent of the query (e.g., factual, exploratory, analytical)")
    entities: List[str] = Field(description="Key entities or topics mentioned in the query")
    reformulated_query: str = Field(description="An optimized version of the query for retrieval")
    query_type: str = Field(description="Type of query: simple, complex, multi-hop, or conversational")
    suggested_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Suggested metadata filters for retrieval"
    )
    requires_multi_step: bool = Field(
        description="Whether the query requires multi-step reasoning"
    )


class QueryAnalyzerAgent:
    """
    Agent responsible for analyzing user queries and optimizing them for retrieval.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0
    ):
        """
        Initialize the Query Analyzer Agent.

        Args:
            model_name: Name of the LLM model (defaults to config)
            temperature: Temperature for generation
        """
        self.model_name = model_name or config.openai.model
        self.temperature = temperature

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=config.openai.api_key
        )

        self.parser = PydanticOutputParser(pydantic_object=QueryAnalysis)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query analysis expert. Your role is to analyze user queries and optimize them for document retrieval.

Your tasks:
1. Identify the primary intent of the query
2. Extract key entities and topics
3. Reformulate the query to be more specific and retrieval-friendly
4. Classify the query type
5. Determine if multi-step reasoning is needed
6. Suggest any metadata filters that might help narrow the search

Be concise and focused on improving retrieval quality.

{format_instructions}"""),
            ("user", "Analyze this query: {query}")
        ])

        logger.info(f"Initialized QueryAnalyzerAgent with model={self.model_name}")

    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a user query.

        Args:
            query: The user's query string

        Returns:
            QueryAnalysis object with structured analysis
        """
        logger.info(f"Analyzing query: '{query[:100]}...'")

        try:
            chain = self.prompt | self.llm | self.parser

            result = await chain.ainvoke({
                "query": query,
                "format_instructions": self.parser.get_format_instructions()
            })

            logger.info(f"Query analyzed: intent={result.intent}, type={result.query_type}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Fallback to simple analysis
            return QueryAnalysis(
                intent="factual",
                entities=[],
                reformulated_query=query,
                query_type="simple",
                requires_multi_step=False
            )

    def analyze_query_sync(self, query: str) -> QueryAnalysis:
        """
        Synchronous version of analyze_query.

        Args:
            query: The user's query string

        Returns:
            QueryAnalysis object with structured analysis
        """
        logger.info(f"Analyzing query (sync): '{query[:100]}...'")

        try:
            chain = self.prompt | self.llm | self.parser

            result = chain.invoke({
                "query": query,
                "format_instructions": self.parser.get_format_instructions()
            })

            logger.info(f"Query analyzed: intent={result.intent}, type={result.query_type}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Fallback to simple analysis
            return QueryAnalysis(
                intent="factual",
                entities=[],
                reformulated_query=query,
                query_type="simple",
                requires_multi_step=False
            )

    async def expand_query(self, query: str) -> List[str]:
        """
        Generate multiple query variations for better retrieval.

        Args:
            query: Original query

        Returns:
            List of query variations
        """
        expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate 3 different variations of the given query that maintain the same intent but use different wording.
This helps improve retrieval by matching different document phrasings.

Return only the variations, one per line, without numbering or additional text."""),
            ("user", "{query}")
        ])

        try:
            chain = expansion_prompt | self.llm
            result = await chain.ainvoke({"query": query})

            variations = [line.strip() for line in result.content.split('\n') if line.strip()]
            variations = [query] + variations  # Include original

            logger.info(f"Generated {len(variations)} query variations")
            return variations

        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]


# For backwards compatibility
from typing import Optional
