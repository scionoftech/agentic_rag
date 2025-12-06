"""
Evaluation Agent for validating and assessing response quality.
"""
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.agents.synthesizer import SynthesisResult
from src.agents.retriever import RetrievalResult
from config.config import config

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Structured result from evaluation."""
    overall_score: float = Field(description="Overall quality score (0-1)", ge=0, le=1)
    relevance_score: float = Field(description="How relevant is the answer to the query (0-1)", ge=0, le=1)
    completeness_score: float = Field(description="How complete is the answer (0-1)", ge=0, le=1)
    accuracy_score: float = Field(description="How accurate based on source documents (0-1)", ge=0, le=1)
    clarity_score: float = Field(description="How clear and well-structured (0-1)", ge=0, le=1)
    issues_found: List[str] = Field(default_factory=list, description="List of issues identified")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    approved: bool = Field(description="Whether the response is approved for delivery")
    needs_regeneration: bool = Field(description="Whether response should be regenerated")


class EvaluationAgent:
    """
    Agent responsible for evaluating the quality of generated responses.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        approval_threshold: float = 0.7
    ):
        """
        Initialize the Evaluation Agent.

        Args:
            model_name: Name of the LLM model (defaults to config)
            temperature: Temperature for generation (low for consistency)
            approval_threshold: Minimum score for approval (0-1)
        """
        self.model_name = model_name or config.openai.model
        self.temperature = temperature
        self.approval_threshold = approval_threshold

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=config.openai.api_key
        )

        self.parser = PydanticOutputParser(pydantic_object=EvaluationResult)

        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality assurance expert evaluating AI-generated answers.

Evaluate the answer based on these criteria:
1. **Relevance**: Does the answer address the user's query?
2. **Completeness**: Is the answer comprehensive and complete?
3. **Accuracy**: Is the answer accurate based on the source documents?
4. **Clarity**: Is the answer clear, well-structured, and easy to understand?

For each criterion, assign a score from 0 to 1 (where 1 is excellent).

Identify any issues such as:
- Hallucinations (information not in sources)
- Missing important information
- Contradictions
- Poor structure or clarity
- Insufficient citation of sources

Provide specific suggestions for improvement.

The answer is approved if the overall score is >= {approval_threshold}.

{format_instructions}"""),
            ("user", """Query: {query}

Source Documents:
{context}

Generated Answer:
{answer}

Please evaluate this answer.""")
        ])

        logger.info(f"Initialized EvaluationAgent with threshold={self.approval_threshold}")

    async def evaluate(
        self,
        query: str,
        synthesis_result: SynthesisResult,
        retrieval_result: RetrievalResult
    ) -> EvaluationResult:
        """
        Evaluate a synthesized answer.

        Args:
            query: The original query
            synthesis_result: Result from synthesis agent
            retrieval_result: Result from retrieval agent

        Returns:
            EvaluationResult with quality assessment
        """
        logger.info(f"Evaluating answer for query: '{query[:100]}...'")

        try:
            # Format context from retrieval
            context = self._format_context(retrieval_result.documents)

            chain = self.evaluation_prompt | self.llm | self.parser

            result = await chain.ainvoke({
                "query": query,
                "answer": synthesis_result.answer,
                "context": context,
                "approval_threshold": self.approval_threshold,
                "format_instructions": self.parser.get_format_instructions()
            })

            # Determine if regeneration is needed
            result.needs_regeneration = (
                not result.approved or
                result.overall_score < 0.5 or
                "hallucination" in str(result.issues_found).lower()
            )

            logger.info(
                f"Evaluation complete: score={result.overall_score:.2f}, "
                f"approved={result.approved}, needs_regen={result.needs_regeneration}"
            )

            return result

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Return conservative evaluation on error
            return EvaluationResult(
                overall_score=0.5,
                relevance_score=0.5,
                completeness_score=0.5,
                accuracy_score=0.5,
                clarity_score=0.5,
                issues_found=["Evaluation failed"],
                suggestions=["Manual review recommended"],
                approved=False,
                needs_regeneration=False
            )

    def evaluate_sync(
        self,
        query: str,
        synthesis_result: SynthesisResult,
        retrieval_result: RetrievalResult
    ) -> EvaluationResult:
        """
        Synchronous version of evaluate.

        Args:
            query: The original query
            synthesis_result: Result from synthesis agent
            retrieval_result: Result from retrieval agent

        Returns:
            EvaluationResult with quality assessment
        """
        logger.info(f"Evaluating answer (sync) for query: '{query[:100]}...'")

        try:
            context = self._format_context(retrieval_result.documents)

            chain = self.evaluation_prompt | self.llm | self.parser

            result = chain.invoke({
                "query": query,
                "answer": synthesis_result.answer,
                "context": context,
                "approval_threshold": self.approval_threshold,
                "format_instructions": self.parser.get_format_instructions()
            })

            result.needs_regeneration = (
                not result.approved or
                result.overall_score < 0.5 or
                "hallucination" in str(result.issues_found).lower()
            )

            logger.info(f"Evaluation complete: score={result.overall_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return EvaluationResult(
                overall_score=0.5,
                relevance_score=0.5,
                completeness_score=0.5,
                accuracy_score=0.5,
                clarity_score=0.5,
                issues_found=["Evaluation failed"],
                suggestions=["Manual review recommended"],
                approved=False,
                needs_regeneration=False
            )

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format documents for evaluation context.

        Args:
            documents: List of document dictionaries

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            context_parts.append(f"[Document {i}]\n{content[:500]}...")  # Truncate for evaluation

        return "\n\n".join(context_parts)

    async def quick_check(
        self,
        answer: str,
        sources: List[str]
    ) -> Dict[str, Any]:
        """
        Perform a quick quality check without full evaluation.

        Args:
            answer: The answer to check
            sources: List of sources cited

        Returns:
            Dictionary with quick check results
        """
        quick_prompt = ChatPromptTemplate.from_messages([
            ("system", """Perform a quick quality check on this answer.

Check for:
1. Length appropriateness (not too short, not too verbose)
2. Presence of source citations
3. Clear structure
4. Professional tone

Return a JSON with: {{"pass": boolean, "issues": [list of issues]}}"""),
            ("user", "Answer: {answer}\nSources cited: {sources}")
        ])

        try:
            chain = quick_prompt | self.llm
            result = await chain.ainvoke({
                "answer": answer,
                "sources": ", ".join(sources) if sources else "None"
            })

            # Simple heuristic checks
            has_citations = "[Source" in answer or "source" in answer.lower()
            appropriate_length = 50 <= len(answer.split()) <= 1000
            has_sources = len(sources) > 0

            return {
                "pass": has_citations and appropriate_length and has_sources,
                "issues": [
                    issue for issue, check in [
                        ("No source citations found", not has_citations),
                        ("Answer length inappropriate", not appropriate_length),
                        ("No sources provided", not has_sources)
                    ] if check
                ]
            }

        except Exception as e:
            logger.error(f"Error in quick check: {e}")
            return {"pass": False, "issues": ["Quick check failed"]}
