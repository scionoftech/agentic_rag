"""
Synthesis Agent for generating responses from retrieved documents.
"""
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.retriever import RetrievalResult
from config.config import config

logger = logging.getLogger(__name__)


class SynthesisResult(BaseModel):
    """Structured result from synthesis."""
    answer: str = Field(description="The generated answer")
    sources_used: List[str] = Field(description="Sources cited in the answer")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    requires_clarification: bool = Field(
        default=False,
        description="Whether the query requires clarification"
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )


class SynthesisAgent:
    """
    Agent responsible for synthesizing answers from retrieved documents.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the Synthesis Agent.

        Args:
            model_name: Name of the LLM model (defaults to config)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response (defaults to config)
        """
        self.model_name = model_name or config.openai.model
        self.temperature = temperature
        self.max_tokens = max_tokens or config.openai.max_tokens

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=config.openai.api_key
        )

        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI assistant that synthesizes information from multiple sources to answer user questions.

Your responsibilities:
1. Provide accurate, comprehensive answers based ONLY on the provided context
2. Cite sources using [Source N] notation where N is the source number
3. If the context doesn't contain enough information, clearly state this
4. Maintain a helpful and professional tone
5. Structure your answer clearly with appropriate formatting
6. If the query is ambiguous, note what clarification is needed

IMPORTANT:
- Never make up information not present in the context
- Always cite your sources
- Be honest about limitations in the available information
"""),
            ("user", """Context Documents:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above.""")
        ])

        logger.info(f"Initialized SynthesisAgent with model={self.model_name}")

    async def synthesize(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        include_metadata: bool = True
    ) -> SynthesisResult:
        """
        Synthesize an answer from retrieved documents.

        Args:
            query: The user's query
            retrieval_result: Result from retrieval agent
            include_metadata: Whether to include document metadata

        Returns:
            SynthesisResult with generated answer
        """
        logger.info(f"Synthesizing answer for query: '{query[:100]}...'")

        if not retrieval_result.documents:
            logger.warning("No documents provided for synthesis")
            return SynthesisResult(
                answer="I don't have enough information to answer this question based on the available documents.",
                sources_used=[],
                confidence="low",
                requires_clarification=True,
                follow_up_questions=[]
            )

        # Format context
        context = self._format_context(retrieval_result.documents, include_metadata)

        try:
            # Generate answer
            chain = self.synthesis_prompt | self.llm | StrOutputParser()

            answer = await chain.ainvoke({
                "query": query,
                "context": context
            })

            # Extract metadata
            sources_used = [
                doc.get("metadata", {}).get("source", f"Document {i+1}")
                for i, doc in enumerate(retrieval_result.documents)
            ]

            # Determine confidence based on retrieval scores
            avg_score = sum(
                doc.get("relevance_score", 0) for doc in retrieval_result.documents
            ) / len(retrieval_result.documents)

            if avg_score > 0.8:
                confidence = "high"
            elif avg_score > 0.6:
                confidence = "medium"
            else:
                confidence = "low"

            # Generate follow-up questions
            follow_ups = await self._generate_follow_ups(query, answer)

            result = SynthesisResult(
                answer=answer,
                sources_used=sources_used,
                confidence=confidence,
                requires_clarification=False,
                follow_up_questions=follow_ups
            )

            logger.info(f"Answer synthesized with confidence={confidence}")
            return result

        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            return SynthesisResult(
                answer="I encountered an error while generating the answer. Please try again.",
                sources_used=[],
                confidence="low",
                requires_clarification=True,
                follow_up_questions=[]
            )

    def synthesize_sync(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        include_metadata: bool = True
    ) -> SynthesisResult:
        """
        Synchronous version of synthesize.

        Args:
            query: The user's query
            retrieval_result: Result from retrieval agent
            include_metadata: Whether to include document metadata

        Returns:
            SynthesisResult with generated answer
        """
        logger.info(f"Synthesizing answer (sync) for query: '{query[:100]}...'")

        if not retrieval_result.documents:
            return SynthesisResult(
                answer="I don't have enough information to answer this question based on the available documents.",
                sources_used=[],
                confidence="low",
                requires_clarification=True,
                follow_up_questions=[]
            )

        context = self._format_context(retrieval_result.documents, include_metadata)

        try:
            chain = self.synthesis_prompt | self.llm | StrOutputParser()

            answer = chain.invoke({
                "query": query,
                "context": context
            })

            sources_used = [
                doc.get("metadata", {}).get("source", f"Document {i+1}")
                for i, doc in enumerate(retrieval_result.documents)
            ]

            avg_score = sum(
                doc.get("relevance_score", 0) for doc in retrieval_result.documents
            ) / len(retrieval_result.documents)

            confidence = "high" if avg_score > 0.8 else "medium" if avg_score > 0.6 else "low"

            result = SynthesisResult(
                answer=answer,
                sources_used=sources_used,
                confidence=confidence,
                requires_clarification=False,
                follow_up_questions=[]
            )

            logger.info(f"Answer synthesized with confidence={confidence}")
            return result

        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            return SynthesisResult(
                answer="I encountered an error while generating the answer. Please try again.",
                sources_used=[],
                confidence="low",
                requires_clarification=True,
                follow_up_questions=[]
            )

    def _format_context(
        self,
        documents: List[Dict[str, Any]],
        include_metadata: bool
    ) -> str:
        """
        Format retrieved documents as context.

        Args:
            documents: List of document dictionaries
            include_metadata: Whether to include metadata

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            score = doc.get("relevance_score", 0)

            part = f"[Source {i}] (Relevance: {score:.2f})\n{content}"

            if include_metadata and "metadata" in doc:
                metadata = doc["metadata"]
                if metadata:
                    meta_str = ", ".join(f"{k}: {v}" for k, v in metadata.items())
                    part += f"\n(Metadata: {meta_str})"

            context_parts.append(part)

        return "\n\n---\n\n".join(context_parts)

    async def _generate_follow_ups(
        self,
        query: str,
        answer: str,
        max_questions: int = 3
    ) -> List[str]:
        """
        Generate follow-up questions based on the query and answer.

        Args:
            query: Original query
            answer: Generated answer
            max_questions: Maximum number of follow-up questions

        Returns:
            List of follow-up questions
        """
        follow_up_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Generate {max_questions} relevant follow-up questions based on the user's original question and the answer provided.
These should help the user explore the topic deeper or clarify aspects not fully covered.

Return only the questions, one per line, without numbering."""),
            ("user", "Original Question: {query}\n\nAnswer: {answer}")
        ])

        try:
            chain = follow_up_prompt | self.llm | StrOutputParser()
            result = await chain.ainvoke({"query": query, "answer": answer})

            questions = [q.strip() for q in result.split('\n') if q.strip()]
            return questions[:max_questions]

        except Exception as e:
            logger.error(f"Error generating follow-ups: {e}")
            return []
