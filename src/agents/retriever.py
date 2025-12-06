"""
Retrieval Agent for fetching relevant documents.
"""
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_core.documents import Document

from src.core.vector_store import VectorStoreManager
from src.agents.query_analyzer import QueryAnalysis
from config.config import config

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """Structured result from retrieval."""
    documents: List[Dict[str, Any]] = Field(description="Retrieved documents with metadata")
    total_retrieved: int = Field(description="Total number of documents retrieved")
    query_used: str = Field(description="The actual query used for retrieval")
    retrieval_strategy: str = Field(description="Strategy used for retrieval")


class RetrievalAgent:
    """
    Agent responsible for retrieving relevant documents from the vector store.
    """

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        top_k: Optional[int] = None
    ):
        """
        Initialize the Retrieval Agent.

        Args:
            vector_store_manager: VectorStoreManager instance (creates new if None)
            top_k: Number of documents to retrieve (defaults to config)
        """
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.top_k = top_k or config.retrieval.top_k_documents

        # Try to load existing vector store
        try:
            self.vector_store_manager.load_vector_store()
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}. Will need to create one.")

        logger.info(f"Initialized RetrievalAgent with top_k={self.top_k}")

    async def retrieve(
        self,
        query: str,
        query_analysis: Optional[QueryAnalysis] = None,
        k: Optional[int] = None,
        use_reranking: bool = False
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query
            query_analysis: Optional query analysis for enhanced retrieval
            k: Number of documents to retrieve (overrides default)
            use_reranking: Whether to rerank results

        Returns:
            RetrievalResult with documents and metadata
        """
        k = k or self.top_k

        # Use reformulated query if analysis is provided
        search_query = query
        strategy = "basic_similarity"

        if query_analysis:
            search_query = query_analysis.reformulated_query
            strategy = "analyzed_query"
            logger.info(f"Using reformulated query: '{search_query[:100]}...'")

        logger.info(f"Retrieving documents with strategy={strategy}, k={k}")

        try:
            # Perform similarity search with scores
            results_with_scores = self.vector_store_manager.similarity_search_with_score(
                query=search_query,
                k=k,
                filter=query_analysis.suggested_filters if query_analysis else None
            )

            # Convert to structured format
            documents = []
            for doc, score in results_with_scores:
                doc_dict = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score)
                }
                documents.append(doc_dict)

            # Optional reranking
            if use_reranking and len(documents) > 1:
                documents = self._rerank_documents(documents, query)
                strategy = f"{strategy}_with_reranking"

            result = RetrievalResult(
                documents=documents,
                total_retrieved=len(documents),
                query_used=search_query,
                retrieval_strategy=strategy
            )

            logger.info(f"Retrieved {len(documents)} documents")
            return result

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            # Return empty result
            return RetrievalResult(
                documents=[],
                total_retrieved=0,
                query_used=search_query,
                retrieval_strategy="error"
            )

    def retrieve_sync(
        self,
        query: str,
        query_analysis: Optional[QueryAnalysis] = None,
        k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Synchronous version of retrieve.

        Args:
            query: The search query
            query_analysis: Optional query analysis
            k: Number of documents to retrieve

        Returns:
            RetrievalResult with documents and metadata
        """
        k = k or self.top_k
        search_query = query
        strategy = "basic_similarity"

        if query_analysis:
            search_query = query_analysis.reformulated_query
            strategy = "analyzed_query"

        logger.info(f"Retrieving documents (sync) with k={k}")

        try:
            results_with_scores = self.vector_store_manager.similarity_search_with_score(
                query=search_query,
                k=k,
                filter=query_analysis.suggested_filters if query_analysis else None
            )

            documents = []
            for doc, score in results_with_scores:
                doc_dict = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score)
                }
                documents.append(doc_dict)

            result = RetrievalResult(
                documents=documents,
                total_retrieved=len(documents),
                query_used=search_query,
                retrieval_strategy=strategy
            )

            logger.info(f"Retrieved {len(documents)} documents")
            return result

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return RetrievalResult(
                documents=[],
                total_retrieved=0,
                query_used=search_query,
                retrieval_strategy="error"
            )

    def _rerank_documents(
        self,
        documents: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance.

        Args:
            documents: List of document dictionaries
            query: Original query

        Returns:
            Reranked list of documents
        """
        # Simple reranking based on query term overlap
        # In production, consider using a cross-encoder or dedicated reranker
        query_terms = set(query.lower().split())

        def relevance_score(doc: Dict[str, Any]) -> float:
            content = doc["content"].lower()
            term_overlap = sum(1 for term in query_terms if term in content)
            # Combine with original similarity score
            original_score = doc["relevance_score"]
            return original_score * 0.7 + (term_overlap / len(query_terms)) * 0.3

        reranked = sorted(documents, key=relevance_score, reverse=True)
        logger.info("Documents reranked")
        return reranked

    async def retrieve_multi_query(
        self,
        queries: List[str],
        k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve documents using multiple query variations.

        Args:
            queries: List of query variations
            k: Total number of unique documents to retrieve

        Returns:
            RetrievalResult with deduplicated documents
        """
        k = k or self.top_k
        all_documents = {}  # Use dict to deduplicate by content

        logger.info(f"Retrieving with {len(queries)} query variations")

        for query in queries:
            result = await self.retrieve(query, k=k)
            for doc in result.documents:
                # Use content as key for deduplication
                content_key = doc["content"][:100]  # First 100 chars as key
                if content_key not in all_documents:
                    all_documents[content_key] = doc
                else:
                    # Keep the one with higher score
                    if doc["relevance_score"] > all_documents[content_key]["relevance_score"]:
                        all_documents[content_key] = doc

        # Sort by relevance and limit to k
        documents = sorted(
            all_documents.values(),
            key=lambda x: x["relevance_score"],
            reverse=True
        )[:k]

        return RetrievalResult(
            documents=documents,
            total_retrieved=len(documents),
            query_used=f"multi_query_{len(queries)}_variations",
            retrieval_strategy="multi_query"
        )
