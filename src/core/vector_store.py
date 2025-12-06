"""
Vector store management for Agentic RAG pipeline.
"""
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

from config.config import config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector store operations including indexing and retrieval.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[Path] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the vector store manager.

        Args:
            collection_name: Name of the collection (defaults to config)
            persist_directory: Directory to persist the vector store (defaults to config)
            embedding_model: Name of the embedding model (defaults to config)
        """
        self.collection_name = collection_name or config.vector_store.collection_name
        self.persist_directory = str(persist_directory or config.vector_store.path)
        self.embedding_model_name = embedding_model or config.openai.embedding_model

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=config.openai.api_key
        )

        self.vector_store: Optional[VectorStore] = None

        logger.info(
            f"Initialized VectorStoreManager with collection={self.collection_name}, "
            f"persist_dir={self.persist_directory}"
        )

    def create_vector_store(self, documents: List[Document]) -> VectorStore:
        """
        Create a new vector store from documents.

        Args:
            documents: List of Document objects to index

        Returns:
            VectorStore instance
        """
        logger.info(f"Creating vector store with {len(documents)} documents")

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )

        logger.info(f"Vector store created successfully")
        return self.vector_store

    def load_vector_store(self) -> VectorStore:
        """
        Load an existing vector store from disk.

        Returns:
            VectorStore instance
        """
        logger.info(f"Loading vector store from {self.persist_directory}")

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        logger.info("Vector store loaded successfully")
        return self.vector_store

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to an existing vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")

        logger.info(f"Adding {len(documents)} documents to vector store")

        ids = self.vector_store.add_documents(documents)

        logger.info(f"Added {len(ids)} documents successfully")
        return ids

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.

        Args:
            query: Search query
            k: Number of results to return (defaults to config)
            filter: Metadata filter

        Returns:
            List of Document objects
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_vector_store first.")

        k = k or config.retrieval.top_k_documents

        logger.info(f"Performing similarity search for query: '{query[:50]}...' with k={k}")

        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )

        logger.info(f"Found {len(results)} results")
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.

        Args:
            query: Search query
            k: Number of results to return (defaults to config)
            filter: Metadata filter

        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_vector_store first.")

        k = k or config.retrieval.top_k_documents

        logger.info(f"Performing similarity search with scores for query: '{query[:50]}...'")

        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )

        # Filter by threshold
        threshold = config.retrieval.similarity_threshold
        filtered_results = [(doc, score) for doc, score in results if score >= threshold]

        logger.info(f"Found {len(filtered_results)} results above threshold {threshold}")
        return filtered_results

    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a retriever interface for the vector store.

        Args:
            search_kwargs: Additional search parameters

        Returns:
            Retriever instance
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_vector_store first.")

        search_kwargs = search_kwargs or {"k": config.retrieval.top_k_documents}

        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )

        logger.info("Created retriever instance")
        return retriever

    def delete_collection(self):
        """Delete the current collection."""
        if self.vector_store is not None:
            self.vector_store.delete_collection()
            logger.info(f"Deleted collection: {self.collection_name}")
            self.vector_store = None

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.

        Returns:
            Dictionary with collection statistics
        """
        if self.vector_store is None:
            return {"status": "not_initialized"}

        try:
            collection = self.vector_store._collection
            count = collection.count()

            return {
                "status": "initialized",
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"status": "error", "error": str(e)}
