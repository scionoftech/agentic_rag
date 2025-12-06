"""
Main orchestrator for the Agentic RAG pipeline.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid

from src.core.document_loader import DocumentProcessor
from src.core.vector_store import VectorStoreManager
from src.core.workflow import AgenticRAGWorkflow
from config.config import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.logging.file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AgenticRAGOrchestrator:
    """
    Main orchestrator for the Agentic RAG pipeline.

    This class provides a high-level interface for:
    - Indexing documents
    - Processing queries
    - Managing the entire RAG workflow
    """

    def __init__(
        self,
        vector_store_path: Optional[Path] = None,
        enable_checkpointing: bool = True
    ):
        """
        Initialize the orchestrator.

        Args:
            vector_store_path: Path to vector store (defaults to config)
            enable_checkpointing: Whether to enable durable execution
        """
        logger.info("Initializing AgenticRAGOrchestrator")

        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager(
            persist_directory=vector_store_path
        )
        self.workflow = AgenticRAGWorkflow(
            enable_checkpointing=enable_checkpointing
        )

        self._initialized = False

        logger.info("AgenticRAGOrchestrator initialized")

    def index_documents(
        self,
        source_path: Path,
        is_directory: bool = True,
        recreate_index: bool = False
    ) -> Dict[str, Any]:
        """
        Index documents into the vector store.

        Args:
            source_path: Path to documents or directory
            is_directory: Whether source is a directory
            recreate_index: Whether to recreate the index from scratch

        Returns:
            Dictionary with indexing results
        """
        logger.info(f"Starting document indexing from {source_path}")

        try:
            # Process documents
            chunks = self.document_processor.process_documents(
                source=source_path,
                is_directory=is_directory
            )

            if not chunks:
                logger.warning("No documents found to index")
                return {
                    "status": "warning",
                    "message": "No documents found",
                    "documents_indexed": 0
                }

            # Create or update vector store
            if recreate_index:
                logger.info("Recreating vector store index")
                self.vector_store_manager.delete_collection()
                self.vector_store_manager.create_vector_store(chunks)
            else:
                # Try to load existing, or create new
                try:
                    self.vector_store_manager.load_vector_store()
                    self.vector_store_manager.add_documents(chunks)
                    logger.info("Added documents to existing index")
                except Exception as e:
                    logger.info(f"Creating new vector store: {e}")
                    self.vector_store_manager.create_vector_store(chunks)

            self._initialized = True

            # Get collection stats
            stats = self.vector_store_manager.get_collection_stats()

            result = {
                "status": "success",
                "message": "Documents indexed successfully",
                "documents_indexed": len(chunks),
                "collection_stats": stats
            }

            logger.info(f"Indexing complete: {len(chunks)} chunks indexed")
            return result

        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            return {
                "status": "error",
                "message": f"Indexing failed: {str(e)}",
                "documents_indexed": 0
            }

    async def query(
        self,
        query: str,
        thread_id: Optional[str] = None,
        max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the Agentic RAG pipeline.

        Args:
            query: User query
            thread_id: Optional thread ID for conversation tracking
            max_retries: Maximum number of retries for failed operations

        Returns:
            Dictionary with answer and metadata
        """
        if not self._initialized:
            logger.info("Workflow not initialized, loading vector store")
            try:
                self.vector_store_manager.load_vector_store()
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                return {
                    "status": "error",
                    "answer": "Vector store not initialized. Please index documents first.",
                    "sources": [],
                    "error": str(e)
                }

        logger.info(f"Processing query: '{query[:100]}...'")

        # Generate thread ID if not provided
        if not thread_id:
            thread_id = str(uuid.uuid4())

        try:
            # Run the workflow
            result = await self.workflow.run(
                query=query,
                thread_id=thread_id,
                max_retries=max_retries
            )

            result["status"] = "success" if not result.get("error") else "error"
            result["thread_id"] = thread_id

            logger.info("Query processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "status": "error",
                "answer": "An error occurred while processing your query.",
                "sources": [],
                "error": str(e),
                "thread_id": thread_id
            }

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current vector store collection.

        Returns:
            Dictionary with collection information
        """
        return self.vector_store_manager.get_collection_stats()

    def reset_index(self):
        """Delete the current vector store index."""
        logger.info("Resetting vector store index")
        self.vector_store_manager.delete_collection()
        self._initialized = False
        logger.info("Index reset complete")

    async def batch_query(
        self,
        queries: List[str],
        thread_id_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.

        Args:
            queries: List of queries
            thread_id_prefix: Optional prefix for thread IDs

        Returns:
            List of results
        """
        logger.info(f"Processing {len(queries)} queries in batch")

        results = []
        for i, query in enumerate(queries):
            thread_id = f"{thread_id_prefix or 'batch'}_{i}"
            result = await self.query(query, thread_id=thread_id)
            results.append(result)

        logger.info(f"Batch processing complete: {len(results)} queries processed")
        return results

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of all components.

        Returns:
            Dictionary with health status
        """
        health = {
            "status": "healthy",
            "components": {}
        }

        # Check vector store
        try:
            stats = self.vector_store_manager.get_collection_stats()
            health["components"]["vector_store"] = {
                "status": "ok",
                "details": stats
            }
        except Exception as e:
            health["components"]["vector_store"] = {
                "status": "error",
                "error": str(e)
            }
            health["status"] = "degraded"

        # Check workflow
        try:
            if self.workflow.app:
                health["components"]["workflow"] = {"status": "ok"}
            else:
                health["components"]["workflow"] = {"status": "not_initialized"}
        except Exception as e:
            health["components"]["workflow"] = {
                "status": "error",
                "error": str(e)
            }
            health["status"] = "degraded"

        return health


# Convenience function for quick setup
def create_orchestrator(
    vector_store_path: Optional[Path] = None,
    enable_checkpointing: bool = True
) -> AgenticRAGOrchestrator:
    """
    Create and return an orchestrator instance.

    Args:
        vector_store_path: Path to vector store
        enable_checkpointing: Whether to enable durable execution

    Returns:
        AgenticRAGOrchestrator instance
    """
    return AgenticRAGOrchestrator(
        vector_store_path=vector_store_path,
        enable_checkpointing=enable_checkpointing
    )
