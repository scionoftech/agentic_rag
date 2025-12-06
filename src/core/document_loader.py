"""
Document loading and preprocessing for Agentic RAG pipeline.
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.config import config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles document loading, preprocessing, and chunking.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of text chunks (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
        """
        self.chunk_size = chunk_size or config.retrieval.chunk_size
        self.chunk_overlap = chunk_overlap or config.retrieval.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        logger.info(
            f"Initialized DocumentProcessor with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def load_document(self, file_path: Path) -> List[Document]:
        """
        Load a single document based on its file type.

        Args:
            file_path: Path to the document

        Returns:
            List of Document objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        suffix = file_path.suffix.lower()

        try:
            if suffix == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif suffix == '.txt':
                loader = TextLoader(str(file_path))
            elif suffix in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            else:
                logger.warning(f"Unsupported file type: {suffix}, attempting TextLoader")
                loader = TextLoader(str(file_path))

            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise

    def load_directory(
        self,
        directory_path: Path,
        glob_pattern: str = "**/*.*",
        show_progress: bool = True
    ) -> List[Document]:
        """
        Load all documents from a directory.

        Args:
            directory_path: Path to the directory
            glob_pattern: Pattern to match files
            show_progress: Whether to show progress

        Returns:
            List of Document objects
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        documents = []
        file_paths = list(directory_path.glob(glob_pattern))

        logger.info(f"Found {len(file_paths)} files in {directory_path}")

        for file_path in file_paths:
            if file_path.is_file():
                try:
                    docs = self.load_document(file_path)
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    continue

        logger.info(f"Successfully loaded {len(documents)} documents from directory")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def process_documents(
        self,
        source: Path,
        is_directory: bool = False
    ) -> List[Document]:
        """
        Load and process documents (load + chunk).

        Args:
            source: Path to file or directory
            is_directory: Whether source is a directory

        Returns:
            List of processed Document chunks
        """
        source = Path(source)

        if is_directory:
            documents = self.load_directory(source)
        else:
            documents = self.load_document(source)

        chunks = self.chunk_documents(documents)

        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "source": str(source),
                "chunk_size": len(chunk.page_content)
            })

        return chunks

    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """
        Extract and enrich document metadata.

        Args:
            document: Document object

        Returns:
            Dictionary of metadata
        """
        metadata = document.metadata.copy()
        metadata.update({
            "content_length": len(document.page_content),
            "word_count": len(document.page_content.split())
        })
        return metadata
