"""
Configuration management for Agentic RAG pipeline.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_STORE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    temperature: float = 0.7
    max_tokens: int = 2000


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    type: str = Field(default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "chroma"))
    path: Path = Field(default_factory=lambda: Path(os.getenv("VECTOR_STORE_PATH", str(VECTOR_STORE_DIR))))
    collection_name: str = Field(default_factory=lambda: os.getenv("COLLECTION_NAME", "agentic_rag_collection"))


class LangGraphConfig(BaseModel):
    """LangGraph configuration."""
    enable_durable_execution: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_DURABLE_EXECUTION", "true").lower() == "true"
    )
    checkpoint_backend: str = Field(default_factory=lambda: os.getenv("CHECKPOINT_BACKEND", "sqlite"))
    checkpoint_path: Path = Field(
        default_factory=lambda: Path(os.getenv("CHECKPOINT_PATH", str(DATA_DIR / "checkpoints.db")))
    )


class AgentConfig(BaseModel):
    """Agent configuration."""
    max_retries: int = Field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("TIMEOUT_SECONDS", "60")))
    enable_streaming: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    )


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k_documents: int = Field(default_factory=lambda: int(os.getenv("TOP_K_DOCUMENTS", "5")))
    similarity_threshold: float = Field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.7")))
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    file: Path = Field(default_factory=lambda: Path(os.getenv("LOG_FILE", str(LOGS_DIR / "agentic_rag.log"))))


class Config(BaseModel):
    """Main configuration class."""
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    langgraph: LangGraphConfig = Field(default_factory=LangGraphConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# Global config instance
config = Config()
