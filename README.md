# Agentic RAG Pipeline

A production-ready **End-to-End Agentic RAG (Retrieval-Augmented Generation)** system built with **Multi-Agent Orchestration** using **LangGraph** and **durable execution** capabilities.

## Overview

This project demonstrates a sophisticated RAG pipeline that uses multiple specialized AI agents working together to provide accurate, context-aware answers to user queries. The system leverages LangGraph for workflow orchestration with checkpointing support for durable execution.

## Architecture

### Multi-Agent System

The pipeline consists of four specialized agents:

1. **Query Analyzer Agent**: Understands user intent, extracts entities, and reformulates queries for optimal retrieval
2. **Retrieval Agent**: Fetches relevant documents from the vector store using similarity search with optional reranking
3. **Synthesis Agent**: Generates comprehensive answers based on retrieved documents with proper citations
4. **Evaluation Agent**: Validates answer quality, checks for hallucinations, and determines if regeneration is needed

### Workflow

```
User Query → Query Analyzer → Retrieval → Synthesis → Evaluation → Final Answer
                                   ↑           ↑            ↓
                                   └───────────┴────────────┘
                                   (retry loop if needed)
```

The LangGraph workflow supports:
- **Durable Execution**: State persistence with SQLite checkpointing
- **Retry Logic**: Automatic retry on failures with configurable max attempts
- **Conditional Routing**: Smart decision-making between workflow steps
- **State Management**: Complete tracking of all workflow states

## Features

- **Multi-Agent Orchestration**: Specialized agents for different tasks
- **Durable Execution**: Fault-tolerant with state checkpointing
- **Vector Store Integration**: ChromaDB for efficient similarity search
- **Document Processing**: Support for PDF, TXT, DOCX, and more
- **Quality Evaluation**: Built-in answer validation and scoring
- **Interactive CLI**: Rich terminal interface for user interaction
- **Batch Processing**: Handle multiple queries efficiently
- **Health Monitoring**: System health checks and diagnostics
- **Conversation Tracking**: Thread-based conversation management

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd agentic_rag
```

2. **Create a virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Quick Start

Use the quick start script:
```bash
./examples/quick_start.sh
```

Or manually:
```bash
# Index sample documents
python3 main.py index data/raw

# Start interactive query mode
python3 main.py query

# Or ask a single question
python3 main.py query "What is machine learning?"
```

## Usage

### Command Line Interface

The CLI provides several commands:

#### 1. Index Documents

Index documents from a file or directory:
```bash
python3 main.py index <path> [--recreate]
```

Examples:
```bash
# Index a directory
python3 main.py index ./data/raw

# Recreate index from scratch
python3 main.py index ./data/raw --recreate

# Index a single file
python3 main.py index ./document.pdf
```

#### 2. Query the System

Interactive mode:
```bash
python3 main.py query
```

Single query:
```bash
python3 main.py query "What are the types of machine learning?"
```

#### 3. System Information

Show collection statistics:
```bash
python3 main.py info
```

Perform health check:
```bash
python3 main.py health
```

#### 4. Reset Index

Delete the vector store index:
```bash
python3 main.py reset
```

### Programmatic Usage

```python
from src.core.orchestrator import create_orchestrator
from pathlib import Path
import asyncio

async def main():
    # Create orchestrator
    orchestrator = create_orchestrator(enable_checkpointing=True)

    # Index documents
    result = orchestrator.index_documents(
        source_path=Path("data/raw"),
        is_directory=True
    )
    print(f"Indexed {result['documents_indexed']} chunks")

    # Query the system
    query_result = await orchestrator.query(
        query="What is supervised learning?",
        thread_id="conversation_001"
    )

    print(f"Answer: {query_result['answer']}")
    print(f"Sources: {query_result['sources']}")
    print(f"Confidence: {query_result['evaluation'].confidence}")

asyncio.run(main())
```

### Batch Processing

Process multiple queries:
```python
queries = [
    "What is machine learning?",
    "Explain supervised learning",
    "What are the applications?"
]

results = await orchestrator.batch_query(queries, thread_id_prefix="batch")

for i, result in enumerate(results):
    print(f"Q{i+1}: {queries[i]}")
    print(f"A{i+1}: {result['answer']}\n")
```

## Configuration

Configuration is managed through environment variables and the `config/config.py` file.

### Environment Variables (.env)

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Vector Store
VECTOR_STORE_TYPE=chroma
VECTOR_STORE_PATH=./data/vector_store
COLLECTION_NAME=agentic_rag_collection

# LangGraph
ENABLE_DURABLE_EXECUTION=true
CHECKPOINT_BACKEND=sqlite
CHECKPOINT_PATH=./data/checkpoints.db

# Agent Settings
MAX_RETRIES=3
TIMEOUT_SECONDS=60

# Retrieval
TOP_K_DOCUMENTS=5
SIMILARITY_THRESHOLD=0.7
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Project Structure

```
agentic_rag/
├── config/
│   └── config.py              # Configuration management
├── src/
│   ├── agents/
│   │   ├── query_analyzer.py  # Query analysis agent
│   │   ├── retriever.py       # Document retrieval agent
│   │   ├── synthesizer.py     # Answer synthesis agent
│   │   └── evaluator.py       # Quality evaluation agent
│   ├── core/
│   │   ├── document_loader.py # Document processing
│   │   ├── vector_store.py    # Vector store management
│   │   ├── workflow.py        # LangGraph workflow
│   │   └── orchestrator.py    # Main orchestrator
│   └── utils/
├── data/
│   ├── raw/                   # Input documents
│   ├── processed/             # Processed documents
│   └── vector_store/          # Vector database
├── examples/
│   ├── sample_usage.py        # Usage examples
│   └── quick_start.sh         # Quick start script
├── main.py                    # CLI interface
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## How It Works

### 1. Document Indexing

Documents are processed through these steps:
1. **Loading**: Support for PDF, TXT, DOCX, and other formats
2. **Chunking**: Split into overlapping chunks (default: 1000 chars with 200 overlap)
3. **Embedding**: Convert to vector embeddings using OpenAI
4. **Storage**: Store in ChromaDB vector database

### 2. Query Processing Workflow

When a query is received:

1. **Query Analysis**: The Query Analyzer Agent:
   - Identifies the intent (factual, exploratory, analytical)
   - Extracts key entities
   - Reformulates the query for better retrieval
   - Determines if multi-step reasoning is needed

2. **Document Retrieval**: The Retrieval Agent:
   - Performs similarity search on the vector store
   - Uses the reformulated query
   - Applies similarity thresholds
   - Optional reranking for better relevance

3. **Answer Synthesis**: The Synthesis Agent:
   - Generates a comprehensive answer
   - Cites sources properly
   - Maintains context from retrieved documents
   - Suggests follow-up questions

4. **Quality Evaluation**: The Evaluation Agent:
   - Scores the answer on multiple dimensions:
     - Relevance (0-1)
     - Completeness (0-1)
     - Accuracy (0-1)
     - Clarity (0-1)
   - Checks for hallucinations
   - Determines if regeneration is needed
   - Approves or rejects the answer

5. **Retry Logic**: If the answer is rejected:
   - Retry synthesis with adjusted parameters
   - Or retry retrieval with expanded queries
   - Up to configurable max retries

### 3. Durable Execution

LangGraph's checkpointing ensures:
- **State Persistence**: All workflow states are saved
- **Crash Recovery**: Resume from last checkpoint after failures
- **Debugging**: Inspect intermediate states
- **Thread Management**: Track multiple conversations

## Advanced Features

### Custom Agents

Create custom agents by extending base classes:

```python
from src.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def process(self, input_data):
        # Your custom logic
        return result
```

### Custom Workflow Nodes

Add custom nodes to the workflow:

```python
def custom_node(state: AgenticRAGState) -> AgenticRAGState:
    # Your custom processing
    state["custom_field"] = process_data(state)
    return state

workflow.add_node("custom_step", custom_node)
workflow.add_edge("analyze_query", "custom_step")
```

### Custom Evaluation Metrics

Define custom evaluation criteria:

```python
evaluator = EvaluationAgent(approval_threshold=0.8)
# Or implement custom evaluation logic
```

## Performance Considerations

- **Chunk Size**: Smaller chunks (500-1000 chars) work better for precise queries
- **Top-K Documents**: Start with 5, increase if answers lack context
- **Similarity Threshold**: Adjust (0.6-0.8) based on domain specificity
- **Model Selection**: GPT-4 for complex reasoning, GPT-3.5 for speed
- **Caching**: Vector store results are cached for repeated queries

## Troubleshooting

### Common Issues

1. **"Vector store not initialized"**
   - Solution: Run `python3 main.py index <path>` first

2. **Low quality answers**
   - Increase `TOP_K_DOCUMENTS` in .env
   - Lower `SIMILARITY_THRESHOLD`
   - Use better quality source documents

3. **Slow performance**
   - Reduce `TOP_K_DOCUMENTS`
   - Use smaller embedding model
   - Enable caching

4. **API rate limits**
   - Increase `TIMEOUT_SECONDS`
   - Reduce batch sizes
   - Add retry delays

## Examples

See `examples/sample_usage.py` for detailed examples including:
- Basic indexing and querying
- Batch query processing
- Conversation tracking with thread IDs
- Health checks and monitoring
- Collection management

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{agentic_rag,
  title = {Agentic RAG Pipeline: Multi-Agent Document Q&A System},
  author = Sai Kumar Yava,
  year = {2024},
  url = {https://github.com/scionoftech/agentic_rag}
}
```

## Support

For issues, questions, or contributions, please:
- Open an issue on GitHub
- Check existing documentation
- Review examples in the `examples/` directory

---
