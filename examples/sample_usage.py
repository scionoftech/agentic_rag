#!/usr/bin/env python3
"""
Sample usage examples for Agentic RAG pipeline.
"""
import asyncio
from pathlib import Path
from src.core.orchestrator import create_orchestrator


async def example_1_basic_usage():
    """Example 1: Basic document indexing and querying."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)

    # Create orchestrator
    orchestrator = create_orchestrator()

    # Index documents
    print("\n1. Indexing documents...")
    result = orchestrator.index_documents(
        source_path=Path("data/raw"),
        is_directory=True
    )
    print(f"   Indexed {result['documents_indexed']} chunks")

    # Query the system
    print("\n2. Querying the system...")
    query_result = await orchestrator.query(
        "What is this document about?"
    )
    print(f"\n   Answer: {query_result['answer']}")
    print(f"   Sources: {query_result['sources']}")


async def example_2_multi_query():
    """Example 2: Processing multiple queries."""
    print("\n" + "="*60)
    print("Example 2: Batch Query Processing")
    print("="*60)

    orchestrator = create_orchestrator()

    queries = [
        "What are the main topics covered?",
        "Can you summarize the key points?",
        "What are the conclusions?"
    ]

    print("\nProcessing multiple queries...")
    results = await orchestrator.batch_query(queries)

    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}: {queries[i-1]}")
        print(f"Answer: {result['answer'][:200]}...")


async def example_3_with_thread_tracking():
    """Example 3: Using thread IDs for conversation tracking."""
    print("\n" + "="*60)
    print("Example 3: Conversation Tracking")
    print("="*60)

    orchestrator = create_orchestrator()

    # Simulate a conversation
    thread_id = "conversation_001"

    queries = [
        "What is machine learning?",
        "Can you explain supervised learning?",
        "What about unsupervised learning?"
    ]

    for query in queries:
        print(f"\nUser: {query}")
        result = await orchestrator.query(query, thread_id=thread_id)
        print(f"Assistant: {result['answer'][:200]}...")


async def example_4_health_check():
    """Example 4: Performing health checks."""
    print("\n" + "="*60)
    print("Example 4: Health Check")
    print("="*60)

    orchestrator = create_orchestrator()

    health = orchestrator.health_check()
    print(f"\nSystem Status: {health['status']}")
    print("\nComponent Status:")
    for component, status in health['components'].items():
        print(f"  - {component}: {status['status']}")


async def example_5_collection_info():
    """Example 5: Getting collection information."""
    print("\n" + "="*60)
    print("Example 5: Collection Information")
    print("="*60)

    orchestrator = create_orchestrator()

    info = orchestrator.get_collection_info()
    print("\nCollection Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")


async def main():
    """Run all examples."""
    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║        AGENTIC RAG PIPELINE - USAGE EXAMPLES              ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    # Run examples
    # await example_1_basic_usage()
    # await example_2_multi_query()
    # await example_3_with_thread_tracking()
    await example_4_health_check()
    await example_5_collection_info()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
