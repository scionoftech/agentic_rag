#!/usr/bin/env python3
"""
Main CLI interface for Agentic RAG pipeline.
"""
import asyncio
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core.orchestrator import AgenticRAGOrchestrator

console = Console()


def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                    AGENTIC RAG PIPELINE                   ║
    ║          Multi-Agent Orchestration with LangGraph         ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


async def index_documents(orchestrator: AgenticRAGOrchestrator, args):
    """Index documents into the vector store."""
    console.print("\n[bold blue]Indexing Documents[/bold blue]")

    source_path = Path(args.source)

    if not source_path.exists():
        console.print(f"[red]Error: Path does not exist: {source_path}[/red]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Indexing documents...", total=None)

        result = orchestrator.index_documents(
            source_path=source_path,
            is_directory=source_path.is_dir(),
            recreate_index=args.recreate
        )

    if result["status"] == "success":
        console.print(f"\n[green]✓ Successfully indexed {result['documents_indexed']} document chunks[/green]")

        # Display collection stats
        if "collection_stats" in result:
            stats = result["collection_stats"]
            table = Table(title="Collection Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in stats.items():
                table.add_row(str(key), str(value))

            console.print(table)
    else:
        console.print(f"\n[red]✗ Indexing failed: {result['message']}[/red]")


async def query_interactive(orchestrator: AgenticRAGOrchestrator):
    """Interactive query mode."""
    console.print("\n[bold green]Interactive Query Mode[/bold green]")
    console.print("Type your questions or 'exit' to quit.\n")

    while True:
        try:
            query = console.input("[bold cyan]You:[/bold cyan] ")

            if not query.strip():
                continue

            if query.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break

            # Process query
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing query...", total=None)
                result = await orchestrator.query(query)

            # Display results
            if result["status"] == "success":
                console.print("\n[bold green]Assistant:[/bold green]")

                # Display answer
                answer_panel = Panel(
                    Markdown(result["answer"]),
                    title="Answer",
                    border_style="green"
                )
                console.print(answer_panel)

                # Display sources
                if result.get("sources"):
                    sources_text = "\n".join(f"- {source}" for source in result["sources"])
                    sources_panel = Panel(
                        sources_text,
                        title="Sources",
                        border_style="blue"
                    )
                    console.print(sources_panel)

                # Display evaluation metrics if available
                if result.get("evaluation"):
                    eval_result = result["evaluation"]
                    metrics = Table(title="Quality Metrics", show_header=False)
                    metrics.add_column("Metric", style="cyan")
                    metrics.add_column("Score", style="yellow")

                    metrics.add_row("Overall Score", f"{eval_result.overall_score:.2f}")
                    metrics.add_row("Relevance", f"{eval_result.relevance_score:.2f}")
                    metrics.add_row("Completeness", f"{eval_result.completeness_score:.2f}")
                    metrics.add_row("Accuracy", f"{eval_result.accuracy_score:.2f}")

                    console.print(metrics)

            else:
                console.print(f"\n[red]Error: {result.get('error', 'Unknown error')}[/red]")

            console.print("\n" + "="*60 + "\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]\n")


async def query_single(orchestrator: AgenticRAGOrchestrator, query: str):
    """Process a single query."""
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing query...", total=None)
        result = await orchestrator.query(query)

    if result["status"] == "success":
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(Panel(Markdown(result["answer"]), border_style="green"))

        if result.get("sources"):
            console.print("\n[bold blue]Sources:[/bold blue]")
            for source in result["sources"]:
                console.print(f"  - {source}")
    else:
        console.print(f"\n[red]Error: {result.get('error', 'Unknown error')}[/red]")


def show_info(orchestrator: AgenticRAGOrchestrator):
    """Show collection information."""
    console.print("\n[bold blue]Collection Information[/bold blue]\n")

    info = orchestrator.get_collection_info()

    table = Table(title="Vector Store Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    for key, value in info.items():
        table.add_row(str(key), str(value))

    console.print(table)


def health_check(orchestrator: AgenticRAGOrchestrator):
    """Perform health check."""
    console.print("\n[bold blue]Health Check[/bold blue]\n")

    health = orchestrator.health_check()

    status_color = "green" if health["status"] == "healthy" else "yellow" if health["status"] == "degraded" else "red"
    console.print(f"Overall Status: [{status_color}]{health['status'].upper()}[/{status_color}]\n")

    for component, status in health["components"].items():
        component_status = status.get("status", "unknown")
        color = "green" if component_status == "ok" else "red"
        console.print(f"  [{color}]●[/{color}] {component}: {component_status}")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG Pipeline - Multi-Agent Document Q&A System"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("source", type=str, help="Path to documents or directory")
    index_parser.add_argument("--recreate", action="store_true", help="Recreate index from scratch")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", type=str, nargs="?", help="Question to ask (if omitted, enters interactive mode)")

    # Info command
    subparsers.add_parser("info", help="Show collection information")

    # Health command
    subparsers.add_parser("health", help="Perform health check")

    # Reset command
    subparsers.add_parser("reset", help="Reset the vector store index")

    args = parser.parse_args()

    print_banner()

    # Initialize orchestrator
    console.print("[yellow]Initializing Agentic RAG Pipeline...[/yellow]")
    orchestrator = AgenticRAGOrchestrator(enable_checkpointing=True)
    console.print("[green]✓ Initialization complete[/green]\n")

    # Execute command
    if args.command == "index":
        await index_documents(orchestrator, args)

    elif args.command == "query":
        if args.question:
            await query_single(orchestrator, args.question)
        else:
            await query_interactive(orchestrator)

    elif args.command == "info":
        show_info(orchestrator)

    elif args.command == "health":
        health_check(orchestrator)

    elif args.command == "reset":
        console.print("[yellow]Resetting vector store index...[/yellow]")
        orchestrator.reset_index()
        console.print("[green]✓ Index reset complete[/green]")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
