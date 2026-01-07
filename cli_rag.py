#!/usr/bin/env python
"""
TPN RAG CLI - Simple command-line interface for the production RAG system.

Usage:
    # Initialize vector store from existing data
    uv run python cli_rag.py init
    uv run python cli_rag.py init --force  # Rebuild from scratch
    
    # Ask questions
    uv run python cli_rag.py ask "What is the protein requirement for preterm infants?"
    
    # Interactive mode
    uv run python cli_rag.py chat
    
    # Check stats
    uv run python cli_rag.py stats
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

console = Console()


def show_banner():
    """Show app banner."""
    console.print()
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       TPN RAG System - Clinical Knowledge Q&A            [/bold cyan]")
    console.print("[bold cyan]       Grounded in ASPEN Guidelines & TPN Protocols       [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print()


def cmd_init(force: bool = False):
    """Initialize/rebuild the vector store."""
    from app.data_loader import rebuild_vectorstore, TPNDataLoader
    from app.config import settings
    
    show_banner()
    
    # Show data stats first
    loader = TPNDataLoader()
    stats = loader.get_stats()
    
    console.print(f"[cyan]Data Directory:[/cyan] {stats['data_dir']}")
    console.print(f"[cyan]JSON Files:[/cyan] {stats['json_files']}")
    console.print(f"[cyan]Markdown Files:[/cyan] {stats['md_files']}")
    console.print()
    
    if force:
        console.print("[yellow]⚠ Force rebuild requested - clearing existing data[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building vector store from TPN documents...", total=None)
        count = rebuild_vectorstore(force=force)
        progress.update(task, completed=True)
    
    console.print(f"\n[green]✓ Vector store ready with {count} chunks[/green]")
    console.print(f"[dim]Location: {settings.chromadb_dir}[/dim]")


def cmd_stats():
    """Show system statistics."""
    from app.config import settings
    from app.data_loader import TPNDataLoader

    show_banner()

    # Data stats
    loader = TPNDataLoader()
    data_stats = loader.get_stats()

    # Vector store stats
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.hf_embedding_model,
        model_kwargs={"trust_remote_code": True}
    )

    try:
        vs = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chromadb_dir),
        )
        vs_count = vs._collection.count()
    except:
        vs_count = 0

    table = Table(title="System Status", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Vector Store Chunks", str(vs_count))
    table.add_row("JSON Documents", str(data_stats['json_files']))
    table.add_row("Markdown Documents", str(data_stats['md_files']))
    table.add_row("Embedding Model", settings.hf_embedding_model)
    table.add_row("LLM Model", settings.hf_llm_model)
    table.add_row("Chunk Size", str(settings.chunk_size))
    table.add_row("ChromaDB Path", str(settings.chromadb_dir))

    console.print(table)

    if vs_count == 0:
        console.print("\n[yellow]⚠ No documents indexed! Run 'python cli_rag.py init' first.[/yellow]")


def cmd_ask(question: str, advanced: bool = False):
    """Ask a single question."""
    
    async def _ask():
        from app.rag_pipeline import TPN_RAG, PipelineConfig, PipelineMode
        
        show_banner()
        console.print(f"[bold]Question:[/bold] {question}")
        console.print()
        
        mode = PipelineMode.AGENTIC if advanced else PipelineMode.STANDARD
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching knowledge base...", total=None)
            
            config = PipelineConfig(mode=mode, require_grounding=True)
            rag = TPN_RAG(config=config)
            await rag.initialize()
            
            progress.update(task, description="Generating answer...")
            result = await rag.ask(question=question)
            
            progress.update(task, completed=True)
        
        # Display result
        console.print()
        
        if result.is_grounded:
            console.print(Panel(
                f"[bold green]Answer:[/bold green]\n{result.reasoning}",
                title="Response",
                border_style="green"
            ))
        else:
            console.print(Panel(
                "[yellow]Unable to find relevant information in the knowledge base.[/yellow]\n\n"
                "This question may require information not in the TPN documents.",
                title="No Grounded Answer",
                border_style="yellow"
            ))
        
        # Sources
        if result.sources:
            console.print("\n[bold]Sources:[/bold]")
            for i, src in enumerate(result.sources[:3], 1):
                console.print(f"  {i}. {src.get('source', 'Unknown')} - {src.get('section', '')[:50]}")
        
        console.print(f"\n[dim]Confidence: {result.confidence} | Time: {result.total_time_ms:.0f}ms[/dim]")
    
    asyncio.run(_ask())


def cmd_chat():
    """Interactive chat mode."""
    
    async def _chat():
        from app.rag_pipeline import TPN_RAG, PipelineConfig
        
        show_banner()
        console.print("[dim]Type 'quit' or 'exit' to stop. Type 'stats' for info.[/dim]")
        console.print()
        
        config = PipelineConfig(require_grounding=True)
        rag = TPN_RAG(config=config)
        
        console.print("[dim]Initializing...[/dim]")
        await rag.initialize()
        console.print("[green]✓ Ready[/green]\n")
        
        while True:
            try:
                question = input("\n[You] ")
            except (KeyboardInterrupt, EOFError):
                break
            
            question = question.strip()
            
            if not question:
                continue
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question.lower() == 'stats':
                stats = await rag.get_stats()
                console.print(f"Documents: {stats['documents_indexed']}, Model: {stats['llm_model']}")
                continue
            
            result = await rag.ask(question=question)
            
            if result.is_grounded:
                console.print(f"\n[TPN-RAG] {result.reasoning}")
                if result.sources:
                    console.print(f"[dim]Source: {result.sources[0].get('source', 'Unknown')}[/dim]")
            else:
                console.print(f"\n[TPN-RAG] [yellow]No grounded answer - information not in knowledge base.[/yellow]")
        
        console.print("\n[dim]Goodbye![/dim]")
    
    asyncio.run(_chat())


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TPN RAG CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # init
    init_parser = subparsers.add_parser('init', help='Initialize vector store')
    init_parser.add_argument('--force', '-f', action='store_true', help='Force rebuild')
    
    # stats
    subparsers.add_parser('stats', help='Show system statistics')
    
    # ask
    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('question', type=str, help='The question to ask')
    ask_parser.add_argument('--advanced', '-a', action='store_true', help='Use agentic mode')
    
    # chat
    subparsers.add_parser('chat', help='Interactive chat mode')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        cmd_init(force=args.force)
    elif args.command == 'stats':
        cmd_stats()
    elif args.command == 'ask':
        cmd_ask(args.question, advanced=args.advanced)
    elif args.command == 'chat':
        cmd_chat()
    else:
        parser.print_help()
        console.print("\nQuick start:")
        console.print("  1. uv run python cli_rag.py init")
        console.print("  2. uv run python cli_rag.py ask \"What is the protein requirement?\"")


if __name__ == "__main__":
    main()
