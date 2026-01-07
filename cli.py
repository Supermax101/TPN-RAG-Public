"""
TPN RAG System - Command Line Interface
Interactive menu-driven interface for the RAG system.
"""
import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from typing import Optional

from app.config import settings
from app.providers import HuggingFaceEmbeddingProvider, ChromaVectorStore
from app.models import HuggingFaceProvider
from app.services import RAGService, DocumentLoader, HybridRAGService, AdvancedRAGConfig

cli = typer.Typer(help="TPN RAG System CLI", invoke_without_command=True)
console = Console()


def get_rag_service(llm_model: str = None, advanced: bool = False) -> RAGService:
    """Creates RAG service instance using HuggingFace models."""
    embedding_provider = HuggingFaceEmbeddingProvider()
    vector_store = ChromaVectorStore()
    llm_provider = HuggingFaceProvider(model_name=llm_model or settings.hf_llm_model)
    
    if advanced:
        # Config aligned with evaluation script settings
        # HyDE and Multi-Query significantly improve retrieval quality
        config = AdvancedRAGConfig(
            enable_bm25_hybrid=True,
            enable_cross_encoder=True,
            enable_multi_query=True,   # ENABLED: Generates query variants
            enable_hyde=True           # ENABLED: Hypothetical Document Embeddings
        )
        return HybridRAGService(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            llm_provider=llm_provider,
            advanced_config=config
        )
    
    return RAGService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_provider=llm_provider
    )


def show_banner():
    """Display the application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║          TPN RAG System - Clinical Q&A Assistant          ║
║      Retrieval-Augmented Generation for TPN Guidelines    ║
╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def show_main_menu():
    """Display the main interactive menu."""
    show_banner()
    
    menu = """
[bold]Available Commands:[/bold]

  [cyan]1. init[/cyan]      - Initialize: Load documents & create embeddings
  [cyan]2. stats[/cyan]     - Show collection statistics  
  [cyan]3. ask[/cyan]       - Ask a question (Simple RAG)
  [cyan]4. ask-adv[/cyan]   - Ask a question (Advanced RAG: BM25 + Reranking)
  [cyan]5. eval[/cyan]      - Run MCQ evaluation
  [cyan]6. serve[/cyan]     - Start API server
  [cyan]7. reset[/cyan]     - Reset embeddings database
  [cyan]0. exit[/cyan]      - Exit

[bold]Quick Start:[/bold]
  1. First run [cyan]init[/cyan] to create embeddings from documents
  2. Then use [cyan]ask[/cyan] or [cyan]ask-adv[/cyan] to query the system
  3. Run [cyan]eval[/cyan] to test accuracy on MCQ questions
"""
    console.print(Panel(menu, title="Main Menu", border_style="blue"))


@cli.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """TPN RAG System - Interactive CLI"""
    if ctx.invoked_subcommand is None:
        interactive_mode()


def interactive_mode():
    """Run interactive menu mode."""
    while True:
        show_main_menu()
        
        choice = Prompt.ask("\n[bold]Enter command[/bold]", default="help")
        
        if choice in ["0", "exit", "quit", "q"]:
            console.print("\n[yellow]Goodbye![/yellow]\n")
            break
        elif choice in ["1", "init"]:
            run_init_interactive()
        elif choice in ["2", "stats"]:
            run_stats()
        elif choice in ["3", "ask"]:
            run_ask_interactive(advanced=False)
        elif choice in ["4", "ask-adv", "advanced"]:
            run_ask_interactive(advanced=True)
        elif choice in ["5", "eval"]:
            run_eval_interactive()
        elif choice in ["6", "serve"]:
            run_serve()
        elif choice in ["7", "reset"]:
            run_reset()
        else:
            console.print(f"[red]Unknown command: {choice}[/red]")
        
        input("\nPress Enter to continue...")


def run_init_interactive():
    """Interactive initialization."""
    console.print("\n[bold]Initialize RAG System[/bold]\n")
    
    rag_service = get_rag_service()
    
    async def do_init():
        stats = await rag_service.get_collection_stats()
        
        if stats["total_chunks"] > 0:
            console.print(f"[yellow]Collection already has {stats['total_chunks']} chunks.[/yellow]")
            if not Confirm.ask("Do you want to reinitialize (delete existing)?"):
                return
            rag_service.vector_store.reset_collection()
        
        console.print("\n[cyan]Loading documents from data/documents/...[/cyan]")
        loader = DocumentLoader(rag_service)
        result = await loader.load_all_documents()
        
        console.print(f"\n[green]Success! Loaded {result['loaded']} documents ({result['total_chunks']} chunks)[/green]")
        if result['failed'] > 0:
            console.print(f"[red]Failed: {result['failed']} documents[/red]")
    
    asyncio.run(do_init())


def run_stats():
    """Show collection statistics."""
    console.print("\n[bold]Collection Statistics[/bold]\n")
    
    rag_service = get_rag_service()
    
    async def do_stats():
        stats = await rag_service.get_collection_stats()
        
        table = Table(title="RAG System Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Chunks", str(stats.get("total_chunks", 0)))
        table.add_row("Total Documents", str(stats.get("total_documents", 0)))
        table.add_row("Collection Name", stats.get("collection_name", "N/A"))
        table.add_row("Embedding Model", rag_service.embedding_provider.model_name)
        table.add_row("LLM Model", rag_service.llm_provider.default_model or "Auto-detected")
        
        console.print(table)
        
        if stats.get("total_chunks", 0) == 0:
            console.print("\n[yellow]No documents loaded. Run 'init' first.[/yellow]")
    
    asyncio.run(do_stats())


def run_ask_interactive(advanced: bool = False):
    """Interactive question asking."""
    from app.models import RAGQuery
    
    mode = "Advanced RAG (BM25 + Cross-Encoder)" if advanced else "Simple RAG"
    console.print(f"\n[bold]Ask a Question - {mode}[/bold]\n")
    
    question = Prompt.ask("[cyan]Enter your question[/cyan]")
    
    if not question.strip():
        console.print("[red]No question provided[/red]")
        return
    
    rag_service = get_rag_service(advanced=advanced)
    
    async def do_ask():
        stats = await rag_service.get_collection_stats()
        if stats.get("total_chunks", 0) == 0:
            console.print("[red]No documents loaded. Run 'init' first.[/red]")
            return
        
        console.print(f"\n[dim]Searching and generating answer...[/dim]\n")
        
        query = RAGQuery(
            question=question,
            search_limit=5,
            temperature=0.1
        )
        
        response = await rag_service.ask(query)
        
        console.print(Panel(response.answer, title="Answer", border_style="green"))
        
        if response.sources:
            table = Table(title="Sources Used")
            table.add_column("Document", style="cyan")
            table.add_column("Score", style="yellow")
            table.add_column("Section")
            
            for source in response.sources[:5]:
                table.add_row(
                    source.document_name[:50],
                    f"{source.score:.3f}",
                    source.chunk.section or "N/A"
                )
            
            console.print(table)
        
        console.print(f"\n[dim]Time: Search {response.search_time_ms:.0f}ms | Generation {response.generation_time_ms:.0f}ms | Total {response.total_time_ms:.0f}ms[/dim]")
    
    asyncio.run(do_ask())


def run_eval_interactive():
    """Interactive evaluation."""
    console.print("\n[bold]Run MCQ Evaluation[/bold]\n")
    
    console.print("""
[cyan]Evaluation Options:[/cyan]

  For MCQ Evaluation:
  1. RAG Evaluation (custom metrics) - Tests RAG pipeline with your MCQ dataset
  2. Baseline Evaluation (LLM only) - Tests LLM without RAG for comparison

  For Open-Ended Q&A:
  3. RAGAS Evaluation (requires OpenAI API) - Industry-standard for paragraph answers
""")
    
    choice = Prompt.ask("Select evaluation type", choices=["1", "2", "3"], default="1")
    
    if choice == "1":
        console.print("\n[cyan]Starting RAG Evaluation...[/cyan]\n")
        import subprocess
        subprocess.run(["uv", "run", "python", "eval/rag_evaluation.py"])
    elif choice == "2":
        console.print("\n[cyan]Starting Baseline Evaluation...[/cyan]\n")
        import subprocess
        subprocess.run(["uv", "run", "python", "eval/baseline_evaluation.py"])
    else:
        console.print("\n[cyan]Starting RAGAS Evaluation (uses OpenAI API)...[/cyan]\n")
        import subprocess
        subprocess.run(["uv", "run", "python", "eval/ragas_evaluation.py"])


def run_serve():
    """Start the API server."""
    console.print("\n[bold]Starting API Server[/bold]\n")
    
    host = Prompt.ask("Host", default="0.0.0.0")
    port = Prompt.ask("Port", default="8000")
    
    console.print(f"\n[cyan]Starting server at http://{host}:{port}[/cyan]")
    console.print("[dim]API docs at /docs | Press Ctrl+C to stop[/dim]\n")
    
    import uvicorn
    uvicorn.run("app.api.app:app", host=host, port=int(port))


def run_reset():
    """Reset the vector store."""
    console.print("\n[bold]Reset Collection[/bold]\n")
    
    if not Confirm.ask("[red]This will delete all embeddings. Continue?[/red]"):
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    rag_service = get_rag_service()
    rag_service.vector_store.reset_collection()
    
    console.print("[green]Collection reset successfully[/green]")


# Direct command versions (for non-interactive use)
@cli.command()
def init(force: bool = typer.Option(False, "--force", "-f", help="Force re-initialization")):
    """Initialize: Load documents and create embeddings."""
    console.print("[bold]Initializing RAG system...[/bold]\n")
    settings.ensure_directories()
    
    rag_service = get_rag_service()
    
    async def do_init():
        stats = await rag_service.get_collection_stats()
        
        if stats["total_chunks"] > 0 and not force:
            console.print(f"[yellow]Collection has {stats['total_chunks']} chunks. Use --force to reinitialize.[/yellow]")
            return
        
        if force:
            rag_service.vector_store.reset_collection()
        
        loader = DocumentLoader(rag_service)
        result = await loader.load_all_documents()
        
        console.print(f"[green]Loaded {result['loaded']} documents ({result['total_chunks']} chunks)[/green]")
    
    asyncio.run(do_init())


@cli.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    advanced: bool = typer.Option(False, "--advanced", "-a", help="Use advanced RAG")
):
    """Ask a question using RAG."""
    from app.models import RAGQuery
    
    mode = "Advanced" if advanced else "Simple"
    console.print(f"[dim]{mode} RAG[/dim]\n")
    
    rag_service = get_rag_service(advanced=advanced)
    
    async def do_ask():
        query = RAGQuery(question=question, search_limit=5, temperature=0.1)
        response = await rag_service.ask(query)
        
        console.print(Panel(response.answer, title="Answer", border_style="green"))
        
        if response.sources:
            table = Table(title="Sources")
            table.add_column("Document")
            table.add_column("Score")
            
            for s in response.sources[:5]:
                table.add_row(s.document_name[:40], f"{s.score:.3f}")
            
            console.print(table)
    
    asyncio.run(do_ask())


@cli.command()
def stats():
    """Show collection statistics."""
    run_stats()


@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p")
):
    """Start the FastAPI server."""
    import uvicorn
    console.print(f"[cyan]Server at http://{host}:{port} | Docs at /docs[/cyan]")
    uvicorn.run("app.api.app:app", host=host, port=port)


@cli.command()
def reset():
    """Reset the embeddings database."""
    if Confirm.ask("[red]Delete all embeddings?[/red]"):
        rag_service = get_rag_service()
        rag_service.vector_store.reset_collection()
        console.print("[green]Reset complete[/green]")


@cli.command(name="eval")
def run_evaluation(
    baseline: bool = typer.Option(False, "--baseline", "-b", help="Run baseline (no RAG)"),
    limit: int = typer.Option(None, "--limit", "-l", help="Max questions")
):
    """Run MCQ evaluation."""
    import subprocess
    
    if baseline:
        cmd = ["uv", "run", "python", "eval/baseline_evaluation.py"]
    else:
        cmd = ["uv", "run", "python", "eval/rag_evaluation.py"]
    
    subprocess.run(cmd)


if __name__ == "__main__":
    cli()
