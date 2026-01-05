#!/usr/bin/env python
"""
TPN RAG CLI v2 - Production Interface.

Interactive CLI for the TPN RAG system with:
- Document ingestion (PDF, Markdown, JSON)
- Question answering with grounding verification
- Evaluation and testing
- System management

Usage:
    uv run python cli_v2.py
    uv run python cli_v2.py ask "What is the protein requirement?"
    uv run python cli_v2.py ingest /path/to/book.pdf
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.logger import logger

# Initialize
console = Console()
app = typer.Typer(
    name="tpn-rag",
    help="TPN RAG System - Clinical Q&A powered by your knowledge base",
    add_completion=False,
)


# =============================================================================
# BANNER
# =============================================================================

def show_banner():
    """Display the application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║              TPN RAG System v2.1 - Clinical Q&A                  ║
║         Grounded in YOUR Knowledge Base (ASPEN Guidelines)       ║
╚══════════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


# =============================================================================
# MAIN MENU
# =============================================================================

@app.command("menu")
def interactive_menu():
    """Start interactive menu mode."""
    show_banner()
    
    while True:
        console.print("\n[bold]Main Menu:[/bold]")
        console.print("  [cyan]1[/cyan] 📚 Ingest Documents (PDF, Markdown)")
        console.print("  [cyan]2[/cyan] ❓ Ask a Question")
        console.print("  [cyan]3[/cyan] 📊 Run Evaluation")
        console.print("  [cyan]4[/cyan] 📈 Show Statistics")
        console.print("  [cyan]5[/cyan] 🔄 Rebuild Vector Store")
        console.print("  [cyan]6[/cyan] ⚙️  Settings")
        console.print("  [cyan]0[/cyan] 🚪 Exit")
        
        choice = Prompt.ask("\n[bold]Select option[/bold]", default="2")
        
        if choice in ["0", "exit", "quit"]:
            console.print("\n[yellow]Goodbye![/yellow]\n")
            break
        elif choice == "1":
            asyncio.run(ingest_interactive())
        elif choice == "2":
            asyncio.run(ask_interactive())
        elif choice == "3":
            asyncio.run(eval_interactive())
        elif choice == "4":
            asyncio.run(show_stats())
        elif choice == "5":
            asyncio.run(rebuild_vectorstore())
        elif choice == "6":
            show_settings()
        else:
            console.print(f"[red]Unknown option: {choice}[/red]")


# =============================================================================
# DOCUMENT INGESTION
# =============================================================================

@app.command("ingest")
def ingest_command(
    path: Path = typer.Argument(..., help="Path to file or directory"),
    chunk_size: int = typer.Option(1000, "--chunk-size", "-s"),
    overlap: int = typer.Option(200, "--overlap", "-o"),
):
    """Ingest documents into the knowledge base."""
    asyncio.run(ingest_documents(path, chunk_size, overlap))


async def ingest_interactive():
    """Interactive document ingestion."""
    console.print("\n[bold]📚 Document Ingestion[/bold]")
    console.print("-" * 50)
    
    console.print("\nSupported formats:")
    console.print("  • PDF files (clinical books, guidelines)")
    console.print("  • Markdown files (.md)")
    console.print("  • JSON files (from parsing service)")
    
    path_str = Prompt.ask("\nEnter file or directory path")
    path = Path(path_str).expanduser()
    
    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        return
    
    chunk_size = int(Prompt.ask("Chunk size (characters)", default="1000"))
    overlap = int(Prompt.ask("Chunk overlap", default="200"))
    
    await ingest_documents(path, chunk_size, overlap)


async def ingest_documents(path: Path, chunk_size: int = 1000, overlap: int = 200):
    """Ingest documents from file or directory."""
    
    console.print(f"\n[cyan]Processing: {path}[/cyan]")
    
    # Import components
    from app.document_processing.chunker import SemanticChunker
    from app.document_processing.pdf_loader import PDFLoader  # We'll create this
    
    chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    all_documents = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        if path.is_file():
            task = progress.add_task(f"Processing {path.name}...")
            
            if path.suffix.lower() == ".pdf":
                # Load PDF
                loader = PDFLoader()
                docs = await loader.load_and_chunk(path, chunk_size, overlap)
                all_documents.extend(docs)
            elif path.suffix.lower() == ".md":
                docs = chunker.process_markdown_file(path)
                all_documents.extend(docs)
            elif path.suffix.lower() == ".json":
                docs = chunker.process_json_document(path)
                all_documents.extend(docs)
            else:
                console.print(f"[yellow]Unsupported format: {path.suffix}[/yellow]")
                return
            
            progress.update(task, completed=True)
            
        elif path.is_dir():
            # Process directory
            pdf_files = list(path.glob("**/*.pdf"))
            md_files = list(path.glob("**/*.md"))
            json_files = list(path.glob("**/*_response.json"))
            
            total = len(pdf_files) + len(md_files) + len(json_files)
            task = progress.add_task(f"Processing {total} files...", total=total)
            
            for pdf_file in pdf_files:
                try:
                    loader = PDFLoader()
                    docs = await loader.load_and_chunk(pdf_file, chunk_size, overlap)
                    all_documents.extend(docs)
                except Exception as e:
                    console.print(f"[yellow]Warning: {pdf_file.name}: {e}[/yellow]")
                progress.advance(task)
            
            for md_file in md_files:
                docs = chunker.process_markdown_file(md_file)
                all_documents.extend(docs)
                progress.advance(task)
            
            for json_file in json_files:
                docs = chunker.process_json_document(json_file)
                all_documents.extend(docs)
                progress.advance(task)
    
    if not all_documents:
        console.print("[yellow]No documents processed[/yellow]")
        return
    
    console.print(f"\n[green]✓ Processed {len(all_documents)} chunks[/green]")
    
    # Add to vector store
    if Confirm.ask("Add to vector store?", default=True):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing documents...")
            
            try:
                from langchain_chroma import Chroma
                from langchain_ollama import OllamaEmbeddings
            except ImportError:
                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings import OllamaEmbeddings
            
            embeddings = OllamaEmbeddings(
                model=settings.ollama_embed_model or "nomic-embed-text",
                base_url=settings.ollama_base_url
            )
            
            vectorstore = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=embeddings,
                persist_directory=str(settings.chromadb_dir),
            )
            
            vectorstore.add_documents(all_documents)
            
            progress.update(task, completed=True)
        
        # Get count
        count = vectorstore._collection.count()
        console.print(f"[green]✓ Vector store now has {count} chunks[/green]")


# =============================================================================
# ASK QUESTION
# =============================================================================

@app.command("ask")
def ask_command(
    question: str = typer.Argument(..., help="Your clinical question"),
    advanced: bool = typer.Option(False, "--advanced", "-a", help="Use agentic RAG"),
):
    """Ask a clinical TPN question."""
    asyncio.run(ask_question(question, advanced))


async def ask_interactive():
    """Interactive question asking."""
    console.print("\n[bold]❓ Ask a Question[/bold]")
    console.print("-" * 50)
    
    console.print("\n[dim]Your answer will be grounded in the TPN knowledge base.")
    console.print("The system will NOT make up information.[/dim]")
    
    question = Prompt.ask("\n[bold]Question[/bold]")
    
    if not question.strip():
        return
    
    # Check for options (MCQ format)
    has_options = Confirm.ask("Is this an MCQ with options?", default=False)
    options = ""
    if has_options:
        console.print("Enter options (e.g., A. First | B. Second | C. Third):")
        options = Prompt.ask("Options")
    
    advanced = Confirm.ask("Use advanced agentic mode?", default=False)
    
    await ask_question(question, advanced, options)


async def ask_question(question: str, advanced: bool = False, options: str = ""):
    """Answer a clinical question using RAG."""
    
    console.print(f"\n[cyan]Question:[/cyan] {question}")
    if options:
        console.print(f"[cyan]Options:[/cyan] {options}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        if advanced:
            task = progress.add_task("Using Agentic RAG (with document grading)...")
            
            from app.chains.agentic_rag import create_agentic_mcq_rag
            rag = await create_agentic_mcq_rag()
            
            result = await rag.answer(
                question=question,
                options=options or "N/A",
                answer_type="single" if options else "open",
            )
        else:
            task = progress.add_task("Using Standard RAG...")
            
            from app.chains.mcq_chain import create_mcq_chain
            chain = create_mcq_chain()
            await chain.initialize()
            
            result = await chain.answer(
                question=question,
                options=options or "N/A",
                answer_type="single" if options else "open",
            )
        
        progress.update(task, completed=True)
    
    # Display result
    console.print("\n" + "=" * 60)
    
    if options and result.get("answer"):
        console.print(f"[bold green]Answer:[/bold green] {result['answer']}")
    
    console.print(f"\n[bold]Clinical Reasoning:[/bold]")
    console.print(Panel(result.get("thinking", "No reasoning provided"), border_style="dim"))
    
    # Grounding info
    console.print(f"\n[dim]Confidence: {result.get('confidence', 'unknown')}[/dim]")
    
    if result.get("rewrite_count", 0) > 0:
        console.print(f"[dim]Query refined {result['rewrite_count']} times for better retrieval[/dim]")
    
    if result.get("sources"):
        console.print("\n[bold]Sources used:[/bold]")
        for i, src in enumerate(result.get("sources", [])[:3], 1):
            console.print(f"  {i}. {src.get('source', 'Unknown')}")
    
    console.print("=" * 60)


# =============================================================================
# EVALUATION
# =============================================================================

async def eval_interactive():
    """Interactive evaluation."""
    console.print("\n[bold]📊 Run Evaluation[/bold]")
    console.print("-" * 50)
    
    csv_path = Path("eval/tpn_mcq_cleaned.csv")
    if not csv_path.exists():
        console.print(f"[red]Evaluation file not found: {csv_path}[/red]")
        return
    
    limit = Prompt.ask("Number of questions (leave empty for all)", default="")
    max_q = int(limit) if limit.isdigit() else None
    
    advanced = Confirm.ask("Use advanced agentic mode?", default=True)
    
    console.print("\n[cyan]Starting evaluation...[/cyan]")
    
    # Import and run
    from eval.rag_evaluation_v2 import RAGEvaluatorV2
    
    evaluator = RAGEvaluatorV2(
        csv_path=str(csv_path),
        model=settings.ollama_llm_model or "qwen2.5:7b",
    )
    
    await evaluator.run_evaluation(max_questions=max_q)


# =============================================================================
# STATISTICS
# =============================================================================

async def show_stats():
    """Show system statistics."""
    console.print("\n[bold]📈 System Statistics[/bold]")
    console.print("-" * 50)
    
    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import OllamaEmbeddings
    
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model or "nomic-embed-text",
        base_url=settings.ollama_base_url
    )
    
    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chromadb_dir),
    )
    
    try:
        count = vectorstore._collection.count()
    except:
        count = 0
    
    table = Table(box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    
    table.add_row("Vector Store Chunks", str(count))
    table.add_row("Collection Name", settings.chroma_collection_name)
    table.add_row("Embedding Model", settings.ollama_embed_model or "nomic-embed-text")
    table.add_row("LLM Model", settings.ollama_llm_model or "qwen2.5:7b")
    table.add_row("Chunk Size", str(settings.chunk_size))
    table.add_row("Chunk Overlap", str(settings.chunk_overlap))
    table.add_row("ChromaDB Path", str(settings.chromadb_dir))
    
    console.print(table)
    
    if count == 0:
        console.print("\n[yellow]⚠ No documents indexed! Run 'Ingest Documents' first.[/yellow]")


# =============================================================================
# REBUILD
# =============================================================================

async def rebuild_vectorstore():
    """Rebuild vector store from scratch."""
    console.print("\n[bold]🔄 Rebuild Vector Store[/bold]")
    console.print("-" * 50)
    
    if not Confirm.ask("[yellow]This will delete existing vectors. Continue?[/yellow]", default=False):
        return
    
    import subprocess
    subprocess.run([
        "uv", "run", "python", "scripts/rechunk_documents.py", "--force"
    ], cwd=project_root)


# =============================================================================
# SETTINGS
# =============================================================================

def show_settings():
    """Show current settings."""
    console.print("\n[bold]⚙️ Current Settings[/bold]")
    console.print("-" * 50)
    
    table = Table(box=box.SIMPLE)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    table.add_column("Source")
    
    table.add_row("Ollama URL", settings.ollama_base_url, "OLLAMA_BASE_URL")
    table.add_row("Embed Model", settings.ollama_embed_model or "default", "OLLAMA_EMBED_MODEL")
    table.add_row("LLM Model", settings.ollama_llm_model or "default", "OLLAMA_LLM_MODEL")
    table.add_row("Chunk Size", str(settings.chunk_size), "CHUNK_SIZE")
    table.add_row("Chunk Overlap", str(settings.chunk_overlap), "CHUNK_OVERLAP")
    
    console.print(table)
    console.print("\n[dim]Edit .env file to change settings[/dim]")


# =============================================================================
# MAIN
# =============================================================================

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """TPN RAG System - Clinical Q&A grounded in your knowledge base."""
    if ctx.invoked_subcommand is None:
        interactive_menu()


if __name__ == "__main__":
    app()
