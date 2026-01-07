"""
Document Rechunking Script.

Re-processes all documents using semantic chunking and rebuilds the vector store.

Usage:
    uv run python scripts/rechunk_documents.py
    uv run python scripts/rechunk_documents.py --chunk-size 1000 --overlap 200
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from app.document_processing.chunker import SemanticChunker
from app.config import settings
from app.logger import logger

console = Console()
app = typer.Typer()


async def rechunk_and_index(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    force_rebuild: bool = False,
):
    """Rechunk documents and rebuild vector store."""
    
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]DOCUMENT RECHUNKING - Semantic Chunking Pipeline[/bold cyan]")
    console.print("=" * 70)
    
    # Paths
    documents_dir = settings.documents_dir
    chromadb_dir = settings.chromadb_dir
    
    console.print(f"\n[dim]Documents: {documents_dir}[/dim]")
    console.print(f"[dim]ChromaDB: {chromadb_dir}[/dim]")
    console.print(f"[dim]Chunk size: {chunk_size}, Overlap: {chunk_overlap}[/dim]")
    
    # Check documents directory
    if not documents_dir.exists():
        console.print(f"[red]❌ Documents directory not found: {documents_dir}[/red]")
        return
    
    json_files = list(documents_dir.glob("*_response.json"))
    md_files = list(documents_dir.glob("*.md"))
    
    console.print(f"\n[green]Found {len(json_files)} JSON docs, {len(md_files)} MD files[/green]")
    
    if not json_files and not md_files:
        console.print("[yellow]No documents to process[/yellow]")
        return
    
    # Optionally clear existing ChromaDB
    if force_rebuild and chromadb_dir.exists():
        console.print("[yellow]Force rebuild: clearing existing vector store...[/yellow]")
        import shutil
        shutil.rmtree(chromadb_dir)
        chromadb_dir.mkdir(parents=True, exist_ok=True)
    
    # Process documents
    console.print("\n[cyan]Step 1: Processing documents with semantic chunking...[/cyan]")
    
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    all_documents = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(json_files) + len(md_files))
        
        for json_file in json_files:
            docs = chunker.process_json_document(json_file)
            all_documents.extend(docs)
            progress.advance(task)
        
        for md_file in md_files:
            # Skip if JSON version exists
            json_version = documents_dir / f"{md_file.stem}_response.json"
            if json_version.exists():
                progress.advance(task)
                continue
            
            docs = chunker.process_markdown_file(md_file)
            all_documents.extend(docs)
            progress.advance(task)
    
    console.print(f"\n[green]✓ Created {len(all_documents)} chunks[/green]")
    
    # Print chunk statistics
    chunk_sizes = [len(d.page_content) for d in all_documents]
    if chunk_sizes:
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        console.print(f"  Chunk sizes: avg={avg_size:.0f}, min={min_size}, max={max_size}")
    
    # Count metadata features
    dosing_chunks = sum(1 for d in all_documents if d.metadata.get("contains_dosing"))
    calc_chunks = sum(1 for d in all_documents if d.metadata.get("contains_calculation"))
    console.print(f"  With dosing info: {dosing_chunks}")
    console.print(f"  With calculations: {calc_chunks}")
    
    # Index into ChromaDB
    console.print("\n[cyan]Step 2: Indexing into vector store...[/cyan]")

    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    console.print(f"  Using embeddings: {settings.hf_embedding_model}")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.hf_embedding_model,
        model_kwargs={"trust_remote_code": True}
    )
    
    # Create or update vector store
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing documents...", total=None)
        
        vectorstore = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            collection_name=settings.chroma_collection_name,
            persist_directory=str(chromadb_dir),
        )
        
        progress.update(task, completed=True)
    
    # Verify
    collection = vectorstore._collection
    count = collection.count()
    
    console.print(f"\n[green]✓ Vector store updated: {count} chunks indexed[/green]")
    
    # Test retrieval
    console.print("\n[cyan]Step 3: Testing retrieval...[/cyan]")
    
    test_query = "neonatal protein requirements amino acids"
    results = vectorstore.similarity_search(test_query, k=3)
    
    console.print(f"  Query: '{test_query}'")
    console.print(f"  Results: {len(results)} documents found")
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")[:30]
        preview = doc.page_content[:100].replace('\n', ' ')
        console.print(f"    {i}. [{source}] {preview}...")
    
    console.print("\n" + "=" * 70)
    console.print("[bold green]✓ RECHUNKING COMPLETE[/bold green]")
    console.print("=" * 70)
    
    return all_documents


@app.command()
def main(
    chunk_size: int = typer.Option(1000, "--chunk-size", "-s", help="Target chunk size"),
    chunk_overlap: int = typer.Option(200, "--overlap", "-o", help="Chunk overlap"),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild (clear existing)"),
):
    """Rechunk documents and rebuild vector store."""
    asyncio.run(rechunk_and_index(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_rebuild=force,
    ))


if __name__ == "__main__":
    app()
