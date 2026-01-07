#!/usr/bin/env python3
"""
TPN RAG - Minimal Production Pipeline

This is the SIMPLEST possible production RAG for the TPN knowledge base.
It uses the pre-chunked JSON segments directly.

Usage:
    # Build index
    python3 tpn_rag.py build

    # Ask question
    python3 tpn_rag.py ask "What is the protein requirement for preterm infants?"

    # Interactive mode
    python3 tpn_rag.py chat
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "documents"
CHROMADB_DIR = PROJECT_ROOT / "data" / "chromadb"

# HuggingFace settings (no Ollama)
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
LLM_MODEL = "chandramax/tpn-gpt-oss-20b"

COLLECTION_NAME = "tpn_documents"


# =============================================================================
# DATA LOADING - Use pre-chunked segments directly
# =============================================================================

def clean_text(text: str) -> str:
    """Clean artifacts from chunk content."""
    text = re.sub(r'<a id=[\'"][^"\']+[\'"]></a>\s*', '', text)
    text = re.sub(r'\s*CAPTION ERROR\s*', '', text)
    text = re.sub(r'^\s*NASPGHAN\s*FOUNDATION\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'<::NASPGHAN[^>]*::\s*\w+::\s*>', '', text)
    text = re.sub(r'<::logo:[^>]+:>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_json_chunks(json_path: Path) -> List[Dict[str, Any]]:
    """Load pre-chunked segments from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path.name}: {e}")
        return []

    source = json_path.stem.replace('_response', '')
    chunks = data.get('chunks', [])

    results = []
    for chunk in chunks:
        ctype = chunk.get('type', 'text')
        if ctype == 'logo':
            continue

        text = clean_text(chunk.get('markdown', ''))
        if len(text) < 50:
            continue

        # Check for clinical content
        has_dosing = bool(re.search(
            r'\d+\.?\d*\s*(mg|g|mcg|mL|mEq|kcal)/kg',
            text, re.IGNORECASE
        ))

        results.append({
            'content': text,
            'metadata': {
                'source': source,
                'type': ctype,
                'page': chunk.get('grounding', {}).get('page', 0),
                'has_dosing': has_dosing,
            }
        })

    return results


def load_all_chunks() -> List[Dict[str, Any]]:
    """Load all chunks from all JSON files."""
    json_files = list(DATA_DIR.glob('*_response.json'))
    print(f"Loading {len(json_files)} JSON files...")

    all_chunks = []
    for jf in json_files:
        chunks = load_json_chunks(jf)
        all_chunks.extend(chunks)

    print(f"Loaded {len(all_chunks)} chunks")
    return all_chunks


# =============================================================================
# VECTOR STORE
# =============================================================================

def build_index():
    """Build the vector store from pre-chunked documents."""
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    import shutil

    # Load chunks
    chunks = load_all_chunks()
    if not chunks:
        print("No chunks found!")
        return

    # Convert to LangChain documents
    documents = [
        Document(page_content=c['content'], metadata=c['metadata'])
        for c in chunks
    ]

    # Clear old index
    if CHROMADB_DIR.exists():
        shutil.rmtree(CHROMADB_DIR)
    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)

    # Create embeddings using HuggingFace
    print(f"Creating embeddings with {EMBED_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"trust_remote_code": True},
    )

    # Build index (in batches for large datasets)
    print("Building vector store...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMADB_DIR),
    )

    count = vectorstore._collection.count()
    print(f"\n✓ Vector store ready with {count} chunks")
    print(f"  Location: {CHROMADB_DIR}")


def get_vectorstore():
    """Get the vector store."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"trust_remote_code": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMADB_DIR),
    )


# =============================================================================
# RAG QUERY
# =============================================================================

def ask(question: str, k: int = 5) -> Dict[str, Any]:
    """Answer a question using RAG."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Retrieve
    vs = get_vectorstore()
    docs = vs.similarity_search(question, k=k)

    if not docs:
        return {
            'answer': "No relevant information found.",
            'sources': [],
            'grounded': False,
        }

    # Format context
    context = "\n\n---\n\n".join([
        f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}"
        for d in docs
    ])

    # Generate - Grounded prompt
    prompt = f"""You are a clinical TPN (Total Parenteral Nutrition) expert assistant.
Answer the question based ONLY on the provided context. If the answer is not in the context, say so.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, clinically accurate answer. Cite specific values and sources from the context."""

    # Load model (lazy loading for efficiency)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return {
        'answer': response,
        'sources': [d.metadata.get('source', 'Unknown') for d in docs],
        'grounded': True,
        'context': context[:500],
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nQuick start:")
        print("  python3 tpn_rag.py build    # Build vector store")
        print("  python3 tpn_rag.py ask 'What is protein requirement?'")
        print("  python3 tpn_rag.py chat     # Interactive mode")
        return

    cmd = sys.argv[1]

    if cmd == 'build':
        build_index()

    elif cmd == 'ask' and len(sys.argv) > 2:
        question = ' '.join(sys.argv[2:])
        print(f"\nQuestion: {question}\n")
        result = ask(question)
        print(result['answer'])
        print(f"\n[Sources: {', '.join(set(result['sources'][:3]))}]")

    elif cmd == 'chat':
        print("TPN RAG - Interactive Mode (type 'quit' to exit)\n")
        while True:
            try:
                q = input("\nYou: ").strip()
                if not q or q.lower() in ['quit', 'exit', 'q']:
                    break
                result = ask(q)
                print(f"\nAssistant: {result['answer']}")
                print(f"[Sources: {', '.join(set(result['sources'][:3]))}]")
            except KeyboardInterrupt:
                break
        print("\nGoodbye!")

    else:
        print(f"Unknown command: {cmd}")
        print("Use: build, ask, or chat")


if __name__ == "__main__":
    main()
