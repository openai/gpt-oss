#!/usr/bin/env python
"""
Minimal RAG + gpt-oss example using FAISS retrieval.
See docs/examples/rag_gpt_oss.md for details.
"""
import os
import sys
import time
import json
import argparse
import glob
import hashlib
import datetime
from pathlib import Path
from typing import List, Dict, Optional

# --- Dependency checks and fallbacks ---
try:
    import faiss
except ImportError:
    print("[ERROR] Missing dependency: faiss-cpu. Install with: pip install faiss-cpu>=1.8", file=sys.stderr)
    sys.exit(2)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR] Missing dependency: sentence-transformers. Install with: pip install sentence-transformers>=2.6", file=sys.stderr)
    sys.exit(2)
try:
    import tiktoken
    def count_tokens(text):
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text):
        return len(text.encode("utf-8")) // 4  # crude fallback
try:
    import fitz  # pymupdf
    def extract_pdf_text(path):
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
except ImportError:
    def extract_pdf_text(path):
        print("[ERROR] pymupdf not installed. Install with: pip install pymupdf>=1.24", file=sys.stderr)
        sys.exit(2)
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] Missing dependency: openai. Install with: pip install openai>=1.40", file=sys.stderr)
    sys.exit(2)

# --- Harmony helpers ---
from examples.utils.harmony_helpers import build_harmony_messages, validate_harmony_response

# --- Chunker ---
def recursive_chunk(text, chunk_size=800, chunk_overlap=120):
    """Chunk text recursively by tokens/bytes."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        if end == text_len:
            break
        start += chunk_size - chunk_overlap
    return chunks

# --- Doc loader ---
def load_docs(data_dir: str) -> List[Dict]:
    docs = []
    for path in glob.glob(os.path.join(data_dir, '*.*')):
        ext = os.path.splitext(path)[1].lower()
        if ext in {'.md', '.txt'}:
            with open(path, encoding='utf-8') as f:
                text = f.read()
        elif ext == '.pdf':
            text = extract_pdf_text(path)
        else:
            continue
        docs.append({'path': path, 'text': text})
    return docs

# --- Indexing ---
def build_or_load_faiss(docs: List[Dict], faiss_dir: str, chunk_size: int, chunk_overlap: int, model_name: str) -> (faiss.IndexFlatIP, List[Dict]):
    os.makedirs(faiss_dir, exist_ok=True)
    meta_path = os.path.join(faiss_dir, 'meta.json')
    index_path = os.path.join(faiss_dir, 'index.bin')
    chunks_path = os.path.join(faiss_dir, 'chunks.jsonl')
    # Check if index exists and is up-to-date
    doc_hash = hashlib.sha1()
    for doc in docs:
        stat = os.stat(doc['path'])
        doc_hash.update(f"{doc['path']}:{stat.st_mtime}".encode())
    hash_hex = doc_hash.hexdigest()
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('hash') == hash_hex and os.path.exists(index_path) and os.path.exists(chunks_path):
            index = faiss.read_index(index_path)
            with open(chunks_path) as f:
                chunks = [json.loads(line) for line in f]
            return index, chunks
    # Rebuild index
    model = SentenceTransformer(model_name)
    all_chunks = []
    vectors = []
    for doc in docs:
        for i, (start, end, chunk) in enumerate(recursive_chunk(doc['text'], chunk_size, chunk_overlap)):
            chunk_id = f"{os.path.basename(doc['path'])}#{i}"
            all_chunks.append({
                'id': chunk_id,
                'text': chunk,
                'source': os.path.basename(doc['path']),
                'span': [start, end],
                'path': doc['path']
            })
            vectors.append(chunk)
    if not all_chunks:
        print("[ERROR] No chunks found for indexing.", file=sys.stderr)
        sys.exit(2)
    embeds = model.encode(vectors, normalize_embeddings=True, show_progress_bar=True)
    dim = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeds)
    faiss.write_index(index, index_path)
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    with open(meta_path, 'w') as f:
        json.dump({'hash': hash_hex, 'dim': dim, 'model': model_name}, f)
    return index, all_chunks

# --- Retrieval ---
def retrieve(query: str, index, chunks: List[Dict], model_name: str, top_k: int) -> List[Dict]:
    model = SentenceTransformer(model_name)
    qvec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(qvec, top_k)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx].copy()
        chunk['score'] = float(D[0][rank])
        chunk['rank'] = rank + 1
        results.append(chunk)
    return results

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser(description="Minimal RAG + gpt-oss example (FAISS retrieval)")
    parser.add_argument('--query', required=True, help='User query')
    parser.add_argument('--top_k', type=int, default=4, help='Top-k chunks to retrieve')
    parser.add_argument('--rebuild-index', action='store_true', help='Force rebuild FAISS index')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming output')
    parser.add_argument('--chunk-size', type=int, default=800, help='Chunk size (chars)')
    parser.add_argument('--chunk-overlap', type=int, default=120, help='Chunk overlap (chars)')
    args = parser.parse_args()

    # Env vars
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')
    model = os.getenv('GPT_OSS_MODEL')
    if not (api_key and base_url and model):
        print("[ERROR] Set OPENAI_API_KEY, OPENAI_BASE_URL, and GPT_OSS_MODEL.", file=sys.stderr)
        sys.exit(2)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    faiss_dir = os.path.join(data_dir, '.faiss')
    runs_dir = os.path.join(data_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)

    docs = load_docs(data_dir)
    if not docs:
        print("[ERROR] No documents found in examples/data/", file=sys.stderr)
        sys.exit(2)

    # Index
    if args.rebuild_index:
        for f in Path(faiss_dir).glob('*'):
            f.unlink()
    index, all_chunks = build_or_load_faiss(docs, faiss_dir, args.chunk_size, args.chunk_overlap, 'sentence-transformers/all-MiniLM-L6-v2')
    if index.ntotal == 0 or not all_chunks:
        print("[ERROR] FAISS index is empty.", file=sys.stderr)
        sys.exit(2)

    # Retrieval
    retrieved = retrieve(args.query, index, all_chunks, 'sentence-transformers/all-MiniLM-L6-v2', args.top_k)
    if not retrieved:
        print("[ERROR] No relevant chunks retrieved.", file=sys.stderr)
        sys.exit(2)

    # Prompt
    system_prompt = "You are a helpful assistant. Use ONLY the provided CONTEXT. Cite sources as [1], [2], ... Map them to filenames at the end under 'Sources'."
    messages = build_harmony_messages(system_prompt, args.query, retrieved)

    # OpenAI-compatible call
    client = OpenAI(base_url=base_url, api_key=api_key)
    start_time = time.time()
    response_text = ""
    try:
        stream = not args.no_stream
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            temperature=0.2,
            max_tokens=512
        )
        if stream:
            print("\nAnswer:", end=" ", flush=True)
            for chunk in completion:
                delta = getattr(chunk.choices[0].delta, 'content', None)
                if delta:
                    print(delta, end="", flush=True)
                    response_text += delta
            print()
        else:
            response_text = completion.choices[0].message.content
            print("\nAnswer:", response_text)
    except Exception as e:
        print(f"[ERROR] Model call failed: {e}", file=sys.stderr)
        sys.exit(2)
    latency_ms = int((time.time() - start_time) * 1000)

    # Validate response
    if not validate_harmony_response(response_text):
        print("[ERROR] Model returned empty or invalid response.", file=sys.stderr)
        sys.exit(2)

    # Citations
    print("\nSources:")
    for i, chunk in enumerate(retrieved, 1):
        print(f"[{i}] {chunk['source']}")

    # Save transcript
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_path = os.path.join(runs_dir, f'{ts}.jsonl')
    with open(run_path, 'w', encoding='utf-8') as f:
        log = {
            'query': args.query,
            'retrieved_ids': [c['id'] for c in retrieved],
            'prompt': messages,
            'model': model,
            'latency_ms': latency_ms,
            'answer': response_text
        }
        f.write(json.dumps(log, ensure_ascii=False) + '\n')
    # Simple inline test
    assert os.path.exists(run_path) and os.path.getsize(run_path) > 0, "Transcript not saved!"

if __name__ == '__main__':
    main()
