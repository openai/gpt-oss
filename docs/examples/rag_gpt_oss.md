# Minimal RAG + gpt-oss Example (FAISS Retrieval)

This example demonstrates a simple, production-style Retrieval-Augmented Generation (RAG) pipeline using FAISS, sentence-transformers, and gpt-oss (or any OpenAI-compatible endpoint).

**No project configs or core files are changed. All code and dependencies are local to `examples/`.**

## Setup

1. Install requirements (in a virtualenv):

```sh
pip install -r examples/requirements-rag.txt
```

2. Set environment variables:

- `OPENAI_API_KEY` (your key)
- `OPENAI_BASE_URL` (e.g., `http://localhost:8000/v1` for vLLM/gpt-oss)
- `GPT_OSS_MODEL` (model name, e.g., `gpt-oss-20b`)

## Usage

```sh
python examples/rag_gpt_oss.py --query "What is vector search?" --top_k 4
```

Optional flags:
- `--rebuild-index` (force reindex)
- `--no-stream` (disable streaming)
- `--chunk-size` (default 800)
- `--chunk-overlap` (default 120)

## What it does

- Loads docs from `examples/data/*.{txt,md,pdf}` (PDFs require `pymupdf`)
- Builds or loads a FAISS index in `examples/data/.faiss/`
- Retrieves top-k chunks with metadata (source file, char span)
- Constructs a Harmony prompt (system guides behavior, user includes question and retrieved context, sources cited)
- Calls an OpenAI-compatible chat endpoint using the official `openai` Python SDK
- Streams output (unless `--no-stream`)
- Prints answer and compact citations list ([source:filename#chunk])
- Saves a JSONL transcript to `examples/data/runs/{timestamp}.jsonl`

## Example Output

```
Answer: Vector search is a method ...

Sources:
[1] intro_vector_search.md
[2] embeddings_and_faiss.md
```

## Pointing to a Local vLLM Server

Set `OPENAI_BASE_URL` to your vLLM/gpt-oss endpoint, e.g.:

```
export OPENAI_BASE_URL=http://localhost:8000/v1
```

## Notes

- This is a minimal, example-only script. It does not alter project configs or CI.
- If required packages (faiss, pymupdf) are missing, install hints are printed and the script exits cleanly.
- All code is self-contained under `examples/`, with no changes to core project files.
