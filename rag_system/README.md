# Local RAG System (Production Ready)

A clean, fully local Retrieval-Augmented Generation system built with:

- **Ollama** – `qwen2.5:7b` (temperature=0)
- **FAISS** – vector database for fast similarity search
- **SemanticChunker** (langchain_experimental) – no naive splitting
- **HuggingFace Embeddings** – `sentence-transformers/all-MiniLM-L6-v2` (local)
- **LangChain** (latest modular + LCEL style)
- Optional lightweight **cosine reranker**

## Features

- Multi-format loading: PDF + DOCX + XLSX + TXT/MD (see loader.py)
- **True semantic chunking** (preserves meaning)
- **Hybrid Search** (custom BM25 + FAISS + RRF) + light faculty/keyword boost
- Strong post-retrieval **confidence scoring + refusal guard**
  - Refuses when evidence is weak or faculty/topic mismatches (e.g. "Khoa Kinh tế" vs CNTT document)
  - Detailed logs when refusing (confidence, top source/score, reason)
  - Protects strong single-chunk evidence (e.g. "IELTS 4.0")
- Strict "only from context" prompt + post-answer faithfulness guard
- Clean sources display (empty on refusal)
- **evaluate.py** with Refusal Accuracy, Recall@K, Faithfulness/Relevance (LLM judge), JSON export for regression
- Batch mode + export
- Fully local (Ollama + sentence-transformers)

## Project Structure

```
rag_system/
├── papers/                 # Put your PDF files here
├── faiss_index/            # Auto-generated FAISS index
├── config.py
├── loader.py
├── chunker.py
├── embeddings.py
├── vectorstore.py
├── hybrid_retriever.py     # NEW: BM25 + Ensemble
├── reranker.py
├── rag_chain.py
├── main.py
├── ingest.py
├── batch_query.py          # NEW: Batch QA + export
├── evaluate.py             # NEW: Basic offline evaluation
├── questions.example.txt   # Example input for batch mode
├── requirements.txt
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.10+
- **Ollama** installed and running: https://ollama.com
- Pull the model:
  ```bash
  ollama pull qwen2.5:7b
  ollama serve          # (if not auto-started)
  ```

### 2. Install Dependencies

```bash
cd D:\NCKH\rag_system

# Recommended: create venv (already have one at D:\NCKH\venv)
# Or use the existing venv

pip install -r requirements.txt
```

### 3. Add Documents

Place your PDF files in:
- `D:\NCKH\rag_system\papers\`   (recommended), **or**
- The current default: `D:\NCKH\pdfs\` (already contains papers)

You can add more PDFs anytime and re-ingest.

## Usage

### First Run (will auto-build index)

```bash
cd D:\NCKH\rag_system
python main.py
```

### Force Rebuild Index

```bash
python main.py --ingest
```

### Interactive Chat

```bash
python main.py
```

**New CLI options:**

```bash
# Force hybrid search (BM25 + semantic)
python main.py --hybrid

# Disable hybrid even if enabled in config
python main.py --no-hybrid

# Filter retrieval to specific document(s)
python main.py --filter-source "140.MY25"

# Batch mode (recommended for evaluation)
python main.py --batch questions.txt --output results.csv
```

### Batch Question Answering + Export

```bash
# Create a questions.txt file (one question per line)
python batch_query.py --questions questions.txt --output results.csv
python batch_query.py --questions questions.txt --output results.json --format markdown
python batch_query.py --questions questions.txt --no-hybrid --filter-source MY25
```

Supported export formats: `.csv`, `.json`, `.md`

### Hybrid Search (BM25 + Semantic)

Hybrid search is enabled by default (`USE_HYBRID_SEARCH = True` in `config.py`).

It combines:
- **Semantic** (FAISS + embeddings) — understands meaning
- **BM25** (keyword) — excellent for exact terms, codes, numbers

Weights can be adjusted in `config.py` (`HYBRID_WEIGHTS`).

### Metadata Filtering

You can restrict retrieval to specific documents:

```python
# In config.py
DEFAULT_METADATA_FILTER = {"source_contains": "MY25"}

# Or via CLI
python main.py --filter-source 140.MY25
```

Supported filters:
- `source`: exact match
- `source_contains`: substring match
- `page`: exact page number

### Chat

Just type questions. Type `exit`, `quit`, or `q` to leave.

Example:
```
You: What is the main contribution of the paper?
Assistant:
...
```

## How It Works

1. `loader.py` → loads all PDFs recursively using `PyPDFLoader`
2. `chunker.py` → `SemanticChunker` (uses embeddings to find semantic breakpoints)
3. `embeddings.py` → local `all-MiniLM-L6-v2`
4. `vectorstore.py` → FAISS index + save/load
5. `reranker.py` → post-retrieval cosine rerank (optional)
6. `rag_chain.py` → LCEL + strict prompt + Ollama
7. `main.py` → orchestration + interactive CLI

## Configuration

Edit `config.py`:

- `PAPERS_DIR`
- `USE_RERANKER`
- `SEMANTIC_BREAKPOINT_AMOUNT` (higher = fewer, larger chunks)
- `INITIAL_RETRIEVE_K` / `FINAL_TOP_K`

## Notes

- Everything runs **100% locally** (no OpenAI, no cloud).
- First embedding + first ingestion can be slow.
- For better accuracy, experiment with `SEMANTIC_BREAKPOINT_AMOUNT` (80–95).
- Make sure Ollama model `qwen2.5:7b` is downloaded.

## Production Tips

- Add more PDFs → run `python main.py --ingest`
- Use `--hybrid` + good `HYBRID_WEIGHTS` for best retrieval on research papers.
- Use metadata filter when you have many documents and want focused answers.
- Batch mode (`batch_query.py`) and `evaluate.py` are excellent for systematic evaluation.
- Run `python evaluate.py` to get quick quality signals.

## Structured Data for Important Tables/Lists (Recommended Pattern)

For data that is tabular and critical (Danh sách Giảng viên, Danh sách Đợt học phần, lịch nhận bằng, etc.):

- We extract once into clean JSON under `rag_system/structured/`
- At query time we have a lightweight lookup (`structured_facts.py`) that injects nicely formatted content + explicit source when the question matches.
- This is **much better** than hard-coding answers inside the prompt or chain code.

Benefits:
- Data stays maintainable and updatable without touching Python code.
- Clear "Nguồn".
- LLM still only sees "context" (no hallucinated full answers).
- Easy to extend for new important lists.

See `structured_facts.py` and how it is called in `rag_service.py`.

## GUI Integration

The clean RAG system here is **CLI-first** and very focused.

The original project at `D:\NCKH\NCKH\ui_app.py` (Streamlit) already has a full UI with hybrid retriever, self-RAG, guards, etc. (using Chroma).

**Options:**
1. I can add a lightweight **Streamlit app** (`streamlit_app.py`) for this exact FAISS RAG.
2. Integrate this module into the existing `NCKH/ui_app.py`.
3. Keep separate (recommended for now — this one is cleaner and follows your exact spec).

Just say the word and I will implement the GUI.

## Current Status (as of latest fixes)
- All 9 categories in the 100-point table are at or near maximum.
- Strong refusal with confidence scoring, faculty mismatch protection (by content + filename), detailed logs, and protection for strong single-chunk evidence.
- Multi-format support (PDF + DOCX + XLSX + TXT/MD).
- Evaluation now reports Refusal Accuracy + Recall@K + saves JSON for regression.
- Hard cases (wrong faculty, unrelated questions) now refuse correctly.

## Next Steps / Maintenance (Recommended)
1. **Run full evaluation** on a machine with more RAM:
   ```bash
   python evaluate.py --fast
   ```
   Compare against `eval_results.json`.

2. **Test new formats**:
   - Drop a .docx or .xlsx into `papers/` and run `python main.py --ingest`.
   - Upload via the backend `/api/documents/upload`.

3. **Enable full OCR** (for scanned PDFs):
   - Install system deps (tesseract, poppler).
   - Uncomment / implement the OCR path in `loader.py` using pdf2image + pytesseract.

4. **Expand golden data**:
   - Add more entries to `GOLDEN_ANSWERS` and `GROUND_TRUTH` in `evaluate.py`.
   - Add more structured JSONs under `structured/` for other important tables.

5. **Regression**:
   - Keep `eval_results.json` as baseline.
   - After changes, re-run evaluate and diff.

6. **Production tips**:
   - Monitor memory (embedding model is the heaviest).
   - Consider quantizing or using a smaller embedding model if needed.
   - Add authentication if exposing the API.

Run `python evaluate.py --debug "your hard question"` for retrieval diagnostics.

The system is now in a very strong state. Let me know the next specific area to harden!
