# Backend - FastAPI for VHU RAG

## Setup

cd backend
pip install -r requirements.txt
# Also install from parent:
# pip install -r ../rag_system/requirements.txt -r ../NCKH/requirements.txt

## Run

python run.py
# or
uvicorn app.main:app --reload --port 8000

API docs: http://localhost:8000/docs

## Endpoints

### Chat
POST /api/chat/
{
  "question": "...",
  "use_hybrid": true,
  "filter_source": "optional filename"
}

### Documents
GET /api/documents/ - list
POST /api/documents/upload - multipart files
DELETE /api/documents/{filename}
POST /api/documents/rebuild - force rebuild

## Config
Use env vars:
RAG_DATA_DIR=...
RAG_INDEX_DIR=...
OLLAMA_MODEL=...
USE_HYBRID=true

Optional Gemini provider:
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash-lite

You can put these in `D:\NCKH\.env`; the backend loads that file at startup. If `GEMINI_API_KEY` is set and `LLM_PROVIDER` is not set, the backend uses Gemini automatically. Without a Gemini key, it falls back to Ollama.

## Next
Connect from React frontend.
Add streaming support for chat (use StreamingResponse + Ollama stream).
