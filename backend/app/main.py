from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Ensure we can import sibling routers/services when loaded either as top-level or package
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
NCKH_DIR = PROJECT_ROOT / "NCKH"
RAG_SYSTEM_DIR = PROJECT_ROOT / "rag_system"

if str(NCKH_DIR) not in sys.path:
    sys.path.insert(0, str(NCKH_DIR))
if str(RAG_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_SYSTEM_DIR))

from routers import chat, documents, auth
from services.rag_service import init_rag, get_index_status, set_index_status
from routers.auth import get_current_user
from database import connect_db, close_db
from services.user_service import seed_default_admin
import threading


def _warmup_rag():
    try:
        print("[Backend] Warming up RAG in background...")
        init_rag()
    except Exception as exc:
        print(f"[Backend] RAG warmup failed: {exc}")
        set_index_status("error", str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await connect_db()
        await seed_default_admin()
        print("[Backend] MongoDB ready.")
    except Exception as exc:
        print(f"[Backend] MongoDB not available at startup ({exc}). Auth will retry on demand.")
    threading.Thread(target=_warmup_rag, daemon=True).start()
    yield
    await close_db()


app = FastAPI(title="VHU Document Assistant API", version="1.0.0", lifespan=lifespan)

import os

_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
_extra_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"], dependencies=[Depends(get_current_user)])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"], dependencies=[Depends(get_current_user)])

# Lazy init: endpoints call init_rag() on first use if needed (prevents long blocking startup)
# @app.on_event("startup")
# async def startup_event():
#     print("[Backend] Initializing RAG...")
#     init_rag()
#     print("[Backend] Ready.")

@app.get("/")
async def root():
    return {"status": "ok", "message": "VHU RAG API"}

@app.get("/api/me")
async def me(user=Depends(get_current_user)):
    return {
        "username": user.username,
        "email": user.email,
        "role": user.role,
    }

@app.get("/api/index-status")
async def index_status():
    """Returns current indexing status for UI polling.
    Possible values: idle | indexing | ready | error
    """
    return get_index_status()