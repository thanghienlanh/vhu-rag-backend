from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

from services.rag_service import (
    get_data_dir, add_documents, init_rag, get_status,
    list_documents, delete_document, rebuild_index, search_documents,
    get_retrieval_debug, remove_documents, set_index_status
)
from services.cloud_sync import sync_from_drive, get_drive_sync_status

router = APIRouter()

DATA_DIR = Path(get_data_dir())

class DocumentInfo(BaseModel):
    name: str
    size: int
    snippet: Optional[str] = None
    match_type: Optional[str] = None

class RebuildResponse(BaseModel):
    status: str
    message: str

@router.get("/", response_model=List[DocumentInfo])
async def list_docs(
    q: str = "",
    mode: str = "name",
    limit: int = 100
):
    """Advanced search.
    - q: search keyword
    - mode: 'name' (filename) or 'semantic' (content similarity)
    """
    return search_documents(q, mode, limit)

@router.post("/upload")
async def upload_docs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    saved = []
    for file in files:
        if not file.filename.lower().endswith(('.pdf', '.docx', '.txt', '.md')):
            continue
        dest = DATA_DIR / file.filename
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved.append(str(dest))
    if saved:
        set_index_status("indexing")
        background_tasks.add_task(add_documents, saved)
    return {
        "saved": [Path(p).name for p in saved],
        "message": f"Uploaded {len(saved)} files. Auto-indexing in background (no manual Rebuild needed)."
    }

@router.get("/file/{filename}")
async def get_file(filename: str):
    """Serve a source PDF/DOCX/TXT/MD for in-app viewing (source citation click-through)."""
    safe_name = Path(filename).name  # strip any path components, block traversal
    path = DATA_DIR / safe_name
    if not path.is_file():
        raise HTTPException(404, "File not found")
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }
    media_type = media_types.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=safe_name)


@router.delete("/{filename}")
async def del_doc(filename: str, background_tasks: BackgroundTasks):
    if delete_document(filename):
        set_index_status("indexing")
        background_tasks.add_task(remove_documents, [filename])
        return {"message": f"Deleted {filename}. Index updated in background."}
    raise HTTPException(404, "File not found")

@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild(background_tasks: BackgroundTasks):
    set_index_status("indexing")
    background_tasks.add_task(rebuild_index)
    return {"status": "started", "message": "Full index rebuild started."}

@router.get("/status")
async def status():
    return get_status()


@router.post("/debug/retrieve")
async def debug_retrieve(payload: dict):
    """
    Debug & evaluation tool.
    POST body example:
    {"question": "Thời hạn đăng ký đề tài NCKH năm 2025-2026 là khi nào?", "fast_mode": false}

    Returns:
    - rewritten query
    - top chunks with approximate scores
    - sources
    - whether rewriting/rerank was used

    Use this to debug retrieval mistakes and evaluate quality.
    """
    question = payload.get("question", "")
    fast_mode = payload.get("fast_mode", False)
    if not question:
        raise HTTPException(400, "question is required")
    return get_retrieval_debug(question, fast_mode=fast_mode)

class SyncResponse(BaseModel):
    added: int
    updated: int
    skipped: int = 0
    indexing: bool = False
    message: str
    errors: List[str] = []

class DriveSyncStatus(BaseModel):
    configured: bool
    folder_id_set: bool
    auth_mode: str
    credentials_found: bool
    oauth_token_ready: bool = False
    credentials_hint: str

@router.get("/sync/status", response_model=DriveSyncStatus)
async def drive_sync_status():
    return get_drive_sync_status()

@router.post("/sync", response_model=SyncResponse)
async def sync_drive(background_tasks: BackgroundTasks, overwrite: bool = True):
    """Sync new/updated files from Google Drive into papers/ and index them."""
    try:
        result = sync_from_drive(data_dir=DATA_DIR, overwrite_if_newer=overwrite)
        changed = result.get("changed_paths") or []
        indexing = False
        if changed:
            set_index_status("indexing")
            background_tasks.add_task(add_documents, changed)
            indexing = True
        return {
            "added": result.get("added", 0),
            "updated": result.get("updated", 0),
            "skipped": result.get("skipped", 0),
            "indexing": indexing,
            "message": result.get("message", "Đồng bộ xong."),
            "errors": result.get("errors", []),
        }
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Sync failed: {str(e)}")