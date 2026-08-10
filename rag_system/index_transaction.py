"""Atomic index directory promotion with rollback."""
from __future__ import annotations
import shutil, uuid
from pathlib import Path
from threading import Lock

REBUILD_LOCK=Lock()
REQUIRED=("index.faiss","chunks_cache.json","table_store.sqlite","table_manifest.json")

def verify(path: Path):
    missing=[name for name in REQUIRED if not (path/name).exists()]
    if missing: raise RuntimeError(f"missing artifacts: {missing}")

def promote(active: Path, temp: Path, loader):
    verify(temp); backup=active.with_name(active.name+".backup."+uuid.uuid4().hex)
    moved=False
    try:
        if active.exists(): active.rename(backup); moved=True
        temp.rename(active)
        loaded=loader(active)
    except Exception:
        if active.exists() and moved: shutil.rmtree(active,ignore_errors=True)
        if moved and backup.exists(): backup.rename(active)
        raise
    if backup.exists(): shutil.rmtree(backup)
    return loaded
