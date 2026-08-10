import asyncio
import os
from pathlib import Path
from typing import Optional

import certifi
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    for raw_line in _env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None

MONGODB_DB = os.getenv("MONGODB_DB", "vhu_rag")


def _resolve_mongodb_uri() -> str:
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017").strip()
    password = os.getenv("MONGODB_PASSWORD", "").strip()
    if password:
        uri = uri.replace("<db_password>", password)
    if "<db_password>" in uri:
        raise ValueError(
            "MONGODB_PASSWORD is missing. Set your Atlas database password in .env"
        )
    return uri


def _mask_mongodb_uri(uri: str) -> str:
    if "@" not in uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    creds, hostpart = rest.split("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:****@{hostpart}"


def _create_client(uri: str) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(
        uri,
        serverSelectionTimeoutMS=15000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000,
        tlsCAFile=certifi.where(),
    )


async def connect_db() -> AsyncIOMotorDatabase:
    global _client, _db
    if _db is not None:
        try:
            await _client.admin.command("ping")
            return _db
        except Exception:
            await close_db()

    uri = _resolve_mongodb_uri()
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            _client = _create_client(uri)
            await _client.admin.command("ping")
            _db = _client[MONGODB_DB]
            await _db.users.create_index("username", unique=True)
            print(f"[MongoDB] Connected: {_mask_mongodb_uri(uri)} / {MONGODB_DB}")
            return _db
        except Exception as exc:
            last_exc = exc
            await close_db()
            if attempt < 2:
                wait = 1.5 * (attempt + 1)
                print(f"[MongoDB] Attempt {attempt + 1} failed ({exc}). Retry in {wait:.1f}s...")
                await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


async def close_db():
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("MongoDB is not connected")
    return _db