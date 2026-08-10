import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import bcrypt

from database import get_db

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,32}$")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))


def _normalize_user(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not doc:
        return None
    return {
        "id": str(doc["_id"]),
        "username": doc["username"],
        "email": doc.get("email"),
        "role": doc.get("role", "user"),
        "is_active": doc.get("is_active", True),
        "created_at": doc.get("created_at"),
    }


async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    doc = await get_db().users.find_one({"username": username.lower()})
    return _normalize_user(doc)


async def get_user_with_hash(username: str) -> Optional[Dict[str, Any]]:
    return await get_db().users.find_one({"username": username.lower()})


async def create_user(username: str, password: str, email: str = "", role: str = "user") -> Dict[str, Any]:
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        raise ValueError("Username must be 3-32 characters (letters, numbers, underscore)")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")

    existing = await get_user_with_hash(username)
    if existing:
        raise ValueError("Username already exists")

    doc = {
        "username": username,
        "email": email.strip() or None,
        "password_hash": hash_password(password),
        "role": role if role in {"admin", "user"} else "user",
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }
    result = await get_db().users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _normalize_user(doc)


async def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    doc = await get_user_with_hash(username.strip().lower())
    if not doc or not doc.get("is_active", True):
        return None
    if not verify_password(password, doc.get("password_hash", "")):
        return None
    return _normalize_user(doc)


async def seed_default_admin() -> None:
    admin_username = os.getenv("ADMIN_USERNAME", "admin").strip().lower()
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    count = await get_db().users.count_documents({})
    if count > 0:
        return

    await create_user(admin_username, admin_password, role="admin")
    print(f"[MongoDB] Seeded default admin user: {admin_username}")