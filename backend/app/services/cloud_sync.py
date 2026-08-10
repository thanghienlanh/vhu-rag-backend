"""
Google Drive → local papers folder sync for VHU RAG.

Hai cach xac thuc:
  A) Service account: credentials/google-drive-service-account.json
     + share thu muc Drive voi email service account
  B) OAuth (Desktop client): credentials/google-oauth-client.json
     + chay scripts/google_drive_auth.py mot lan de luu token

Can GOOGLE_DRIVE_FOLDER_ID trong .env.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DEFAULT_SERVICE_ACCOUNT = PROJECT_ROOT / "credentials" / "google-drive-service-account.json"
DEFAULT_OAUTH_CLIENT = PROJECT_ROOT / "credentials" / "google-oauth-client.json"
DEFAULT_OAUTH_TOKEN = PROJECT_ROOT / "credentials" / "google-drive-token.json"

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _parse_drive_time(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        return None


def _resolve_oauth_client_path() -> Optional[Path]:
    custom = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_FILE", "").strip()
    if custom and Path(custom).exists():
        return Path(custom)
    if DEFAULT_OAUTH_CLIENT.exists():
        return DEFAULT_OAUTH_CLIENT
    return None


def _resolve_service_account_path() -> Optional[Path]:
    creds_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "").strip()
    if creds_json:
        return Path("__inline_json__")

    creds_file = os.getenv("GOOGLE_DRIVE_CREDENTIALS_FILE", "").strip()
    if creds_file and Path(creds_file).exists():
        return Path(creds_file)
    if DEFAULT_SERVICE_ACCOUNT.exists():
        return DEFAULT_SERVICE_ACCOUNT
    return None


def detect_auth_mode() -> str:
    if _resolve_service_account_path():
        return "service_account"
    if _resolve_oauth_client_path():
        if DEFAULT_OAUTH_TOKEN.exists():
            return "oauth"
        return "oauth_needs_login"
    return "none"


def get_drive_sync_status() -> Dict[str, Any]:
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    mode = detect_auth_mode()
    configured = bool(folder_id and mode in ("service_account", "oauth"))
    return {
        "configured": configured,
        "folder_id_set": bool(folder_id),
        "auth_mode": mode,
        "credentials_found": mode != "none",
        "oauth_token_ready": DEFAULT_OAUTH_TOKEN.exists(),
        "credentials_hint": str(DEFAULT_OAUTH_CLIENT if mode.startswith("oauth") else DEFAULT_SERVICE_ACCOUNT),
    }


def _load_service_account_credentials():
    from google.oauth2 import service_account

    creds_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "").strip()
    if creds_json:
        info = json.loads(creds_json)
        return service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)

    path = _resolve_service_account_path()
    if path and path.exists() and str(path) != "__inline_json__":
        return service_account.Credentials.from_service_account_file(str(path), scopes=DRIVE_SCOPES)

    raise FileNotFoundError("Khong tim thay service account credentials.")


def _load_oauth_credentials(*, allow_interactive: bool = False):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    client_path = _resolve_oauth_client_path()
    if not client_path:
        raise FileNotFoundError(
            "Chua co OAuth client JSON. Dat file tai credentials/google-oauth-client.json"
        )

    creds = None
    if DEFAULT_OAUTH_TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(DEFAULT_OAUTH_TOKEN), DRIVE_SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        DEFAULT_OAUTH_TOKEN.write_text(creds.to_json(), encoding="utf-8")
        return creds

    if allow_interactive:
        return _oauth_flow(client_path)

    raise FileNotFoundError(
        "Chua dang nhap Google Drive. Chay: "
        "D:\\NCKH\\venv\\Scripts\\python.exe D:\\NCKH\\scripts\\google_drive_auth.py"
    )


def _oauth_flow(client_path: Path):
    from google_auth_oauthlib.flow import InstalledAppFlow

    DEFAULT_OAUTH_TOKEN.parent.mkdir(parents=True, exist_ok=True)
    flow = InstalledAppFlow.from_client_secrets_file(str(client_path), DRIVE_SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)
    DEFAULT_OAUTH_TOKEN.write_text(creds.to_json(), encoding="utf-8")
    return creds


def authorize_oauth_interactive() -> str:
    client_path = _resolve_oauth_client_path()
    if not client_path:
        raise FileNotFoundError(f"Khong tim thay {DEFAULT_OAUTH_CLIENT}")
    _oauth_flow(client_path)
    return str(DEFAULT_OAUTH_TOKEN)


def _load_credentials(*, allow_interactive: bool = False):
    mode = detect_auth_mode()
    if mode == "service_account":
        return _load_service_account_credentials()
    if mode.startswith("oauth"):
        return _load_oauth_credentials(allow_interactive=allow_interactive)
    raise FileNotFoundError(
        "Chua cau hinh Google Drive. Can service account JSON hoac OAuth client JSON."
    )


def _build_drive_service():
    from googleapiclient.discovery import build

    creds = _load_credentials(allow_interactive=False)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_files_recursive(service, folder_id: str) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                pageToken=page_token,
                pageSize=200,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

        for item in response.get("files", []):
            mime = item.get("mimeType", "")
            if mime == "application/vnd.google-apps.folder":
                collected.extend(_list_files_recursive(service, item["id"]))
                continue
            name = item.get("name", "")
            if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            collected.append(item)

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return collected


def _download_file(service, file_id: str, dest: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dest.write_bytes(buffer.getvalue())


def sync_from_drive(
    data_dir: Optional[str | Path] = None,
    overwrite_if_newer: bool = True,
) -> Dict[str, Any]:
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    if not folder_id:
        raise ValueError(
            "GOOGLE_DRIVE_FOLDER_ID chua cau hinh. "
            "Lay ID tu URL Drive: https://drive.google.com/drive/folders/<FOLDER_ID>"
        )

    status = get_drive_sync_status()
    if not status["configured"]:
        if status["auth_mode"] == "oauth_needs_login":
            raise FileNotFoundError(
                "Da co OAuth client nhung chua dang nhap. Chay scripts/google_drive_auth.py"
            )
        raise FileNotFoundError("Google Drive chua duoc cau hinh day du.")

    if data_dir is None:
        from rag_config import PAPERS_DIR

        data_dir = os.getenv("RAG_DATA_DIR", PAPERS_DIR)

    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    service = _build_drive_service()
    remote_files = _list_files_recursive(service, folder_id)

    added = 0
    updated = 0
    skipped = 0
    changed_paths: List[str] = []
    details: List[Dict[str, str]] = []
    errors: List[str] = []

    for remote in remote_files:
        name = remote.get("name", "").strip()
        file_id = remote.get("id")
        if not name or not file_id:
            continue

        dest = target_dir / name
        remote_mtime = _parse_drive_time(remote.get("modifiedTime"))
        local_mtime = dest.stat().st_mtime if dest.exists() else None

        should_download = False
        action = "skipped"

        if not dest.exists():
            should_download = True
            action = "added"
        elif overwrite_if_newer and remote_mtime and (local_mtime is None or remote_mtime > local_mtime + 1):
            should_download = True
            action = "updated"
        else:
            skipped += 1
            continue

        if not should_download:
            skipped += 1
            continue

        try:
            _download_file(service, file_id, dest)
            changed_paths.append(str(dest))
            details.append({"name": name, "action": action})
            if action == "added":
                added += 1
            else:
                updated += 1
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    message_parts = []
    if added:
        message_parts.append(f"{added} file moi")
    if updated:
        message_parts.append(f"{updated} file cap nhat")
    if skipped:
        message_parts.append(f"{skipped} file da co san")
    if errors:
        message_parts.append(f"{len(errors)} loi")

    message = "Dong bo Drive: " + (", ".join(message_parts) if message_parts else "khong co thay doi")

    return {
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "changed_paths": changed_paths,
        "details": details,
        "errors": errors,
        "message": message,
    }