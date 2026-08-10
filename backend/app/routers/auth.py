import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from pydantic import BaseModel, Field

from database import connect_db, get_db
from services.user_service import authenticate_user, create_user, get_user_by_username, seed_default_admin

router = APIRouter()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-key-for-vhu-demo-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24 * 7)))
ALLOW_PUBLIC_REGISTER = os.getenv("ALLOW_PUBLIC_REGISTER", "false").strip().lower() == "true"
LOCAL_AUTH_FALLBACK = os.getenv("LOCAL_AUTH_FALLBACK", "true").strip().lower() == "true"
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserPublic(BaseModel):
    username: str
    email: Optional[str] = None
    role: str = "user"


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)
    email: str = ""


class AdminCreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)
    email: str = ""
    role: str = "user"


class AuthStatusResponse(BaseModel):
    provider: str = "mongodb"
    connected: bool
    database: str
    public_register: bool = False
    local_fallback: bool = False


def _verify_local_login(username: str, password: str) -> Optional[UserPublic]:
    if username.strip().lower() == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return UserPublic(username=ADMIN_USERNAME, role="admin")
    return None


def _user_from_token_payload(payload: dict) -> Optional[UserPublic]:
    username = payload.get("sub")
    if not username:
        return None
    if not LOCAL_AUTH_FALLBACK:
        return None
    return UserPublic(
        username=username,
        role=payload.get("role", "user"),
    )


async def _ensure_db_ready():
    try:
        get_db()
    except RuntimeError:
        await connect_db()
        await seed_default_admin()


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


@router.get("/status", response_model=AuthStatusResponse)
async def auth_status():
    from database import MONGODB_DB

    try:
        await _ensure_db_ready()
        return AuthStatusResponse(
            connected=True,
            database=MONGODB_DB,
            public_register=ALLOW_PUBLIC_REGISTER,
        )
    except Exception:
        return AuthStatusResponse(
            connected=False,
            database=MONGODB_DB,
            public_register=ALLOW_PUBLIC_REGISTER,
            local_fallback=LOCAL_AUTH_FALLBACK,
            provider="local-fallback" if LOCAL_AUTH_FALLBACK else "mongodb",
        )


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserPublic:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(401, "Invalid token")

    username: str = payload.get("sub")
    if not username:
        raise HTTPException(401, "Invalid token")

    if LOCAL_AUTH_FALLBACK:
        fallback_user = _user_from_token_payload(payload)
        if fallback_user:
            return fallback_user

    try:
        await _ensure_db_ready()
        user = await get_user_by_username(username)
        if user and user.get("is_active", True):
            return UserPublic(
                username=user["username"],
                email=user.get("email"),
                role=user.get("role", "user"),
            )
    except Exception as exc:
        fallback_user = _user_from_token_payload(payload)
        if fallback_user:
            print(f"[Auth] MongoDB unavailable, accepting local token for {username}: {exc}")
            return fallback_user
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    fallback_user = _user_from_token_payload(payload)
    if fallback_user:
        return fallback_user

    raise HTTPException(401, "Invalid token")


async def require_admin(current_user: UserPublic = Depends(get_current_user)) -> UserPublic:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Chỉ admin mới được phép thực hiện thao tác này.")
    return current_user


@router.post("/register", response_model=UserPublic)
async def register(req: RegisterRequest):
    if not ALLOW_PUBLIC_REGISTER:
        raise HTTPException(
            status_code=403,
            detail="Đăng ký công khai đã tắt. Vui lòng liên hệ admin để được cấp tài khoản.",
        )
    try:
        await _ensure_db_ready()
        user = await create_user(req.username, req.password, req.email, role="user")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        hint = (
            f" Không kết nối được MongoDB Atlas. Dùng tài khoản demo: {ADMIN_USERNAME}/{ADMIN_PASSWORD}."
            if LOCAL_AUTH_FALLBACK
            else ""
        )
        raise HTTPException(status_code=503, detail=f"Database error: {exc}.{hint}")

    return UserPublic(username=user["username"], email=user.get("email"), role=user.get("role", "user"))


@router.post("/users", response_model=UserPublic)
async def admin_create_user(req: AdminCreateUserRequest, _: UserPublic = Depends(require_admin)):
    """Admin tạo tài khoản mới (user hoặc admin)."""
    await _ensure_db_ready()
    role = req.role if req.role in {"admin", "user"} else "user"
    try:
        user = await create_user(req.username, req.password, req.email, role=role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}")

    return UserPublic(username=user["username"], email=user.get("email"), role=user.get("role", "user"))


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Fast path: tránh chờ MongoDB timeout khi Atlas lỗi (benchmark / demo local)
    if LOCAL_AUTH_FALLBACK:
        local_user = _verify_local_login(form_data.username, form_data.password)
        if local_user:
            token = create_access_token(
                data={"sub": local_user.username, "role": local_user.role},
                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            )
            return {"access_token": token, "token_type": "bearer"}

    user: Optional[UserPublic] = None
    try:
        await _ensure_db_ready()
        db_user = await authenticate_user(form_data.username, form_data.password)
        if db_user:
            user = UserPublic(
                username=db_user["username"],
                email=db_user.get("email"),
                role=db_user.get("role", "user"),
            )
    except Exception as exc:
        if not LOCAL_AUTH_FALLBACK:
            raise HTTPException(status_code=503, detail=f"Database error: {exc}")
        print(f"[Auth] MongoDB login failed, trying local fallback: {exc}")

    if user is None and LOCAL_AUTH_FALLBACK:
        user = _verify_local_login(form_data.username, form_data.password)

    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}