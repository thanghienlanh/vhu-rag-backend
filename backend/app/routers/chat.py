from fastapi import APIRouter, HTTPException, Depends
import asyncio
import os
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from datetime import datetime
from services.rag_service import (
    ask_question,
    ask_question_async,
    PureGenerationTimeout,
    stream_answer_with_context,
    get_status,
    get_sources_for_question,
    get_response_mode_for_question,
)
from routers.auth import get_current_user
import json

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    use_hybrid: Optional[bool] = None
    filter_source: Optional[str] = None
    history: Optional[List[dict]] = None  # Recent chat messages for context-aware rewriting
    fast_mode: Optional[bool] = False     # Faster but may be less accurate (disables rewrite + rerank)

@router.post("/")
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    try:
        async with asyncio.timeout(float(os.getenv("PURE_RAG_REQUEST_TIMEOUT", "240"))):
            result = await ask_question_async(
                question=request.question,
                use_hybrid=request.use_hybrid,
                filter_source=request.filter_source,
                history=request.history,
                fast_mode=request.fast_mode,
            )
    except PureGenerationTimeout as exc:
        raise HTTPException(status_code=504, detail={"code": str(exc)}) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail={"code": "request_timeout"}) from exc
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "mode": result.get("mode", "rag"),
        "technical_status": result.get("technical_status"),
    }

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Real-time streaming with retrieved context. First sends sources, then answer chunks."""
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    async def generate():
        try:
            result = await ask_question_async(
                request.question,
                use_hybrid=request.use_hybrid,
                filter_source=request.filter_source,
                history=request.history,
                fast_mode=request.fast_mode,
            )
            yield f"data: {json.dumps({'sources': result.get('sources', []), 'mode': result.get('mode', 'rag')})}\n\n"
            answer = result.get("answer", "")
            had_chunk = bool(answer)
            if answer:
                yield f"data: {json.dumps({'chunk': answer})}\n\n"
            if not had_chunk:
                fallback = (
                    "Không nhận được câu trả lời từ LLM. "
                    "Kiểm tra Ollama đang chạy (`ollama serve`) và thử gửi lại câu hỏi."
                )
                yield f"data: {json.dumps({'chunk': fallback})}\n\n"
            yield "data: [DONE]\n\n"
        except PureGenerationTimeout as exc:
            yield f"data: {json.dumps({'error': {'code': str(exc)}})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@router.get("/status")
async def status():
    return get_status()

class FeedbackRequest(BaseModel):
    message_index: int
    feedback: str  # "up" or "down"
    question: str
    answer: str

FEEDBACK_FILE = Path(__file__).parent.parent.parent.parent / "data" / "feedback.jsonl"

@router.post("/feedback")
async def feedback(req: FeedbackRequest, user = Depends(get_current_user)):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "username": user.username,
        "message_index": req.message_index,
        "feedback": req.feedback,
        "question": req.question,
        "answer": req.answer,
    }
    print(f"[FEEDBACK] User {user.username} | Index {req.message_index} | {req.feedback}")
    try:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[FEEDBACK] Could not persist: {exc}")
    return {"status": "recorded", "message": "Cảm ơn phản hồi của bạn!"}


class RewriteResponse(BaseModel):
    original: str
    rewritten: str


@router.post("/rewrite", response_model=RewriteResponse)
async def rewrite_question(request: ChatRequest):
    """Debug / demo endpoint to see what query rewriting does (with history)."""
    from services.rag_service import rewrite_query
    rewritten = rewrite_query(request.question, request.history)
    return {"original": request.question, "rewritten": rewritten}
