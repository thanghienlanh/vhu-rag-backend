import os
import uvicorn
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

if __name__ == "__main__":
    # reload=True gây ConnectionReset giữa request dài (RAG + Ollama) → đáp án trống trên UI
    reload = os.getenv("UVICORN_RELOAD", "false").strip().lower() == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=reload)