FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY rag_system/requirements.txt /app/rag_system/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir -r /app/rag_system/requirements.txt \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

COPY rag_system /app/rag_system
COPY backend /app/backend

ENV EMBEDDING_DEVICE=cpu \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=7860

RUN mkdir -p /app/.cache/huggingface /app/logs && chmod -R 777 /app

WORKDIR /app/backend
EXPOSE 7860

CMD ["python", "-c", "import os,uvicorn; import sys; sys.path.insert(0,'app'); uvicorn.run('main:app', host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))"]
