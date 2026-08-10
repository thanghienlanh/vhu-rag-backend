---
title: VHU Document Assistant API
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# VHU Document Assistant — Backend API

FastAPI backend for a RAG (Retrieval-Augmented Generation) system answering questions
about Van Hien University (VHU) official student notices, built from PDF documents in
`rag_system/papers/`.

- Embedding: `BAAI/bge-m3` (local, CPU)
- Vector store: FAISS (rebuilt automatically on first boot if missing)
- LLM: Gemini API (`GEMINI_MODEL`, requires `GEMINI_API_KEY` secret)
- Frontend (deployed separately, e.g. on Vercel) talks to this API; set `CORS_ORIGINS`
  to the frontend's origin.

Required Space secrets: `GEMINI_API_KEY`, `MONGODB_URI`, `MONGODB_PASSWORD`,
`JWT_SECRET_KEY`, `CORS_ORIGINS`.
