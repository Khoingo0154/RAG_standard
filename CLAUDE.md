# CLAUDE.md — RAG Project

## Dự án
RAG (Retrieval-Augmented Generation) system: ingest PDF → chunk → embed → ChromaDB + MongoDB → query → generate answer bằng Ollama/OpenAI.

## Kiến trúc
```
rag_project/
├── ingestion/     # PDF → chunk → embed → store (ChromaDB + MongoDB, có rollback)
├── retrieval/     # query → search Chroma → generate answer (Ollama/OpenAI)
├── api/           # FastAPI: POST /ingest, POST /query, DELETE /files/{id}, GET /health
├── shared/        # config (Settings dataclass, auto-load .env)
└── tests/         # 57 tests / 8 file, tất cả dùng mock
```

## Cách chạy
```bash
# Cài dependencies
pip install -r requirements.txt

# Khởi động MongoDB (bắt buộc cho ingestion)
docker run -d -p 27017:27017 --name rag_mongo mongo:7
# Hoặc docker-compose up -d

# Chạy API
python -m uvicorn rag_project.api.main:app --host 0.0.0.0 --port 8000

# Chạy tests
pytest rag_project/tests/ -v
```

## API Endpoints
- `GET /health` → `{"status":"ok"}`
- `POST /ingest` (form-data: file=@doc.pdf) → `{"status":"success","file_id":"...","total_chunks":N}`
- `DELETE /files/{file_id}` → `{"status":"deleted","deleted_chunks":N}`
- `POST /query` (json: {"query":"..."}) → `{"answer":"...","sources":[...],"query_time_ms":N}`

## State hiện tại (2026-08-02)
- **Code**: hoàn chỉnh 4 module + docker-compose + Dockerfile + .env
- **Tests**: 57/57 pass
- **Runtime**: server chạy được, health + query OK. Ingestion fail vì MongoDB chưa chạy trên máy
- **Ollama**: khả dụng tại `100.71.230.7:11434` (models: gemma4, llama3.2, deepseek-r1, qwen3, gemma3)

## Docs quan trọng
| File | Nội dung |
|---|---|
| `docs/PROJECT_REPORT.md` | Báo cáo tổng quan 10 phần (~420 dòng) |
| `docs/EXECUTION_TRACE.md` | Trace từng bước thực thi mọi flow |
| `docs/RUNTIME_LOG.md` | Log runtime thực tế |
| `docs/superpowers/specs/*.md` | Design spec ingestion |
| `docs/rag_ingestion_context.md` | Context thiết kế ban đầu |

## Cách tiếp tục
Khi mở lại terminal, nói: "Đọc CLAUDE.md và docs/PROJECT_REPORT.md, tiếp tục phát triển RAG project"

## TODO / Hướng phát triển
1. Chạy MongoDB để hoàn tất integration test
2. Thêm streaming response cho /query (SSE)
3. Thêm authentication
4. Hỗ trợ file format khác (DOCX, TXT, HTML)
5. Dedup file upload
6. Tách embedder ra shared/ để retrieval không import từ ingestion
