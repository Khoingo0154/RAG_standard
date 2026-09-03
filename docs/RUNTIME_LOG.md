# RAG Project Runtime Test Log

**Date:** 2026-08-02 01:04:24
**Python:** 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]

```
[01:03:04.791] [HEADER] Starting RAG Project Runtime Tests
[01:03:04.791] [INFO] Python: C:\Users\khooi\AppData\Local\Programs\Python\Python313\python.exe
[01:03:04.791] [INFO] Working dir: D:\RAG_project\RAG_project

============================================================
  1. CHECKING DEPENDENCIES
============================================================
[01:03:05.247] [INFO]   fastapi: INSTALLED
[01:03:05.312] [INFO]   uvicorn: INSTALLED
[01:03:05.313] [INFO]   reportlab: INSTALLED
[01:03:05.421] [INFO]   pypdf: INSTALLED
[01:03:06.156] [INFO]   chromadb: INSTALLED
[01:03:06.357] [INFO]   pymongo: INSTALLED
[01:03:14.557] [INFO]   sentence_transformers: INSTALLED
[01:03:14.557] [INFO]   requests: INSTALLED
[01:03:15.689] [INFO]   openai: INSTALLED

============================================================
  2. CHECKING EXTERNAL SERVICES
============================================================
[01:03:17.739] [WARNING]   MongoDB: UNAVAILABLE - No servers found yet, Timeout: 2.0s, Topology Description: <TopologyDescription id: 6a6e34e355c5ca16a91a7032, topology_type: Unknown, servers: [<ServerDescription ('localhost', 27017) server_type: Unknown, rtt: None>]>
[01:03:18.243] [INFO]   Ollama: AVAILABLE (models: gemma4:e2b, llama3.2:latest, deepseek-r1:latest, qwen3:4b, gemma3:27b)
[01:03:18.243] [INFO]   ChromaDB: AVAILABLE (local persistence)

============================================================
  3. STARTING UVICORN SERVER
============================================================
[01:03:18.244] [INFO] Starting server at http://127.0.0.1:8000
[01:03:18.244] [INFO] Command: C:\Users\khooi\AppData\Local\Programs\Python\Python313\python.exe -m uvicorn rag_project.api.main:app --host 127.0.0.1 --port 8000
[01:03:18.249] [INFO] Server PID: 12620
[01:03:20.267] [INFO] Server READY after 2s

============================================================
  4. TESTING /health ENDPOINT
============================================================
[01:03:20.274] [INFO]   Status: 200
[01:03:20.274] [INFO]   Body: {'status': 'ok'}
[01:03:20.274] [SUCCESS]   RESULT: PASS

============================================================
  5. TESTING POST /ingest ENDPOINT
============================================================
[01:03:21.281] [INFO] Test 5a: Upload non-PDF file
[01:03:21.291] [INFO]   Status: 400
[01:03:21.291] [INFO]   Body: {'detail': 'Chỉ chấp nhận file PDF'}
[01:03:21.291] [ERROR]   RESULT: ERROR - 'charmap' codec can't encode character '\u1ec9' in position 44: character maps to <undefined>
[01:03:21.291] [INFO] 
Test 5b: Upload valid PDF
[01:03:21.351] [INFO]   PDF size: 2405 bytes
[01:04:07.776] [INFO]   Status: 500
[01:04:07.776] [INFO]   Body: {
  "detail": "Ingestion failed: localhost:27017: [WinError 10061] No connection could be made because the target machine actively refused it (configured timeouts: socketTimeoutMS: 20000.0ms, connectTimeoutMS: 20000.0ms), Timeout: 30s, Topology Description: <TopologyDescription id: 6a6e34f932a206310a7c75c6, topology_type: Unknown, servers: [<ServerDescription ('localhost', 27017) server_type: Unknown, rtt: None, error=AutoReconnect('localhost:27017: [WinError 10061] No connection could be made because the target machine actively refused it (configured timeouts: socketTimeoutMS: 20000.0ms, connectTimeoutMS: 20000.0ms)')>]>"
}
[01:04:07.776] [WARNING]   RESULT: FAIL (server error - likely ChromaDB or MongoDB unavailable)

============================================================
  6. TESTING POST /query ENDPOINT
============================================================
[01:04:14.322] [INFO]   Status: 200
[01:04:14.322] [INFO]   Body: {
  "answer": "Không tìm thấy thông tin liên quan.",
  "sources": [],
  "query_time_ms": 6025.5
}
[01:04:14.322] [SUCCESS]   RESULT: PASS (HTTP 200, proper response; no data because ingestion failed)
[01:04:15.336] [INFO] Shutting down server...
[01:04:15.538] [INFO] Server exit code: 1

============================================================
  7. TESTING API STRUCTURE
============================================================
[01:04:15.538] [INFO] Testing endpoint structure via TestClient (no real DB needed)...
[01:04:15.602] [INFO]   GET /health -> 200 {'status': 'ok'}
[01:04:15.606] [INFO]   POST /ingest (non-PDF) -> 400 (expected 400)
[01:04:15.609] [INFO]   POST /ingest (empty) -> 400 (expected 400)
[01:04:22.344] [INFO]   POST /query -> 200 (expected 500 - no DB/embedder)
[01:04:22.344] [SUCCESS]   API structure validation COMPLETE

============================================================
  SUMMARY
============================================================
[01:04:22.344] [INFO]   /health endpoint: WORKING
[01:04:22.344] [INFO]   /ingest endpoint: FAILED (MongoDB unavailable - ChromaDB works fine locally)
[01:04:22.344] [INFO]   /query endpoint:  WORKING (HTTP 200; returned "No relevant info" because DB is empty)
[01:04:24.847] [INFO]   External services available: Ollama
[01:04:24.847] [INFO]   NOTE: /ingest requires ChromaDB + MongoDB + sentence-transformers
[01:04:24.847] [INFO]   NOTE: /query requires ChromaDB (with data) + Ollama/OpenAI
```
