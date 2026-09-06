# BÁO CÁO DỰ ÁN RAG

## 1. Tổng quan

### Mục đích dự án

Dự án xây dựng một hệ thống **Retrieval-Augmented Generation (RAG)** hoàn chỉnh, cho phép:
- Upload file PDF bất kỳ
- Tự động cắt nhỏ (chunk), tạo vector embedding, lưu vào vector database và metadata database
- Tìm kiếm ngữ nghĩa (semantic search) trên toàn bộ kho tài liệu
- Sinh câu trả lời bằng LLM dựa trên các đoạn văn bản truy xuất được

Dự án được thiết kế theo kiến trúc module hóa, tách biệt ingestion, retrieval và API layer để dễ bảo trì, mở rộng và kiểm thử.

### Công nghệ sử dụng

| Thành phần | Công nghệ | Mục đích |
|---|---|---|
| Ngôn ngữ | Python 3.11+ | Toàn bộ logic backend |
| Web framework | FastAPI + Uvicorn | REST API |
| PDF parsing | pypdf | Đọc và trích xuất text từ PDF |
| Chunking | Custom sliding window | Cắt văn bản thành các đoạn nhỏ |
| Embedding | `gemini-embedding-001` / sentence-transformers / OpenAI | Tạo vector cho text |
| Vector DB | ChromaDB (Persistent / Docker volume) | Lưu vector và similarity search Cosine |
| Metadata DB | MongoDB 7 | Lưu text gốc + metadata đầy đủ |
| Object Storage | MinIO | Lưu trữ file PDF gốc theo file_id |
| Query Router | `gemini-3.1-flash-lite` | Phân loại intent: Document, Chitchat, Out-of-scope |
| LLM Generation | `gemini-3.1-flash-lite` (Flash-Lite) / Ollama / OpenAI | Sinh câu trả lời từ context + trích dẫn trang PDF |
| Container | Docker + Docker Compose | Đóng gói và chạy đa service (API, Mongo, Chroma, MinIO, Telegram) |
| Live Football API | API-Football / API-Sports | Tỷ số trực tiếp, lịch thi đấu, câu lạc bộ, cầu thủ & Widgets HTML |
| Testing | pytest | Unit test & integration test (89/89 passed) |
| PDF generation (test) | ReportLab | Tạo PDF mẫu cho test |
### Sơ đồ kiến trúc (ASCII Art)

```
┌──────────────────────────────────────────────────────────────────────┐
│                           USER / CLIENT                               │
│                    (curl, Postman, Web App)                           │
└──────┬──────────────┬──────────────────────┬─────────────────────────┘
       │              │                      │
       ▼              ▼                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     FASTAPI LAYER (api/main.py)                       │
│                                                                       │
│   GET  /health    ─── health check                                    │
│   POST /ingest    ─── upload PDF → trả file_id + stats               │
│   DELETE /files/{id} ─ xóa file khỏi cả 2 DB                         │
│   POST /query     ─── câu hỏi → câu trả lời + sources                │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                                 ▼
┌──────────────────────┐          ┌──────────────────────┐
│   INGESTION MODULE   │          │  RETRIEVAL MODULE     │
│                      │          │                       │
│  pipeline.py         │          │  pipeline.py          │
│  ├─ chunker.py       │          │  ├─ retriever.py      │
│  ├─ embedder.py      │          │  └─ generator.py      │
│  └─ storage/sore.py  │          │                       │
│                      │          │  models.py            │
│  models.py           │          │  ├─ QueryRequest      │
│  ├─ FileRecord       │          │  ├─ RetrievedChunk    │
│  ├─ ChunkDoc         │          │  └─ RAGResponse       │
│  ├─ EmbeddedChunk    │          │                       │
│  └─ StoreResult      │          │                       │
└──────────┬───────────┘          └──────────┬────────────┘
           │                                 │
           ▼                                 ▼
┌──────────────────────┐          ┌──────────────────────┐
│     MongoDB           │          │     Ollama/OpenAI     │
│  (text gốc + meta)    │          │   (LLM Generation)     │
└──────────────────────┘          └──────────────────────┘
           ▲
           │
┌──────────────────────┐
│      ChromaDB         │
│  (vector + cosine)    │
└──────────────────────┘
```

---

## 2. Cấu trúc thư mục (cây thư mục đầy đủ)

```
RAG_project/
├── .env                              # Biến môi trường (chunking, embedding, DB, LLM)
├── .pytest_cache/                    # Cache của pytest
├── CauTrucProject.png                # Ảnh sơ đồ cấu trúc dự án
├── Dockerfile                        # Docker image cho API service
├── docker-compose.yml                # Orchestration: mongo + chroma + api
├── requirements.txt                  # Python dependencies
├── rag_ingestion_context.md          # Context doc cho ingestion module
├── chroma_db/
│   └── chroma.sqlite3                # ChromaDB persistent storage
├── PROJECT_REPORT.md                 # ← BÁO CÁO NÀY
│
└── rag_project/                      # Package chính
    ├── __init__.py
    │
    ├── api/                          # FastAPI REST API layer
    │   ├── __init__.py
    │   └── main.py                   # entry point: uvicorn.run(app, port=8000)
    │
    ├── shared/                       # Shared utilities
    │   ├── __init__.py
    │   └── config.py                 # Settings dataclass (auto-load từ .env)
    │
    ├── ingestion/                    # Ingestion module (PDF → store)
    │   ├── __init__.py
    │   ├── models.py                 # FileRecord, ChunkDoc, EmbeddedChunk, StoreResult
    │   ├── pipeline.py               # run_ingestion(): 4-stage pipeline
    │   ├── services/
    │   │   ├── __init__.py
    │   │   ├── chunker.py            # chunk_pdf(), _clean_text(), _is_valid_chunk()
    │   │   └── embedder.py           # BaseEmbedder, LocalEmbedder, OpenAIEmbedder, OllamaEmbedder
    │   └── storage/
    │       ├── __init__.py
    │       └── store.py              # VectorStore, MetadataStore, IngestionStore
    │
    ├── retrieval/                    # Retrieval module (query → answer)
    │   ├── __init__.py
    │   ├── models.py                 # QueryRequest, RetrievedChunk, RAGResponse
    │   ├── retriever.py              # Retriever: ChromaDB similarity search
    │   ├── generator.py              # Generator: prompt builder + LLM call
    │   └── pipeline.py               # search_and_generate(): end-to-end retrieval
    │
    ├── tests/                        # Test suite (pytest)
    │   ├── __init__.py
    │   ├── conftest.py               # _make_sample_pdf() helper
    │   ├── test_api.py               # FastAPI endpoint tests
    │   ├── test_chunker.py           # Chunking unit tests
    │   ├── test_chunker_edge.py      # Edge case chunker tests
    │   ├── test_config.py            # Settings default tests
    │   ├── test_config_env.py        # Settings env override tests
    │   ├── test_embedder.py          # Embedder unit tests + MockEmbedder
    │   ├── test_models.py            # Data model tests
    │   ├── test_pipeline.py          # Ingestion pipeline integration tests
    │   ├── test_retrieval.py         # Retrieval pipeline integration tests
    │   └── test_store.py             # Store rollback tests
    │
    └── docs/superpowers/specs/
        └── 2026-07-29-ingestion-module-design.md  # Design document
```

---

## 3. Luồng dữ liệu chi tiết

### 3.1. PDF → Chunk → Embed → Store (Ingestion Pipeline)

```
┌──────────┐     ┌─────────────┐     ┌──────────────┐     ┌────────────────────┐
│ PDF      │────▶│ chunk_pdf() │────▶│ embed_chunks()│────▶│ IngestionStore     │
│ (bytes)  │     │             │     │               │     │ .save()            │
│          │     │ Stage 1     │     │ Stage 2       │     │                    │
└──────────┘     └─────────────┘     └──────────────┘     │ 1. Chroma upsert   │
      │                 │                    │            │ 2. MongoDB insert   │
      ▼                 ▼                    ▼            │ (rollback nếu fail) │
  FileRecord      list[ChunkDoc]     list[EmbeddedChunk]  └────────┬───────────┘
  file_id=UUID    chunk_id, text,    + vector (384/1536d)          │
  filename        page_start/end     + model_name           ┌─────▼──────┐
                  chunk_index                               │ StoreResult│
                                                            └────────────┘
```

**Chi tiết từng bước:**

#### Stage 0: Tạo FileRecord
```python
file_id = str(uuid.uuid4())
file_record = FileRecord(
    file_id=file_id,
    filename="hopdong.pdf",
    minio_path=f"documents/{file_id}/hopdong.pdf",
    size_bytes=len(pdf_bytes),
)
# file_id được sinh ngay từ đầu và truyền xuyên suốt làm foreign key
```

#### Stage 1: Chunk PDF
```python
chunks = chunk_pdf(
    pdf_bytes=pdf_bytes,
    file_id=file_id,
    strategy="token",      # "token" (sliding window) hoặc "page" (1 trang = 1 chunk)
    chunk_size=1000,        # ký tự mỗi chunk
    chunk_overlap=200,      # overlap để không mất context
)
```
- Chiến lược **token** (mặc định): gom text toàn bộ PDF → sliding window cắt đều
  - Ví dụ: chunk_size=1000, overlap=200 → window 1: [0:1000], window 2: [800:1800], ...
  - `_clean_text()`: xóa header/footer "1/100", "Trang 1", gom khoảng trắng
  - `_is_valid_chunk()`: lọc chunk < 50 ký tự hoặc tỷ lệ chữ < 30%
- Chiến lược **page**: mỗi trang là 1 chunk (đơn giản, giữ cấu trúc trang)

#### Stage 2: Embed
```python
emb = get_embedder("local")  # hoặc "ollama", "openai"
embedded_chunks = emb.embed_chunks(chunks, batch_size=32)
```
- `LocalEmbedder`: sentence-transformers với `all-MiniLM-L6-v2` → vector 384 chiều
- `OllamaEmbedder`: gọi `POST /api/embed` tới server Ollama (VD: phi4-mini)
- `OpenAIEmbedder`: gọi API `text-embedding-3-small` → vector 1536 chiều
- Xử lý theo batch, có sleep 0.1s giữa các batch để tránh rate limit

#### Stage 3: Store (với Rollback)
```python
store = IngestionStore(
    vector_store=VectorStore("./chroma_db", "rag_chunks"),
    metadata_store=MetadataStore("mongodb://localhost:27017", "rag_db"),
)
store_result = store.save(embedded_chunks)
```
- **Bước 3a**: Upsert vào ChromaDB (vector + minimal metadata)
- **Bước 3b**: Insert vào MongoDB (text đầy đủ + metadata)
- **Rollback**: nếu MongoDB fail → gọi `VectorStore.delete_by_file()` để xóa Chroma, tránh split-brain

### 3.2. Advanced Retrieval Pipeline (Hybrid Search BM25 + FlashRank Reranker + Generation)

```
[1. User Query] 
       │
       ▼
[2. Query Router: gemini-3.1-flash-lite] ──(DOCUMENT_QUERY)──┐
       │ (CHITCHAT / OUT_OF_SCOPE)                           │
       ▼ (Trả lời ngay < 1.2s)                               ▼
                    [3. HYBRID RETRIEVAL (Chạy song song)]
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        [Branch A: Dense Vector]            [Branch B: Sparse BM25]
           (Độ tương đồng ngữ nghĩa)            (Khớp từ khóa chính xác)
         ChromaDB Cosine (Top-15)            BM25 Chunks Index (Top-15)
                    │                                   │
              Top-15 Chunks                       Top-15 Chunks
                    └─────────────────┬─────────────────┘
                                      ▼
                        [4. RRF FUSION & DEDUPLICATION]
                     (Gộp chung ~20-25 Chunks độc nhất)
                                      │
                                      ▼
                      [5. RERANKING STAGE (FlashRank)]
             (Mô hình Cross-Encoder tối ưu hóa qua ONNX CPU)
              Input: Mỗi cặp [Query + Chunk Text]
              Output: Điểm tương quan thực sự (0.0 ➔ 1.0)
                                      │
                                      ▼
                             [CẮT LẤY TOP 3 - 5]
                        (Loại bỏ sạch các chunk nhiễu)
                                      │
                                      ▼
                        [6. GENERATION + CITATION]
                  gemini-3.1-flash-lite đọc Top-5 tinh khiết
               ➔ Trả lời chính xác + Trích dẫn số trang PDF
```

**Chi tiết luồng thực thi:**
1. **Query Routing (`retrieval/router.py`)**: Dùng `gemini-3.1-flash-lite` phân loại câu hỏi vào 3 intents: `DOCUMENT_QUERY`, `CHITCHAT` (phản hồi ngay < 1.2s), `OUT_OF_SCOPE` (từ chối lịch sự).
2. **Cầu nối Đa ngữ (`retrieval/pipeline.py`)**: Hàm `get_search_terms()` tự động bóc tách `retrieval_query` (song ngữ) cho Vector Embedding và `english_query` (thuật ngữ tiếng Anh) cho BM25 và Reranker.
3. **Branch A - Dense Vector Search**: Quét ChromaDB lấy Top-15 chunks có độ tương đồng Cosine cao nhất.
4. **Branch B - Sparse BM25 Search (`retrieval/bm25_retriever.py`)**: Dùng `rank-bm25` lọc stopwords và tìm Top-15 chunks khớp từ khóa cứng (`"11m"`, `"Law 11"`, `"DOGSO"`).
5. **RRF Fusion (`retrieval/fusion.py`)**: Hợp nhất thứ hạng theo công thức $RRF\_Score(d) = \sum \frac{1}{60 + Rank(d)}$, khử trùng lặp `chunk_id`, gom thành 20–25 ứng viên.
6. **FlashRank Reranker (`retrieval/reranker.py`)**: Mô hình Cross-Encoder `ms-marco-MiniLM-L-12-v2` chấm điểm chéo theo công thức: $\text{Score} = 0.6 \times \text{Normalized\_RRF} + 0.4 \times \text{FlashRank\_Score}$. Cắt lấy Top-5 chunks tinh khiết nhất.
7. **LLM Generation (`retrieval/generator.py`)**: Gửi Top-5 chunks vào `gemini-3.1-flash-lite` (kèm cơ chế Auto-retry với Exponential Backoff khi gặp lỗi 429/503), tự động gắn trích dẫn số trang: `(Tham khảo tại trang X–Y của PDF...)`.
8. **Trả về:** `RAGResponse(answer, sources, query_time_ms, intent)`.
**Ví dụ prompt:**
```
Dựa vào các đoạn văn bản sau đây, hãy trả lời câu hỏi.

=== NGỮ CẢNH ===
[1] Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính...
[2] Machine Learning là một nhánh của AI, tập trung vào việc xây dựng...

=== CÂU HỎI ===
AI là gì?

Câu trả lời:
```

---

## 4. API Endpoints

### 4.1. `GET /health`

**Mục đích**: Health check, kiểm tra API đang chạy.

**Response:**
```json
{
    "status": "ok"
}
```

**Ví dụ:**
```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

---

### 4.2. `POST /ingest`

**Mục đích**: Upload file PDF, tự động chunk → embed → store. Trả về thông tin file đã xử lý.

**Method**: `POST`

**Content-Type**: `multipart/form-data`

**Request Body**: File upload với key `file`, tên file phải kết thúc bằng `.pdf`.

**Validation**:
- Chỉ chấp nhận file `.pdf` (400 nếu sai định dạng)
- File không được rỗng (400)
- File không quá **50MB** (413)

**Response (200):**
```json
{
    "status": "success",
    "file_id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "hopdong.pdf",
    "total_chunks": 42,
    "duration_seconds": 3.14
}
```

**Ví dụ:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/document.pdf"

# Response:
# {
#   "status": "success",
#   "file_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
#   "filename": "document.pdf",
#   "total_chunks": 15,
#   "duration_seconds": 2.71
# }
```

**Xử lý nội bộ:**
```python
pdf_bytes = await file.read()                    # Đọc bytes từ upload
config = IngestionConfig(                        # Tạo config từ settings
    chunk_strategy=settings.CHUNK_STRATEGY,      # "token"
    chunk_size=settings.CHUNK_SIZE,              # 1000
    chunk_overlap=settings.CHUNK_OVERLAP,        # 200
    embed_provider=settings.EMBED_PROVIDER,      # "local"
    ...
)
result = run_ingestion(pdf_bytes, file.filename, config)  # Chạy pipeline
```

---

### 4.3. `DELETE /files/{file_id}`

**Mục đích**: Xóa toàn bộ dữ liệu của một file khỏi cả ChromaDB và MongoDB.

**Method**: `DELETE`

**Path Parameters**: `file_id` (string) - UUID của file cần xóa

**Response (200):**
```json
{
    "status": "deleted",
    "file_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "deleted_chunks": 15
}
```

**Ví dụ:**
```bash
curl -X DELETE http://localhost:8000/files/a1b2c3d4-e5f6-7890-abcd-ef1234567890

# Response:
# {
#   "status": "deleted",
#   "file_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
#   "deleted_chunks": 15
# }
```

**Xử lý nội bộ:**
1. `VectorStore.delete_by_file(file_id)` → xóa khỏi ChromaDB
2. `MetadataStore.delete_by_file(file_id)` → xóa khỏi MongoDB

---

### 4.4. `POST /query`

**Mục đích**: Gửi câu hỏi, nhận câu trả lời được sinh từ LLM dựa trên các tài liệu đã ingest.

**Method**: `POST`

**Content-Type**: `application/json`

**Request Body:**
```json
{
    "query": "Trí tuệ nhân tạo là gì?",
    "file_id": null,
    "top_k": 5
}
```

| Field | Type | Required | Default | Mô tả |
|---|---|---|---|---|
| `query` | string | yes | - | Câu hỏi cần trả lời |
| `file_id` | string \| null | no | null | Lọc kết quả theo file cụ thể. null = tìm trên tất cả file |
| `top_k` | int | no | 5 | Số lượng chunk liên quan nhất cần truy xuất |

**Response (200):**
```json
{
    "answer": "Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính, tập trung vào việc tạo ra các hệ thống có khả năng thực hiện các nhiệm vụ đòi hỏi trí thông minh của con người...",
    "sources": [
        {
            "chunk_id": "c1b2c3d4-...",
            "file_id": "a1b2c3d4-...",
            "text": "Trí tuệ nhân tạo (AI) là một lĩnh vực...",
            "page_start": 1,
            "page_end": 2,
            "score": 0.9234,
            "filename": "ai_intro.pdf"
        },
        {
            "chunk_id": "d5e6f7g8-...",
            "file_id": "a1b2c3d4-...",
            "text": "Machine Learning là một nhánh của AI...",
            "page_start": 3,
            "page_end": 3,
            "score": 0.8712,
            "filename": "ai_intro.pdf"
        }
    ],
    "query_time_ms": 1250.5
}
```

| Field | Mô tả |
|---|---|
| `answer` | Câu trả lời do LLM sinh ra |
| `sources[].chunk_id` | UUID của chunk nguồn |
| `sources[].file_id` | UUID của file nguồn |
| `sources[].text` | 500 ký tự đầu của chunk (đã truncate) |
| `sources[].page_start` | Trang bắt đầu |
| `sources[].page_end` | Trang kết thúc |
| `sources[].score` | Điểm tương đồng (cosine similarity, 0-1) |
| `sources[].filename` | Tên file PDF gốc |
| `query_time_ms` | Thời gian xử lý (milliseconds) |

**Ví dụ với file filter:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "AI là gì?", "file_id": "a1b2c3d4-...", "top_k": 3}'
```


### 4.5. `GET /files`
**Mục đích**: Liệt kê danh sách tất cả `file_id` và thông tin tài liệu PDF đang lưu trữ trong MinIO.

### 4.6. `GET /widgets`
**Mục đích**: Trang giao diện trực quan HTML nhúng trực tiếp API-Sports Widgets (`leagues`, `livescore`, `config`).

### 4.7. Nhóm Endpoints Football Realtime
- `GET /football/status`: Kiểm tra hạn mức sử dụng và gói API-Sports.
- `GET /football/live`: Danh sách các trận đấu đang diễn ra trực tiếp.
- `GET /football/teams?search=...`: Tra cứu thông tin câu lạc bộ, sân vận động.
- `GET /football/players?search=...`: Tra cứu hồ sơ cầu thủ (tuổi, quốc tịch, chiều cao...).
---

## 5. Cấu hình

### `.env` file

```
# === Chunking ===
CHUNK_STRATEGY=token          # "token" hoặc "page"
CHUNK_SIZE=1000               # Ký tự mỗi chunk
CHUNK_OVERLAP=200             # Overlap giữa các chunk

# === Embedding (local | openai | ollama) ===
EMBED_PROVIDER=local          # Provider mặc định
EMBED_BATCH_SIZE=32           # Số chunk xử lý mỗi batch

# === ChromaDB ===
CHROMA_PATH=./chroma_db       # Đường dẫn lưu ChromaDB
CHROMA_COLLECTION=rag_chunks  # Tên collection

# === MongoDB ===
MONGO_URI=mongodb://localhost:27017
MONGO_DB=rag_db

# === Ollama ===
OLLAMA_BASE_URL=http://100.71.230.7:11434  # Server Ollama
OLLAMA_MODEL=phi4-mini                     # Model mặc định

# === OpenAI (optional) ===
# OPENAI_API_KEY=sk-...

# === Retrieval ===
RETRIEVAL_TOP_K=5             # Số chunk truy xuất mặc định

# === LLM Generation ===
LLM_PROVIDER=ollama           # "ollama" hoặc "openai"
LLM_OLLAMA_MODEL=phi4-mini    # Model LLM để generate
```

### Cơ chế load config

Trong `rag_project/shared/config.py`, `Settings` dataclass tự động load biến môi trường trong `__post_init__`:

```python
@dataclass
class Settings:
    CHUNK_STRATEGY: str = "token"
    CHUNK_SIZE: int = 1000
    # ...

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            env_val = os.getenv(key)
            if env_val is not None:
                if key.endswith("_SIZE") or key.endswith("_BATCH_SIZE") or key.endswith("_TOP_K"):
                    setattr(self, key, int(env_val))    # Auto cast int
                else:
                    setattr(self, key, env_val)

settings = Settings()  # Singleton, dùng toàn project
```

Biến môi trường ghi đè giá trị mặc định của dataclass. Các field kết thúc bằng `_SIZE`, `_BATCH_SIZE`, `_TOP_K` được tự động cast sang `int`.

### `docker-compose.yml`

```yaml
services:
  mongodb:
    image: mongo:7
    ports: ["27017:27017"]
    volumes: [mongo_data:/data/db]

  minio:
    image: minio/minio:latest
    ports: ["9000:9000", "9001:9001"]
    volumes: [minio_data:/data]

  api:
    build: { context: ., dockerfile: Dockerfile }
    ports: ["8000:8000"]
    environment:
      - CHROMA_PATH=/data/chroma_db
      - MONGO_URI=mongodb://mongodb:27017
      - MINIO_ENDPOINT=minio:9000
    volumes: [chroma_db:/data/chroma_db]
    depends_on: [mongodb, minio]

---

## 6. Cách chạy dự án

### Bước 1: Cài dependencies

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
# source venv/bin/activate    # Linux/macOS

# Cài dependencies
pip install -r requirements.txt
```

Danh sách dependencies chính:
| Package | Mục đích |
|---|---|
| `pypdf>=4.0.0` | Đọc PDF |
| `openai>=1.0.0` | OpenAI embed + LLM |
| `sentence-transformers>=2.0.0` | Local embedding |
| `chromadb>=0.5.0` | Vector database |
| `pymongo>=4.0.0` | MongoDB client |
| `fastapi>=0.100.0` | Web framework |
| `uvicorn[standard]>=0.20.0` | ASGI server |
| `python-multipart>=0.0.6` | File upload |
| `requests>=2.31.0` | HTTP calls (Ollama) |
| `reportlab>=4.0.0` | PDF generation (test) |
| `pytest>=7.0.0` | Testing |

### Bước 2: Chạy MongoDB + ChromaDB

```bash
# Chạy cả MongoDB và ChromaDB bằng Docker
docker-compose up -d mongodb chroma
```

Hoặc chạy thủ công:
```bash
# MongoDB
docker run -d --name rag_mongo -p 27017:27017 mongo:7

# ChromaDB
docker run -d --name rag_chroma -p 8001:8000 \
  -e IS_PERSISTENT=TRUE \
  -v chroma_data:/chroma/chroma \
  chromadb/chroma:latest
```

### Bước 3: Chạy API

```bash
# Cách 1: Chạy trực tiếp
python -m rag_project.api.main

# Cách 2: Chạy bằng uvicorn
uvicorn rag_project.api.main:app --host 0.0.0.0 --port 8000 --reload

# Cách 3: Chạy toàn bộ stack bằng Docker
docker-compose up -d
```

Output khi chạy thành công:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Bước 4: Gọi API

```bash
# 1. Kiểm tra health
curl http://localhost:8000/health

# 2. Upload file PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@D:\path\to\document.pdf"

# 3. Đặt câu hỏi
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Nội dung chính của tài liệu là gì?"}'

# 4. Câu hỏi với file filter
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Điều khoản thanh toán", "file_id": "<FILE_ID>", "top_k": 3}'

# 5. Xóa file
curl -X DELETE http://localhost:8000/files/<FILE_ID>
```

---

## 7. Cách test

### Chạy toàn bộ test suite

```bash
cd D:\RAG_project\RAG_project
pytest rag_project/tests/ -v
```

### Cấu trúc test

```
rag_project/tests/
├── conftest.py           # _make_sample_pdf(): tạo PDF mẫu bằng ReportLab
├── test_models.py        # Kiểm tra dataclass: FileRecord, ChunkDoc, EmbeddedChunk, StoreResult
├── test_chunker.py       # Test chunking strategies (page, token), clean text, valid chunk
├── test_chunker_edge.py  # Edge cases: empty text, đúng/sai 50 ký tự, cross-page range
├── test_embedder.py      # MockEmbedder test + get_embedder() factory
├── test_store.py         # IngestionStore: save success, rollback MongoDB fail, Chroma fail
├── test_config.py        # Settings default values
├── test_config_env.py    # Settings env variable override + auto int cast
├── test_pipeline.py      # Integration: full ingestion pipeline với mock store
├── test_retrieval.py     # Integration: retriever search, generator prompt, full retrieval pipeline
└── test_api.py           # API endpoints: health, ingest (valid/invalid), query (with/without filter)
```

### Test coverage theo module

| Module | Số test case | Nội dung test chính |
|---|---|---|
| Models | 4 | FileRecord, ChunkDoc.char_count auto, EmbeddedChunk.from_chunk, StoreResult |
| Chunker | 8 | Page/token strategy, overlap validation, text cleaning, valid chunk |
| Chunker Edge | 7 | Empty text, whitespace, boundary crossing, exact 50 chars |
| Embedder | 5 | Mock embedder, metadata preservation, empty list, factory |
| Store | 4 | Save success, MongoDB rollback, empty chunks, Chroma failure |
| Config | 2 | Default values, custom values |
| Config Env | 4 | String override, int auto-cast, TOP_K, partial override |
| Pipeline | 3 | Full mock pipeline, empty PDF error, config defaults |
| Retrieval | 8 | Models, retriever search (with/without filter), generator prompt, ollama mock, full pipeline |
| API | 6 | Health, ingest rejection (non-PDF, empty), ingest success mock, query success, query with filter |
| **Total** | **51** | |

### Ví dụ output pytest

```
rag_project/tests/test_chunker.py::TestChunker::test_chunk_by_page_basic PASSED
rag_project/tests/test_chunker.py::TestChunker::test_chunk_by_token_sliding_window PASSED
rag_project/tests/test_chunker.py::TestChunker::test_chunk_overlap_validation PASSED
rag_project/tests/test_store.py::TestIngestionStore::test_save_success PASSED
rag_project/tests/test_store.py::TestIngestionStore::test_save_rollback_on_mongo_failure PASSED
rag_project/tests/test_api.py::TestAPI::test_health_endpoint PASSED
rag_project/tests/test_api.py::TestAPI::test_ingest_pdf_success PASSED
...
========================= 51 passed in 2.38s =========================
```

---

## 8. Điểm mạnh

1. **Kiến trúc module hóa rõ ràng**
   - Tách biệt ingestion, retrieval, API thành các module độc lập
   - Ingestion và retrieval là pure Python, không phụ thuộc FastAPI → có thể reuse trong script, CLI, worker

2. **Abstract BaseEmbedder interface**
   - Dễ dàng swap giữa Local/OpenAI/Ollama embedder chỉ bằng config
   - Không cần sửa code pipeline khi đổi embedder

3. **Rollback strategy trong Store**
   - Chroma upsert trước, MongoDB insert sau
   - Nếu MongoDB fail → tự động xóa Chroma để tránh split-brain
   - Đủ tốt cho dev/staging environment

4. **Text cleaning khi chunk**
   - Tự động xóa header/footer PDF ("Trang 1", "1/100", "Page 1")
   - Lọc chunk rác (< 50 ký tự, tỷ lệ chữ < 30%)
   - Gom nhiều dòng trống, chuẩn hóa whitespace

5. **Dual database strategy**
   - ChromaDB: lưu vector + minimal metadata (tối ưu cho similarity search)
   - MongoDB: lưu text gốc đầy đủ (tối ưu cho đọc và filter)
   - Cùng `file_id` làm foreign key, liên kết 2 DB

6. **Config linh hoạt qua `.env` + dataclass**
   - Settings tự động load biến môi trường, auto cast int
   - Hỗ trợ cả giá trị mặc định và override

7. **Docker Compose orchestrator**
   - Chạy full stack với 1 lệnh: `docker-compose up -d`
   - Volume persist data cho cả MongoDB và ChromaDB

8. **Test coverage tốt (51 test cases)**
   - Unit test từng hàm nhỏ
   - Integration test full pipeline
   - API endpoint test với TestClient
   - Mock triệt để dependency ngoài (ChromaDB, MongoDB, Ollama)

9. **Hỗ trợ file filter trong query**
   - Có thể giới hạn truy vấn trong 1 file cụ thể qua `file_id`
   - Hữu ích khi có nhiều tài liệu khác chủ đề

---

## 9. Hạn chế / Cần cải thiện

1. **Chưa hỗ trợ file format khác ngoài PDF**
   - Chỉ hỗ trợ `.pdf`. Chưa có docx, txt, html, csv...

2. **Chưa có MinIO integration thực sự**
   - `FileRecord.minio_path` chỉ là placeholder string, chưa upload file thật lên object storage

3. **Chunking tương đối đơn giản**
   - Strategy "token" là character-based sliding window, KHÔNG phải token-aware (không dựa trên BPE tokenizer như tiktoken)
   - Có thể bị cắt giữa câu nếu text không có dấu câu rõ ràng
   - Nên bổ sung sentence-based chunking hoặc semantic chunking

4. **OllamaEmbedder latency có thể cao**
   - Gọi API đồng bộ qua `requests`, không có retry, không có connection pool
   - Batch size mặc định 32 có thể không tối ưu cho model phi4-mini

5. **Chưa có streaming response cho `/query`**
   - LLM generate xong mới trả kết quả, người dùng phải chờ
   - Nên hỗ trợ Server-Sent Events (SSE) để stream từng token

6. **Chưa có authentication / authorization**
   - API mở hoàn toàn, CORS allow all origins
   - Không có API key, JWT, rate limiting

7. **Prompt template cứng**
   - Prompt trong `generator.py` là hard-coded tiếng Việt
   - Chưa hỗ trợ custom system prompt hoặc prompt template qua config

8. **Hệ thống Evaluation & Benchmark đã hoàn thiện (Mới cập nhật):**
   - Đã xây dựng `evals/metrics.py` và lớp OOP `RAGBenchmark` trong `evals/benchmark.py`.
   - Hỗ trợ đo đạc đầy đủ: Recall@K, Precision@K, MRR, Hit@K, Latency, Fact Coverage và Citation Faithfulness.
   - Hỗ trợ xuất đồng thời 3 định dạng: JSON, CSV (Excel), Markdown và có shortcut 1-click `run_benchmark.cmd`.
9. **Logging còn basic**
   - Chỉ dùng `logging.basicConfig`, chưa có structured logging
   - Chưa có request_id tracing xuyên suốt

10. **Chưa hỗ trợ metadata filter nâng cao**
    - Retriever chỉ hỗ trợ filter theo `file_id`, chưa có filter theo page range, date, tag...

---

## 10. Hướng phát triển tương lai

### Ngắn hạn (1-2 tuần)

1. **Hỗ trợ thêm file format**: docx (python-docx), txt, markdown, html
2. **Cải thiện chunking**: thêm sentence-based strategy dùng spaCy hoặc NLTK sentence tokenizer
3. **Streaming response SSE**: `/query` stream từng token qua Server-Sent Events
4. **Thêm health check cho MongoDB và ChromaDB** trong `/health` endpoint
5. **Thêm endpoint `GET /files`**: liệt kê danh sách file đã ingest, kèm thông tin (số chunk, ngày upload)

### Trung hạn (1-2 tháng)

6. **MinIO integration**: upload PDF bytes thực sự lên MinIO, dùng presigned URL khi cần đọc lại
7. **Semantic chunking**: dùng embedding similarity để xác định ranh giới chunk tự nhiên
8. **HyDE (Hypothetical Document Embeddings)**: sinh câu trả lời giả định trước khi search → tăng chất lượng retrieval
9. **Reranking**: thêm bước re-rank bằng cross-encoder (VD: `ms-marco-MiniLM`) sau khi ChromaDB search
10. **Authentication**: JWT-based auth + API key management
11. **Rate limiting**: giới hạn số request/giây theo IP hoặc user

### Dài hạn (3-6 tháng)

12. **Multimodal RAG**: hỗ trợ ảnh, bảng biểu trong PDF (dùng Unstructured.io hoặc LlamaParse)
13. **Agentic RAG**: cho phép LLM tự quyết định cần search gì, search mấy lần (multi-hop retrieval)
14. **Evaluation pipeline**: tích hợp RAGAS hoặc DeepEval để đo recall, precision, faithfulness, answer relevancy
15. **Vector DB alternatives**: hỗ trợ thêm Qdrant, Milvus, Weaviate bên cạnh ChromaDB
16. **Knowledge graph integration**: trích xuất entities + relationships, kết hợp graph traversal với vector search
17. **Conversational RAG**: hỗ trợ hội thoại nhiều lượt với chat history, context window management
18. **Admin dashboard**: web UI để upload file, xem lịch sử query, stats (FastAPI + React/Vue)
