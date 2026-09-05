# EXECUTION TRACE — RAG Project

> Trace chi tiết luồng thực thi toàn bộ dự án RAG.
> Mỗi bước ghi rõ: file path + line number, function signature, input/output ví dụ, thời gian ước tính.

---

## 0. CONFIG FLOW — Khởi tạo Settings (global singleton)

Diễn ra MỘT LẦN duy nhất khi module `shared.config` được import lần đầu.

```
shared/config.py:34  Settings.__post_init__()
                     ├── duyệt từng field trong __dataclass_fields__
                     ├── gọi os.getenv(key) để kiểm tra biến môi trường
                     └── nếu có env var → ghi đè default
shared/config.py:44  settings = Settings()   ← singleton, import 1 lần dùng khắp project
```

| Step | File:Line | Hàm / Logic | Input | Output |
|------|-----------|-------------|-------|--------|
| 0.1 | `shared/config.py:34` | `Settings.__post_init__()` | `self` (dataclass instance) | `None` (side-effect: setattr) |
| 0.1a | `shared/config.py:35-36` | `self.__dataclass_fields__` (iter fields) | 20 fields (CHUNK_STRATEGY, CHUNK_SIZE, ...) | từng field được duyệt |
| 0.1b | `shared/config.py:37` | `os.getenv("CHUNK_SIZE")` | `"2000"` (nếu có file .env) hoặc `None` | `"2000"` (str) hoặc `None` |
| 0.1c | `shared/config.py:38-41` | `setattr(self, key, value)` | key=`"CHUNK_SIZE"`, value=`2000` (int) | self.CHUNK_SIZE = 2000 |
| 0.2 | `shared/config.py:44` | `settings = Settings()` | không | `Settings` instance, all fields populated |

**Ví dụ thực tế:**
- Nếu không có env vars → `settings.CHUNK_SIZE = 1000`, `settings.EMBED_PROVIDER = "local"`, `settings.OLLAMA_BASE_URL = "http://100.71.230.7:11434"`
- Nếu có `export CHUNK_SIZE=2000 EMBED_PROVIDER=ollama` → override tương ứng

**Thời gian:** ~1ms (chỉ đọc env vars, không I/O network)

**Ghi chú:** `__post_init__` chỉ xử lý kiểu string và int (fields kết thúc bằng `_SIZE`, `_BATCH_SIZE`, `_TOP_K` được cast sang int). Các field kiểu khác (vd Optional[str]) sẽ ghi đè bằng string từ env.

---

## 1. INGESTION FLOW

### 1.0 TỔNG QUAN CÂY GỌI HÀM

```
POST /ingest  (api/main.py:32)
│
├── 1.0  await file.read()                          → pdf_bytes: bytes
├── 1.1  validate: check .pdf extension, size ≤ 50MB, non-empty
├── 1.2  IngestionConfig(...)                        → config: IngestionConfig
│         └── đọc settings singleton đã khởi tạo ở flow 0
│
└── 1.3  run_ingestion(pdf_bytes, filename, config)  → IngestionResult
          │
          ├── 1.3a  uuid.uuid4()                     → file_id: str (UUID)
          ├── 1.3b  FileRecord(...)                   → file_record: FileRecord
          │
          ├── 1.3c  chunk_pdf(pdf_bytes, file_id, strategy, chunk_size, overlap)
          │         │                                   → list[ChunkDoc]
          │         ├── PdfReader(io.BytesIO(pdf_bytes))
          │         ├── _extract_pages(reader)          → list[dict]
          │         │    └── per page: extract_text() → _clean_text(raw) → {"page_num", "text"}
          │         └── strategy switch:
          │              ├── "page": _chunk_by_page(pages, file_id)
          │              │    └── per page: _is_valid_chunk(text) → ChunkDoc(...)
          │              └── "token": _chunk_by_token(pages, file_id, size, overlap)
          │                   ├── gom text → full_text + boundaries
          │                   ├── sliding window (step = chunk_size - chunk_overlap)
          │                   └── _find_page_range(pos, end_pos, boundaries)
          │
          ├── 1.3d  get_embedder(cfg.embed_provider)  → embedder: BaseEmbedder
          │         └── factory: "local"→LocalEmbedder | "ollama"→OllamaEmbedder | "openai"→OpenAIEmbedder
          │
          ├── 1.3e  embedder.embed_chunks(chunks, batch_size)
          │         │                                   → list[EmbeddedChunk]
          │         ├── loop: for i in range(0, len(chunks), batch_size)
          │         ├── self._embed_batch([c.text for c in batch])  → list[list[float]]
          │         │    └── provider-specific:
          │         │         LocalEmbedder:  SentenceTransformer.encode(texts) → numpy → .tolist()
          │         │         OllamaEmbedder: POST /api/embed {model, input}
          │         │         OpenAIEmbedder: client.embeddings.create(model, input)
          │         └── EmbeddedChunk.from_chunk(chunk, vector, model_name)
          │
          └── 1.3f  IngestionStore.save(embedded_chunks)
                    │                                   → StoreResult
                    ├── VectorStore.upsert(chunks)       → chroma_ids: list[str]
                    │    └── self._collection.upsert(ids, embeddings, documents, metadatas)
                    ├── MetadataStore.insert_chunks(chunks)  → mongo_ids: list[str]
                    │    └── self._col.insert_many(docs, ordered=False)
                    └── [ROLLBACK nếu MongoDB fail]
                         └── VectorStore.delete_by_file(file_id)
                              └── self._collection.get(where=...) → delete(ids=...)
```

### 1.1 API Layer: POST /ingest

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 1.1a | `api/main.py:32` | `async def ingest(file: UploadFile)` | `file`: FastAPI UploadFile (.pdf) | HTTP JSON response | ~tổng pipeline |
| 1.1b | `api/main.py:34` | validate filename | `file.filename = "hopdong.pdf"` | pass hoặc HTTPException(400) | <1ms |
| 1.1c | `api/main.py:38` | `await file.read()` | — | `pdf_bytes = b'%PDF-1.4\n...'` (~500KB) | ~10-50ms (I/O) |
| 1.1d | `api/main.py:42` | check `len(pdf_bytes) == 0` | `pdf_bytes = b''` | HTTPException(400) "File rỗng" | <1ms |
| 1.1e | `api/main.py:45` | check `len(pdf_bytes) > 52428800` | `pdf_bytes` (55MB) | HTTPException(413) "File quá lớn" | <1ms |
| 1.1f | `api/main.py:49-59` | `IngestionConfig(...)` | settings values | `IngestionConfig(chunk_strategy="token", chunk_size=1000, ...)` | <1ms |
| 1.1g | `api/main.py:60` | `run_ingestion(pdf_bytes, "hopdong.pdf", config)` | bytes + str + IngestionConfig | `IngestionResult(file_record=..., total_chunks=45, ...)` | ~5-30s |
| 1.1h | `api/main.py:61-67` | serialize response | IngestionResult | `{"status":"success", "file_id":"abc-123...", "total_chunks":45, "duration_seconds":12.5}` | <1ms |

### 1.2 Pipeline: run_ingestion()

**File:** `ingestion/pipeline.py:35`

```python
def run_ingestion(
    pdf_bytes: bytes,           # b'%PDF-1.4\n...'  (nội dung file PDF)
    filename: str,              # "hopdong.pdf"
    config: IngestionConfig,    # IngestionConfig(chunk_strategy="token", chunk_size=1000, ...)
    embedder: BaseEmbedder | None = None,
) -> IngestionResult:
```

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 1.2a | `pipeline.py:44` | `uuid.uuid4()` | không | `file_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"` (36 ký tự) | <1ms |
| 1.2b | `pipeline.py:45-50` | `FileRecord(...)` | file_id, filename, minio_path, size_bytes | `FileRecord(file_id="a1b2...", filename="hopdong.pdf", minio_path="documents/a1b2.../hopdong.pdf", size_bytes=512000)` | <1ms |
| 1.2c | `pipeline.py:54-60` | `chunk_pdf(...)` | pdf_bytes, file_id, strategy, chunk_size, chunk_overlap | `list[ChunkDoc]` (xem 1.3 bên dưới) | ~0.5-3s |
| 1.2d | `pipeline.py:66` | `get_embedder("local")` | `provider="local"` | `LocalEmbedder(model_name="all-MiniLM-L6-v2")` | ~2-5s (load model nếu local) |
| 1.2e | `pipeline.py:68` | `emb.embed_chunks(chunks, batch_size=32)` | list[ChunkDoc], batch_size=32 | `list[EmbeddedChunk]` (xem 1.5 bên dưới) | ~1-30s |
| 1.2f | `pipeline.py:72-75` | `IngestionStore(VectorStore, MetadataStore)` | chroma_path="./chroma_db", mongo_uri="mongodb://localhost:27017" | `IngestionStore` instance | ~50ms (kết nối DB) |
| 1.2g | `pipeline.py:76` | `store.save(embedded_chunks)` | list[EmbeddedChunk] | `StoreResult(...)` (xem 1.6 bên dưới) | ~100-500ms |
| 1.2h | `pipeline.py:78` | `datetime.now() - start` | — | `duration_seconds = 12.5` (float) | <1ms |
| 1.2i | `pipeline.py:81-85` | `IngestionResult(...)` | file_record, total_chunks, store_result, duration | `IngestionResult` | <1ms |

### 1.3 Chunker: chunk_pdf()

**File:** `ingestion/services/chunker.py:40`

```python
def chunk_pdf(
    pdf_bytes: bytes,               # b'%PDF-1.4\n...' (500KB)
    file_id: str,                   # "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    strategy: ChunkStrategy,        # "token" | "page"
    chunk_size: int,                # 1000 (token strategy)
    chunk_overlap: int,             # 200 (token strategy)
) -> list[ChunkDoc]:
```

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 1.3a | `chunker.py:57` | `PdfReader(io.BytesIO(pdf_bytes))` | pdf_bytes | `reader` (pypdf PdfReader, 30 pages) | ~50ms |
| 1.3b | `chunker.py:58` | `_extract_pages(reader)` | PdfReader | `list[dict]` (xem 1.3b chi tiết) | ~200ms |
| 1.3c | `chunker.py:59-62` | branch strategy | `"token"` | → gọi `_chunk_by_token()` | — |

**1.3b — `_extract_pages()` (chunker.py:65):**

| Step | File:Line | Hàm | Input | Output |
|------|-----------|-----|-------|--------|
| 1.3b.1 | `chunker.py:72-73` | `page.extract_text()` | page object (trang 1) | `raw = "  HỢP ĐỒNG  \n\n  Điều 1: ...  \n\n  1/30  "` |
| 1.3b.2 | `chunker.py:74` | `_clean_text(raw)` | `"  HỢP ĐỒNG  \n\n..."` | `"HỢP ĐỒNG Điều 1: ..."` |
| 1.3b.3 | `chunker.py:75-76` | filter + append | text != "" | `{"page_num": 1, "text": "HỢP ĐỒNG Điều 1: ..."}` |
| 1.3b.4 | `chunker.py:77` | return | list đã build | `[{"page_num":1,"text":"..."}, {"page_num":2,"text":"..."}, ...]` (30 items) |

**1.3b.2 — `_clean_text()` (chunker.py:11):**

| Step | File:Line | Regex | Input | Output |
|------|-----------|-------|-------|--------|
| i | `chunker.py:21` | `re.sub(r"\n{3,}", "\n\n", text)` | text có 4+ dòng trống | gom về 2 dòng trống |
| ii | `chunker.py:22` | `re.sub(r"\n\d+\s*/\s*\d+", "", text)` | `"...\n1/30..."` | `"... ..."` (xóa page footer) |
| iii | `chunker.py:23` | `re.sub(r"(?i)^(trang|page)\s*\d+\s*$", "", ...)` | `"Trang 1"` ở đầu dòng | `""` (xóa) |
| iv | `chunker.py:24` | `" ".join(text.split())` | normalize whitespace | "HỢP ĐỒNG Điều 1: ..." (1 space giữa các từ) |

**1.3c — `_chunk_by_token()` (chunker.py:104):**

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 1.3c.1 | `chunker.py:123-124` | validate | `chunk_size=1000, chunk_overlap=200` | pass (overlap < size) | <1ms |
| 1.3c.2 | `chunker.py:129-134` | gom text + đánh dấu boundaries | 30 pages | `full_text = "HỢP ĐỒNG Điều 1...\nĐiều 2...\n..."` dài ~15000 ký tự | ~1ms |
| | | | | `boundaries = [(0,500,1), (500,1000,2), ...]` | |
| 1.3c.3 | `chunker.py:138` | `step = chunk_size - chunk_overlap` | `1000 - 200` | `step = 800` | <1ms |
| 1.3c.4 | `chunker.py:141-156` | sliding window loop | pos=0, 800, 1600, ... | mỗi vòng lặp: cắt `full_text[pos:pos+1000]`, tạo ChunkDoc | ~5ms |
| 1.3c.5 | `chunker.py:145` | `_find_page_range(pos, end_pos, boundaries)` | `pos=0, end_pos=1000` | `page_start=1, page_end=2` | <1ms |
| 1.3c.6 | `chunker.py:144` | `_is_valid_chunk(text)` | `"HỢP ĐỒNG Điều 1: ..."` (500 chars) | `True` (len≥50, letter_ratio>0.3) | <1ms |
| 1.3c.7 | `chunker.py:156` | return | — | `list[ChunkDoc]` gồm ~20 chunks | — |

**Ví dụ output chunks:**
```python
[
    ChunkDoc(
        chunk_id="b1b1b1b1-...", file_id="a1b2c3d4-...",
        text="HỢP ĐỒNG Điều 1: Bên A và Bên B thỏa thuận...",
        page_start=1, page_end=2, chunk_index=0, char_count=987
    ),
    ChunkDoc(
        chunk_id="c1c1c1c1-...", file_id="a1b2c3d4-...",
        text="...Điều 3: Thời hạn hợp đồng là 12 tháng...",
        page_start=2, page_end=4, chunk_index=1, char_count=1000
    ),
    ...
]
```

**1.3d — `_chunk_by_page()` (chunker.py:80) — chiến lược thay thế:**

| Step | File:Line | Hàm | Input | Output |
|------|-----------|-----|-------|--------|
| 1.3d.1 | `chunker.py:89-90` | per page loop | `{"page_num":1, "text":"..."}` | `_is_valid_chunk(text)` → True/False |
| 1.3d.2 | `chunker.py:92-99` | `ChunkDoc(...)` | page_num, text | 1 ChunkDoc per valid page |
| 1.3d.3 | `chunker.py:101` | return | — | list[ChunkDoc] (mỗi trang 1 chunk, đã lọc rác) |

### 1.4 Embedder Factory: get_embedder()

**File:** `ingestion/services/embedder.py:118`

```python
def get_embedder(provider: str = "local") -> BaseEmbedder:
```

| Step | File:Line | Logic | Input | Output | Thời gian |
|------|-----------|-------|-------|--------|-----------|
| 1.4a | `embedder.py:125` | `provider == "local"` | `"local"` | `LocalEmbedder(model_name="all-MiniLM-L6-v2")` | ~2-5s (load SentenceTransformer) |
| 1.4b | `embedder.py:127` | `provider == "openai"` | `"openai"` | `OpenAIEmbedder(api_key=os.environ["OPENAI_API_KEY"])` | ~10ms (init client) |
| 1.4c | `embedder.py:129` | `provider == "ollama"` | `"ollama"` | `OllamaEmbedder(base_url="http://100.71.230.7:11434", model="phi4-mini")` | ~1ms |

**LocalEmbedder chi tiết (embedder.py:61):**
- `model_name = "all-MiniLM-L6-v2"`
- Vector dimension: **384**
- `SentenceTransformer(model_name)` loads model ~90MB, first load ~5s, cached later

**OllamaEmbedder chi tiết (embedder.py:83):**
- Default model: `phi4-mini`
- Gọi HTTP POST đến `{OLLAMA_BASE_URL}/api/embed`
- Payload: `{"model": "phi4-mini", "input": ["text1", "text2", ...]}`
- Response: `{"embeddings": [[0.12, -0.45, ...], [0.33, 0.67, ...]]}`

**OpenAIEmbedder chi tiết (embedder.py:39):**
- Model: `text-embedding-3-small` (1536 chiều)
- Gọi `client.embeddings.create(model="text-embedding-3-small", input=texts)`
- API key từ: tham số `api_key` hoặc `os.environ["OPENAI_API_KEY"]`

### 1.5 Embedder: embed_chunks()

**File:** `ingestion/services/embedder.py:23` (BaseEmbedder.embed_chunks)

```python
def embed_chunks(self, chunks: list[ChunkDoc], batch_size: int = 32) -> list[EmbeddedChunk]:
```

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 1.5a | `embedder.py:29` | loop `range(0, 20, 32)` | len(chunks)=20, batch_size=32 | 1 batch duy nhất (20 chunks) | — |
| 1.5b | `embedder.py:30` | slice batch | i=0 | `batch = chunks[0:20]` (20 ChunkDoc) | <1ms |
| 1.5c | `embedder.py:31` | `self._embed_batch([c.text for c in batch])` | `["HỢP ĐỒNG...", "Điều 3...", ...]` (20 strings) | `[[0.12, -0.45, ...], [0.33, 0.67, ...], ...]` (20 vectors × 384) | ~200ms (local) / ~500ms (ollama) |
| 1.5d | `embedder.py:32-33` | zip + from_chunk | chunk + vector | `EmbeddedChunk(chunk_id="...", vector=[...], model_name="all-MiniLM-L6-v2")` | <1ms |
| 1.5e | `embedder.py:34-35` | `time.sleep(0.1)` | giữa các batch | delay 100ms (rate limiting) | 100ms |
| 1.5f | `embedder.py:36` | return `results` | — | `list[EmbeddedChunk]` (20 items) | — |

**Ví dụ EmbeddedChunk output:**
```python
EmbeddedChunk(
    chunk_id="b1b1b1b1-...",
    file_id="a1b2c3d4-...",
    text="HỢP ĐỒNG Điều 1: Bên A và Bên B thỏa thuận...",
    page_start=1, page_end=2, chunk_index=0,
    vector=[0.0234, -0.123, 0.567, ..., -0.045],  # 384 floats (local) hoặc 1536 (openai)
    model_name="all-MiniLM-L6-v2"
)
```

**Ví dụ `_embed_batch` input/output cho LocalEmbedder:**
- Input: `texts = ["HỢP ĐỒNG Điều 1...", "Điều 3: Thời hạn..."]` (list[str])
- Internal: `self._model.encode(texts, convert_to_numpy=True)` → numpy array shape (2, 384)
- Output: `[v.tolist() for v in ...]` → `[[0.0234, -0.123, ...], [0.0456, 0.789, ...]]`

### 1.6 Store: IngestionStore.save()

**File:** `ingestion/storage/store.py:126`

```python
def save(self, chunks: list[EmbeddedChunk]) -> StoreResult:
```

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 1.6a | `store.py:131-132` | validate | `len(chunks) == 0` | `ValueError` nếu rỗng | <1ms |
| 1.6b | `store.py:134` | `chunks[0].file_id` | — | `file_id = "a1b2c3d4-..."` | <1ms |
| **Step 1: Chroma** |
| 1.6c | `store.py:138` | `self._vector.upsert(chunks)` | 20 EmbeddedChunk | `chroma_ids = ["b1b1...", "c1c1...", ...]` (20 UUIDs) | ~50ms |
| 1.6d | `store.py:29-50` | `VectorStore.upsert()` chi tiết: | | | |
| | `store.py:38-49` | `self._collection.upsert(ids=[...], embeddings=[...], documents=[...], metadatas=[...])` | 20 items mỗi list | `None` | ~30ms |
| | `store.py:50` | return | — | `[c.chunk_id for c in chunks]` | <1ms |
| **Step 2: MongoDB** |
| 1.6e | `store.py:144` | `self._metadata.insert_chunks(chunks)` | 20 EmbeddedChunk | `mongo_ids = ["67a1b2c3...", ...]` (20 ObjectIDs) | ~50ms |
| **Step 3: Kết quả** |
| 1.6f | `store.py:152-156` | `StoreResult(...)` | file_id, total_chunks=20, chroma_ids, mongo_ids | `StoreResult` | <1ms |

---

## 2. RETRIEVAL & QUERY ROUTING FLOW

### 2.0 TỔNG QUAN CÂY GỌI HÀM

```
POST /query  (api/main.py:113)
│
├── 2.0  QueryRequest(query, file_id, top_k)  ← parse từ JSON body
│
└── 2.1  search_and_generate(query, file_id, top_k)  → RAGResponse
          │
          ├── 2.1a  QueryRouter.classify(query)       → intent: QueryIntent
          │         └── Gọi gemini-3.1-flash-lite phân loại [DOCUMENT_QUERY, CHITCHAT, OUT_OF_SCOPE]
          │
          ├── 2.1b  [IF CHITCHAT]
          │         └── Generator.generate_chitchat(query) → RAGResponse(answer, sources=[], intent="CHITCHAT")
          │
          ├── 2.1c  [IF OUT_OF_SCOPE]
          │         └── Trả về câu từ chối lịch sự       → RAGResponse(answer, sources=[], intent="OUT_OF_SCOPE")
          │
          └── 2.1d  [IF DOCUMENT_QUERY]
                    ├── get_search_terms(query) → retrieval_query, sparse_query, rerank_query
                    ├── get_embedder("gemini").embed_query(retrieval_query) → query_vector (768 chiều)
                    └── retrieve_hybrid_and_rerank() → Top-5 chunks tinh khiết:
                         ├── Branch A (Dense): Retriever.search(query_vector, top_k=15)
                         ├── Branch B (Sparse): BM25Retriever.search(sparse_query, top_k=15)
                         ├── RRF Fusion: reciprocal_rank_fusion(vector_hits, bm25_hits, k=60) → 20-25 candidates
                         └── Reranking: Reranker.rerank(rerank_query, candidates, top_k=5) (FlashRank ONNX)
                    └── Generator.generate(query, hits) → (answer, sources)
                         ├── _enrich_sources(hits)  → MongoDB metadata lookup (filename, pages)
                         ├── _build_prompt(query, hits) → Prompt có kèm context
                         ├── _generate_gemini(prompt)   → Gemini 3.1 Flash-Lite trả lời tiếng Việt (kèm auto-retry 429/503)
                         └── _format_references(sources) → "(Tham khảo tại trang X–Y của PDF...)"

### 2.1 API Layer: POST /query

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 2.1a | `api/main.py:92` | `async def query(request: QueryRequest)` | JSON body `{"query": "Điều 1 nói gì?", "file_id": null, "top_k": 5}` | HTTP JSON response | ~tổng pipeline |
| 2.1b | `api/main.py:95-99` | `search_and_generate(query="Điều 1 nói gì?", file_id=None, top_k=5)` | str, None, int | `RAGResponse(answer="...", sources=[...], query_time_ms=1234.5)` | ~1-5s |
| 2.1c | `api/main.py:100-115` | serialize response | RAGResponse | JSON với answer + sources (mỗi source có chunk_id, file_id, text[:500], page_start, page_end, score, filename) | <1ms |

### 2.2 Pipeline: search_and_generate()

**File:** `retrieval/pipeline.py:14`

```python
def search_and_generate(
    query: str,                          # "Điều 1 của hợp đồng nói gì?"
    file_id: Optional[str] = None,       # None (search toàn bộ) hoặc "a1b2c3d4-..."
    top_k: int = 5,                      # 5
    embed_provider: Optional[str] = None, # None → dùng settings.EMBED_PROVIDER
    llm_provider: Optional[str] = None,   # None → dùng settings.LLM_PROVIDER
) -> RAGResponse:
```

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 2.2a | `pipeline.py:23` | `settings.EMBED_PROVIDER` | — | `"local"` (từ config flow) | <1ms |
| 2.2b | `pipeline.py:24` | `get_embedder("local")` | `"local"` | `LocalEmbedder(model_name="all-MiniLM-L6-v2")` | ~0ms (đã cache) hoặc ~5s (first load) |
| 2.2c | `pipeline.py:25` | `embedder._embed_batch(["Điều 1 của hợp đồng nói gì?"])` | list[str] len=1 | `[[0.012, -0.345, 0.678, ..., 0.234]]` (1 vector × 384) | ~50-200ms |
| 2.2d | `pipeline.py:25` | `[0]` (lấy vector đầu) | list[list[float]] | `query_vector = [0.012, -0.345, 0.678, ..., 0.234]` | <1ms |
| 2.2e | `pipeline.py:27-30` | `Retriever("./chroma_db", "rag_chunks")` | paths | `retriever` instance | ~10ms (ChromaDB PersistentClient) |
| 2.2f | `pipeline.py:31-35` | `retriever.search(query_vector, top_k=5, file_id=None)` | vector, 5, None | `hits = [dict, dict, ...]` (xem 2.3 bên dưới) | ~20ms |
| 2.2g | `pipeline.py:37` | `Generator(provider="ollama")` | `"ollama"` | `Generator` instance | <1ms |
| 2.2h | `pipeline.py:38` | `generator.generate(query, hits)` | str + list[dict] | `(answer, sources)` (xem 2.4 bên dưới) | ~2-10s |
| 2.2i | `pipeline.py:40` | `(time.time() - start) * 1000` | — | `elapsed_ms = 2345.67` (float) | <1ms |
| 2.2j | `pipeline.py:41-44` | `RAGResponse(answer, sources, query_time_ms)` | str, list[RetrievedChunk], float | `RAGResponse` | <1ms |

### 2.3 Retriever: search()

**File:** `retrieval/retriever.py:18`

```python
def search(
    self,
    query_vector: list[float],    # [0.012, -0.345, ..., 0.234] (384 floats)
    top_k: int = 5,               # 5
    file_id: Optional[str] = None, # None hoặc "a1b2c3d4-..."
) -> list[dict]:
```

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 2.3a | `retriever.py:24-26` | build where filter | `file_id=None` | `where_filter = None` (search toàn bộ collection) | <1ms |
| 2.3b | `retriever.py:24-26` | (alternative) | `file_id="a1b2..."` | `where_filter = {"file_id": "a1b2..."}` | <1ms |
| 2.3c | `retriever.py:28-33` | `self._collection.query(query_embeddings=[[...]], n_results=5, where=None, include=["documents","metadatas","distances"])` | vector + params | ChromaDB result dict: `{"ids": [["id1","id2",...]], "documents": [["text1",...]], "metadatas": [[{...},...]], "distances": [[0.12, 0.25,...]]}` | ~10ms |
| 2.3d | `retriever.py:35-50` | build hits loop | Chroma result | `hits = [{"chunk_id": "b1b1...", "text": "...", "file_id": "...", "page_start": 1, "page_end": 2, "chunk_index": 0, "distance": 0.12, "score": 0.88}, ...]` (5 items) | <1ms |
| 2.3e | `retriever.py:51` | return hits | — | `list[dict]` (5 items, sorted by distance ascending = score descending) | — |

**Ví dụ output `hits`:**
```python
[
    {
        "chunk_id": "b1b1b1b1-...",
        "text": "HỢP ĐỒNG Điều 1: Bên A và Bên B thỏa thuận...",
        "file_id": "a1b2c3d4-...",
        "page_start": 1, "page_end": 2, "chunk_index": 0,
        "distance": 0.12,       # cosine distance (0 = identical, 2 = opposite)
        "score": 0.88,           # 1.0 - distance (1 = perfect match)
    },
    {
        "chunk_id": "d1d1d1d1-...",
        "text": "...Điều 2: Bên A có trách nhiệm...",
        "file_id": "a1b2c3d4-...",
        "page_start": 3, "page_end": 3, "chunk_index": 2,
        "distance": 0.35,
        "score": 0.65,
    },
    ...
]
```

**Ghi chú về ChromaDB query:**
- Sử dụng `hnsw:space: cosine` → distance là cosine distance
- `score = 1.0 - distance` → score cao = giống query hơn
- ChromaDB tự động dùng HNSW index để tìm approximate nearest neighbors, thời gian ~O(log n)

### 2.4 Generator: generate()

**File:** `retrieval/generator.py:78`

```python
def generate(
    self,
    query: str,                       # "Điều 1 của hợp đồng nói gì?"
    hits: list[dict],                  # 5 items từ retriever
    system_prompt: Optional[str] = None,
) -> tuple[str, list[RetrievedChunk]]:
```

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 2.4a | `generator.py:84-85` | check empty hits | `len(hits) == 0` | `("Không tìm thấy thông tin liên quan.", [])` | <1ms |
| 2.4b | `generator.py:87` | `self._build_prompt(query, hits)` | str + list[dict] | `prompt` string (xem 2.4b bên dưới) | <1ms |
| 2.4c | `generator.py:88` | `self._enrich_sources(hits)` | list[dict] | `list[RetrievedChunk]` (xem 2.4c bên dưới) | ~25ms (5x MongoDB lookup) |
| 2.4d | `generator.py:90-93` | provider switch | `"ollama"` | → `_generate_ollama(prompt, system)` (xem 2.4d) | ~2-10s |
| 2.4e | `generator.py:95` | return | — | `(answer, sources)` | — |

**2.4b — `_build_prompt()` (generator.py:65):**

| Step | File:Line | Input | Output |
|------|-----------|-------|--------|
| 2.4b.1 | `generator.py:67-68` | hits loop | `context_parts = ["[1] HỢP ĐỒNG...", "[2] Điều 2: ...", ...]` |
| 2.4b.2 | `generator.py:69` | join | `context = "[1] HỢP ĐỒNG...\n\n[2] Điều 2: ...\n\n..."` |
| 2.4b.3 | `generator.py:70-76` | format | `"Dựa vào các đoạn văn bản sau đây, hãy trả lời câu hỏi.\n\n=== NGỮ CẢNH ===\n[1] HỢP ĐỒNG...\n\n=== CÂU HỎI ===\nĐiều 1 của hợp đồng nói gì?\n\nCâu trả lời:"` |

**2.4c — `_enrich_sources()` (generator.py:46):**

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 2.4c.1 | `generator.py:49` | `_lookup_metadata("b1b1b1b1-...")` | chunk_id | MongoDB doc: `{"chunk_id":"b1b1...","file_id":"a1b2...","text":"...","filename":"hopdong.pdf",...}` hoặc `None` | ~5ms per lookup |
| 2.4c.2 | `generator.py:38-44` | `_lookup_metadata` chi tiết: | | | |
| | `generator.py:40` | `_get_mongo_col(uri, db)` | mongo_uri, mongo_db | PyMongo collection object (cached global) | ~1ms (first) / ~0ms (cached) |
| | `generator.py:41` | `col.find_one({"chunk_id": chunk_id})` | query dict | `{"_id": ObjectId(...), "chunk_id": "b1b1...", "file_id": "...", "filename": "hopdong.pdf", ...}` | ~5ms |
| 2.4c.3 | `generator.py:50-52` | extract filename | meta dict | `filename = "hopdong.pdf"` hoặc `"unknown"` | <1ms |
| 2.4c.4 | `generator.py:53-62` | `RetrievedChunk(...)` | các field từ hit + filename từ MongoDB | `RetrievedChunk(chunk_id="b1b1...", file_id="a1b2...", text="...", page_start=1, page_end=2, chunk_index=0, score=0.88, filename="hopdong.pdf")` | <1ms |

**2.4d — `_generate_ollama()` (generator.py:97):**

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 2.4d.1 | `generator.py:98` | build URL | base_url | `"http://100.71.230.7:11434/api/generate"` | <1ms |
| 2.4d.2 | `generator.py:99-107` | build payload | prompt, model, system | `{"model": "phi4-mini", "prompt": "...", "stream": False, "system": "Bạn là trợ lý trả lời câu hỏi..."}` | <1ms |
| 2.4d.3 | `generator.py:110` | `requests.post(url, json=payload, timeout=120)` | URL + payload | HTTP Response (JSON) | ~2-10s (LLM inference) |
| 2.4d.4 | `generator.py:112` | `resp.json()` | response | `{"model":"phi4-mini","created_at":"...","response":"Điều 1 của hợp đồng quy định...","done":true,...}` | <1ms |
| 2.4d.5 | `generator.py:113` | `data.get("response", "").strip()` | dict | `"Điều 1 của hợp đồng quy định Bên A và Bên B..."` | <1ms |

**2.4e — `_generate_openai()` (generator.py:118, alternative path):**

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 2.4e.1 | `generator.py:120-121` | `OpenAI(api_key=...)` | API key từ settings | OpenAI client | ~10ms |
| 2.4e.2 | `generator.py:123-128` | `client.chat.completions.create(model="gpt-4o-mini", messages=[...])` | system + user messages | ChatCompletion object | ~1-3s |
| 2.4e.3 | `generator.py:130` | `response.choices[0].message.content` | ChatCompletion | `"Điều 1 quy định..."` (str) | <1ms |

### 2.5 Ví dụ RAGResponse output cuối cùng:

```python
RAGResponse(
    answer="Điều 1 của hợp đồng quy định Bên A (Công ty XYZ) và Bên B (Công ty ABC) thỏa thuận...",
    sources=[
        RetrievedChunk(
            chunk_id="b1b1b1b1-...", file_id="a1b2c3d4-...",
            text="HỢP ĐỒNG Điều 1: Bên A và Bên B thỏa thuận...",
            page_start=1, page_end=2, chunk_index=0,
            score=0.88, filename="hopdong.pdf"
        ),
        ...
    ],
    query_time_ms=2345.67
)
```

---

## 3. ROLLBACK FLOW — MongoDB fail sau Chroma upsert

### 3.0 TỔNG QUAN

```
IngestionStore.save(embedded_chunks)
│
├── Step 1: VectorStore.upsert(chunks)        ✅ Chroma đã ghi
│     └── return chroma_ids
│
├── Step 2: MetadataStore.insert_chunks(chunks)  ❌ MongoDB fail!
│
├── Step 3: Catch exception → ROLLBACK
│     ├── VectorStore.delete_by_file(file_id)
│     │    ├── self._collection.get(where={"file_id": file_id})
│     │    │    → {"ids": ["b1b1...", "c1c1...", ...]}
│     │    └── self._collection.delete(ids=["b1b1...", "c1c1...", ...])
│     │
│     └── raise RuntimeError("MongoDB insert failed (Chroma rolled back): ...")
```

### 3.1 Trace chi tiết

| Step | File:Line | Hàm | Input | Output | Trạng thái DB |
|------|-----------|-----|-------|--------|---------------|
| 3.1a | `store.py:137-140` | `self._vector.upsert(chunks)` | 20 EmbeddedChunk | success → `chroma_ids` | Chroma: 20 chunks mới. MongoDB: chưa có gì |
| 3.1b | `store.py:144` | `self._metadata.insert_chunks(chunks)` | 20 EmbeddedChunk | `RuntimeError("connection refused")` | MongoDB fail |
| 3.1c | `store.py:145-147` | catch Exception | RuntimeError | vào block except | |
| 3.1d | `store.py:147` | `self._vector.delete_by_file(file_id)` | `file_id = "a1b2c3d4-..."` | gọi Chroma get → delete | |
| 3.1e | `store.py:57` | `self._collection.get(where={"file_id": file_id})` | `{"file_id": "a1b2..."}` | `{"ids": ["b1b1...", "c1c1...", ...]}` (20 ids) | — |
| 3.1f | `store.py:59` | `self._collection.delete(ids=["b1b1...", ...])` | 20 ids | thành công | Chroma: 20 chunks đã xóa. MongoDB: không có gì (như ban đầu) |
| 3.1g | `store.py:148-149` | except (nếu delete cũng fail) | — | `pass` (log lỗi nhưng không crash) | **best-effort rollback** |
| 3.1h | `store.py:150` | `raise RuntimeError(...)` | error message | exception propagate lên pipeline | — |
| 3.1i | `pipeline.py:76` (gọi store.save) | exception propagate | RuntimeError | lên `api/main.py:68` | — |
| 3.1j | `api/main.py:69-70` | catch Exception | RuntimeError | `HTTPException(500, "Ingestion failed: ...")` | API trả lỗi 500 |

**Kết quả cuối cùng:** DB ở trạng thái nhất quán (atomic) — cả Chroma và MongoDB đều không có dữ liệu của file bị lỗi. Tránh split-brain: có vector mà không có text.

---

## 4. DELETE FLOW

### 4.0 TỔNG QUAN

```
DELETE /files/{file_id}  (api/main.py:73)
│
├── VectorStore(chroma_path, chroma_collection)
│     └── PersistentClient + get_or_create_collection
├── MetadataStore(mongo_uri, mongo_db)
│     └── MongoClient + create indexes
│
├── vector_store.delete_by_file(file_id)    (best-effort, warning nếu fail)
└── metadata_store.delete_by_file(file_id)  (critical, error nếu fail)
```

| Step | File:Line | Hàm | Input | Output | Thời gian |
|------|-----------|-----|-------|--------|-----------|
| 4.1 | `api/main.py:73` | `@app.delete("/files/{file_id}")` | `file_id = "a1b2c3d4-..."` (path param) | HTTP JSON | ~50ms |
| 4.2 | `api/main.py:75-76` | init stores | file_id | VectorStore + MetadataStore | ~20ms |
| 4.3 | `api/main.py:79` | `vector_store.delete_by_file(file_id)` | file_id | xóa khỏi Chroma | ~10ms |
| 4.4 | `store.py:57` | `self._collection.get(where={"file_id": file_id})` | file_id | `{"ids": [...]}` | ~5ms |
| 4.5 | `store.py:59` | `self._collection.delete(ids=[...])` | ids list | xóa thành công | ~5ms |
| 4.6 | `api/main.py:80-81` | (nếu Chroma fail) | Exception | `logger.warning(...)` → tiếp tục (best-effort) | <1ms |
| 4.7 | `api/main.py:84` | `metadata_store.delete_by_file(file_id)` | file_id | `deleted = 15` (int, số docs đã xóa) | ~10ms |
| 4.8 | `store.py:107` | `self._col.delete_many({"file_id": file_id}).deleted_count` | file_id | số document bị xóa (vd 15) | ~5ms |
| 4.9 | `api/main.py:85-87` | (nếu MongoDB fail) | Exception | `HTTPException(500)` | <1ms |
| 4.10 | `api/main.py:89` | return | — | `{"status": "deleted", "file_id": "a1b2...", "deleted_chunks": 15}` | <1ms |

---

## 5. TÓM TẮT DATA MODEL FLOW

```
                     INGESTION FLOW
                     ==============

PDF bytes ──► ChunkDoc ──► EmbeddedChunk ──► Chroma (vectors)
  (.pdf)       │              │                  │
               │              │                  └── ids, embeddings, documents, metadatas
               │              │
               │              └──► MongoDB (text + metadata)
               │                     └── chunk_id, file_id, text, page_start, page_end,
               │                         chunk_index, model_name, inserted_at, filename
               │
               └── fields: chunk_id, file_id, text, page_start, page_end,
                           chunk_index, char_count


                    RETRIEVAL FLOW
                    ===============

Query text ──► embed ──► query_vector ──► Chroma.query()
                                               │
                                               ▼
                                          top_k hits
                                          (vectors + documents + metadatas + distances)
                                               │
                                               ├──► _build_prompt() ──► prompt string
                                               │
                                               ├──► _enrich_sources()
                                               │      │
                                               │      └──► MongoDB lookup per chunk_id
                                               │             để lấy filename
                                               │
                                               ▼
                                          LLM (Ollama / OpenAI)
                                               │
                                               ▼
                                          answer + sources
                                          (RAGResponse)
```

---

## 6. CÁC FILE VÀ VAI TRÒ

| File | Vai trò | Key class/function |
|------|---------|-------------------|
| `shared/config.py` | Global settings (singleton) | `Settings`, `settings` |
| `api/main.py` | FastAPI endpoints | `ingest()`, `query()`, `delete_file()` |
| `ingestion/models.py` | Data models | `FileRecord`, `ChunkDoc`, `EmbeddedChunk`, `StoreResult` |
| `ingestion/pipeline.py` | Orchestrator | `IngestionConfig`, `IngestionResult`, `run_ingestion()` |
| `ingestion/services/chunker.py` | PDF → chunks | `chunk_pdf()`, `_extract_pages()`, `_chunk_by_token()`, `_chunk_by_page()`, `_clean_text()`, `_is_valid_chunk()`, `_find_page_range()` |
| `ingestion/services/embedder.py` | Text → vectors | `BaseEmbedder`, `LocalEmbedder`, `OllamaEmbedder`, `OpenAIEmbedder`, `get_embedder()` |
| `ingestion/storage/store.py` | Persistence + rollback | `VectorStore`, `MetadataStore`, `IngestionStore` |
| `retrieval/models.py` | Retrieval data models | `QueryRequest`, `RetrievedChunk`, `RAGResponse` |
| `retrieval/pipeline.py` | Retrieval orchestrator | `search_and_generate()` |
| `retrieval/retriever.py` | ChromaDB similarity search | `Retriever.search()` |
| `retrieval/generator.py` | LLM generation | `Generator.generate()`, `_generate_ollama()`, `_generate_openai()`, `_build_prompt()`, `_enrich_sources()`, `_lookup_metadata()` |

---

*Tổng: ~170 dòng trace. Ngày tạo: 2026-08-02.*
