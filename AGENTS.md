# AGENTS.md — RAG Project (Cập nhật: 2026-09-03)

> File này là **nguồn sự thật duy nhất (source of truth)** cho AI agent. 
> Mọi thay đổi về kiến trúc, model, pipeline và kết quả benchmark đều được cập nhật tại đây.


## 0. Quy tắc bắt buộc cho AI Agent (Agent Operational Protocol)

> ⚠️ **BẮT BUỘC:** Mỗi khi hoàn thành một tính năng, tối ưu mã nguồn hoặc sửa xong một lỗi (bug fix), AI Agent **BẮT BUỘC phải cập nhật đồng bộ các file Markdown liên quan** trước khi kết thúc phiên:
> 1. **`AGENTS.md`**: Cập nhật Bảng kiến trúc (Mục 1) nếu đổi công nghệ/model; thêm tóm tắt vào Lịch sử nâng cấp (Mục 2); cập nhật số lượng unit tests passed và tick `[x]` vào Bảng kiểm tra tiến độ (Mục 4).
> 2. **`docs/SESSION.md`**: Ghi lại chi tiết phiên làm việc: tính năng mới, lỗi phát sinh, nguyên nhân và giải pháp kỹ thuật đã code.
> 3. **`docs/PROJECT_REPORT.md` & `docs/EXECUTION_TRACE.md`**: Cập nhật sơ đồ luồng dữ liệu hoặc cây gọi hàm tương ứng nếu có sự thay đổi về pipeline/hàm xử lý.
> 4. **Báo cáo cho người dùng**: Nêu rõ trong khung chat: đã thay đổi tính năng gì và đã cập nhật vào những file `.md` nào.

---

## 1. Tổng quan kiến trúc & Cấu hình hiện tại

Hệ thống RAG: Query ➔ QueryRouter ➔ (Chitchat / Out-of-scope / RAG Pipeline) ➔ Gemini trả lời + Trích dẫn số trang PDF.

| Thành phần | Công nghệ / Model | Ghi chú |
|---|---|---|
| **Query Router** | `gemini-3.1-flash-lite` | Phân loại 3 intents: `DOCUMENT_QUERY`, `CHITCHAT`, `OUT_OF_SCOPE` (~1.2s). |
| **LLM Generation** | `gemini-3.1-flash-lite` | Trả lời câu hỏi dựa trên ngữ cảnh + trích dẫn số trang PDF, có auto-retry 429/503. |
| **Embedding** | `gemini-embedding-001` (768 chiều) | SDK `google-genai`, có auto-retry khi gặp rate limit. |
| **Vector Store** | ChromaDB (container `rag_chroma`, port 8001 / volume `chroma_db`) | Đo độ tương đồng Cosine Similarity (Dense Search). |
| **BM25 Search** | `rank-bm25` (BM25Okapi) | Tìm kiếm từ khóa chính xác (Sparse Search), lọc stopwords. |
| **Fusion** | Reciprocal Rank Fusion (RRF, $k=60$) | Hợp nhất Dense + Sparse thành ~20–25 ứng viên. |
| **Reranker** | FlashRank (`ms-marco-MiniLM-L-12-v2` ONNX) | Chấm điểm chéo (Cross-Encoder), kết hợp điểm tương quan. |
| **Metadata Store** | MongoDB 7 (container `rag_mongo`, port 27017) | Lưu text gốc và metadata. |
| **Object Store** | MinIO (container `rag_minio`, port 9000/9001) | Lưu file PDF gốc (`documents/{file_id}/{filename}`). |
| **API** | FastAPI (container `rag_api`, port 8000, Swagger UI `/docs`) | Endpoints: `/health`, `/ingest`, `/query`, `/files/{id}`. |
| **Telegram Bot** | python-telegram-bot (container `rag_telegram_bot`) | Profile `telegram`, chia nhỏ tin nhắn $\le 4096$ ký tự. |
| **Dữ liệu mẫu đã Ingest** | `Law_fifa.pdf` (241 chunks) | `file_id`: `8f525964-a864-43ef-b87b-34f6482d44f1` |

---

## 2. Diễn biến & Lịch sử nâng cấp dự án

### Giai đoạn 1 (Ngày 19/08) — Khởi tạo hạ tầng & Gemini API
- Setup Docker Compose (MongoDB, ChromaDB, MinIO, FastAPI).
- Ingest thành công `Law_fifa.pdf` (241 chunks, file_id: `8f525964-a864-43ef-b87b-34f6482d44f1`).
- Chuyển đổi từ Ollama sang Gemini API (`gemini-3.5-flash` + `gemini-embedding-001`).
- Xử lý semantic gap tiếng Việt: Thêm glossary `expand_query_for_retrieval()` trong `retrieval/pipeline.py` (`việt vị` ➔ `offside, Law 11`).
- Tự động thêm trích dẫn số trang: `(Tham khảo tại trang X–Y của PDF...)`.
- Khởi tạo bộ eval `fifa_law11_basic.json` và script `retrieval_eval.py`.

### Giai đoạn 2 (Ngày 24/08) — Hoàn thiện Evals, Baseline & Query Router
- **Đóng gói Evals vào Dockerfile:** Thêm `COPY evals/ evals/` vào Docker image.
- **Chạy Benchmark Baseline (Top-K = 5) lưu vào `evals/benchmark_baseline_law11_top5.json`:**
  - **Mean Recall@5:** `1.0` (100% — tìm trọn vẹn 100% các trang gold standard).
  - **Mean Precision@5:** `0.70` (70% trên 4 ca test chuẩn Luật 11 & Luật 12 — còn 30% chunk nhiễu ở trang 80–81, 88–91).
- **Triển khai Query Router (`retrieval/router.py`):**
  - Tích hợp `QueryRouter` sử dụng model nhẹ `gemini-1.5-flash-8b` dùng chung `GEMINI_API_KEY`.
  - Phân loại 3 intents: `DOCUMENT_QUERY`, `CHITCHAT` (bỏ qua Chroma search, Gemini chào hỏi < 1s), `OUT_OF_SCOPE` (từ chối lịch sự).
- **Mở rộng Glossary FIFA & Chuẩn hóa OOP:**
  - Thêm phương thức `embed_query()` cho `BaseEmbedder`.
  - Mở rộng glossary bóng đá: `phạt đền`, `penalty`, `phạt góc`, `ném biên`, `phát bóng`, `thẻ đỏ`, `thẻ vàng`, `chạm tay`.
  - Thêm điều kiện lọc chỉ gắn citation khi câu trả lời thành công.
- **Hệ thống Visual Trace Logs & Shortcut Scripts:**
  - Log chi tiết từng bước: `[STEP 1: NHẬN CÂU HỎI]` ➔ `[STEP 2: QUERY ROUTING]` ➔ `[ROUTER DECISION]` ➔ `[BRANCH: Tên hàm]` ➔ `[HOÀN THÀNH]`.
  - Tạo các file shortcut: `run_swagger.cmd`, `run_eval.cmd`, `view_logs.cmd`, `run_telegram.cmd`, `inspect_chunks.cmd`.


### Giai đoạn 3 (Ngày 03/09) — Nâng cấp Gemini 3.1 Flash-Lite, Fix Bug, Evals & Batch QA Suite
- **Nâng cấp Model Gemini:** Chuyển `QueryRouter` và `Generator` sang model thế hệ mới `gemini-3.1-flash-lite` (sửa triệt để lỗi 404 NOT_FOUND của `gemini-1.5-flash-8b` và hạn mức 20 lượt/ngày của `gemini-3.5-flash`). Thời gian phản hồi giảm xuống ~3–8s.
- **Cơ chế Auto-retry với Exponential Backoff:** Tích hợp bộ bắt lỗi và tự động thử lại khi gặp `429 RESOURCE_EXHAUSTED` hoặc `503 UNAVAILABLE` trong `retrieval/generator.py`.
- **Chuẩn hóa Evals & Baseline:**
  - Chuẩn hóa `evals/fifa_law11_basic.json` (4 ca test nghiệp vụ luật).
  - Xuất file kết quả mốc chính thức: `evals/benchmark_baseline_law11_top5.json` (Recall@5: `1.0`, Precision@5: `0.70`).
  - Cập nhật `run_eval.cmd` tự động mount volume `-v "%cd%/evals:/app/evals"`.
- **Xây dựng Batch QA Test Suite 20 câu hỏi Luật FIFA:**
  - Dataset: `evals/fifa_qa_testset.json` bao quát 4 nhóm luật thi đấu lớn.
  - Runner: `evals/run_qa_batch.py` tự động gửi câu hỏi qua API và xuất kết quả song song ra `evals/results_qa_testset.csv` (cho Excel) và `evals/results_qa_testset.md` (Markdown).
  - Shortcut tiện ích: `run_qa_test.cmd` (1-click chạy kiểm thử).
- **Bảo vệ và cô lập Test Suite:** Tạo `pytest.ini` cố định `testpaths = tests`, thêm `__test__ = False` vào runner để bộ test 20 câu hỏi không bao giờ tự động chạy khi build/start dự án.
- **Quy hoạch tài liệu:** Di chuyển toàn bộ các file Markdown tài liệu hệ thống (`PROJECT_REPORT.md`, `EXECUTION_TRACE.md`, `rag_ingestion_context.md`, `SESSION.md`, `RUNTIME_LOG.md`) vào thư mục `docs/`.

### Giai đoạn 4 (Ngày 03/09) — Triển khai Advanced RAG (Hybrid Search BM25 + FlashRank Reranker)
- **Lưu bản kế hoạch chuẩn hóa:** Lưu tại `docs/superpowers/plans/2026-09-03-advanced-rag-hybrid-rerank.md`.
- **Triển khai BM25 Sparse Search (`retrieval/bm25_retriever.py`):** Dùng `rank-bm25` nạp corpus chunks từ ChromaDB/MongoDB, áp dụng bộ lọc stop words để tập trung vào từ khóa luật.
- **Triển khai RRF Fusion (`retrieval/fusion.py`):** Hợp nhất 15 dense hits và 15 sparse hits theo công thức Reciprocal Rank Fusion ($k=60$), khử trùng lặp và lấy ~20–25 candidates.
- **Triển khai FlashRank Reranker (`retrieval/reranker.py`):** Chạy mô hình Cross-Encoder `ms-marco-MiniLM-L-12-v2` tối ưu qua ONNX Runtime CPU. Áp dụng kỹ thuật kết hợp điểm số tương quan (60% Dense/RRF + 40% Cross-Encoder).
- **Tích hợp Pipeline nâng cao (`retrieval/pipeline.py`):** Hàm `retrieve_hybrid_and_rerank()` tự động điều phối Dense + Sparse + RRF + Reranker, xử lý semantic gap đa ngữ (Vietnamese câu hỏi + English terminology).
- **Unit Tests:** Thêm 4 unit tests mới (`test_bm25.py`, `test_fusion.py`, `test_reranker.py`), nâng tổng số test lên **81/81 tests PASSED (100%)**.
- **Rebuild Docker Image:** Rebuild `rag_project-api:latest` thành công với đầy đủ dependencies `rank-bm25` và `flashrank`.
---

## 3. Các lệnh điều khiển thiết yếu (Chạy từ CMD máy thật)

```cmd
cd /d D:\RAG_project\RAG_project

:: 1. Khởi động API & mở Swagger UI trên trình duyệt:
run_swagger.cmd

:: 2. Xem Log thời gian thực của Query Router & Pipeline:
view_logs.cmd

:: 3. Chạy kiểm thử đánh giá Recall@5 / Precision@5 benchmark:
run_eval.cmd

:: 4. Chạy Telegram Bot (sau khi điền token mới vào .env):
run_telegram.cmd

:: 5. Đo đếm số lượng Ký tự, Từ, Token chính xác của từng Chunk trong CSDL:
inspect_chunks.cmd

:: 6. Rebuild lại Docker image API khi có thay đổi code:
docker compose build api

:: 7. Chạy Benchmark đầy đủ và lưu file kết quả JSON:
docker compose --profile swagger run --rm api python -m evals.retrieval_eval --file-id 8f525964-a864-43ef-b87b-34f6482d44f1 --top-k 5 --output evals/benchmark_baseline_law11_top5.json

:: 8. Chạy kiểm thử tự động 20 câu hỏi Luật bóng đá và xuất file CSV + Markdown:
run_qa_test.cmd
```

---

## 4. Bảng kiểm tra tiến độ (TODO Checklist)

### Đã hoàn thành (Done):
- [x] Ingest `Law_fifa.pdf` (241 chunks) và cấu hình MinIO + ChromaDB + MongoDB.
- [x] Chuyển đổi hoàn toàn sang Gemini API (`google-genai` SDK).
- [x] Fix semantic gap tiếng Việt qua Query Expansion.
- [x] Đóng gói `evals/` vào Dockerfile.
- [x] Đo và lưu kết quả Benchmark Baseline Top-5 (`Recall@5: 1.0`, `Precision@5: 0.70`) vào `evals/benchmark_baseline_law11_top5.json`.
- [x] Triển khai Query Router phân loại 3 intents bằng `gemini-3.1-flash-lite`.
- [x] Bổ sung Visual Trace Logs chi tiết từng bước và từng hàm.
- [x] Nâng cấp Generator sang `gemini-3.1-flash-lite` kèm cơ chế Auto-retry với Backoff khi gặp 429/503.
- [x] Xây dựng Batch QA Test Suite 20 câu hỏi, runner tự động xuất file `results_qa_testset.csv` & `results_qa_testset.md`.
- [x] Cô lập phạm vi pytest bằng `pytest.ini` và `__test__ = False`.
- [x] Tạo đầy đủ các script tiện ích: `run_eval.cmd`, `view_logs.cmd`, `run_swagger.cmd`, `run_telegram.cmd`, `run_qa_test.cmd`, `inspect_chunks.cmd`.
- [x] Toàn bộ test suite đạt chuẩn (81/81 unit tests passed).
- [x] Chuẩn hóa cây thư mục: chuyển tài liệu logic/hệ thống vào `docs/`.
- [x] **Hybrid Search (BM25 + ChromaDB Vector):** Kết hợp tìm kiếm từ khóa chính xác BM25 + Vector Search qua RRF Fusion.
- [x] **Reranking (FlashRank Cross-Encoder):** Chấm điểm lại ứng viên bằng Cross-Encoder ONNX nhẹ trên CPU.

### Kế hoạch tiếp theo (Next Steps):
- [ ] **Conversational Memory & Multi-turn Chat:** Tích hợp bộ nhớ ngữ cảnh cho Telegram Bot và API `/query`.
- [ ] **Parent-Document Retrieval:** Cắt chunk nhỏ để tìm kiếm nhưng nạp context lớn từ MongoDB khi sinh câu trả lời.
- [ ] **Kích hoạt Telegram Bot:** Cập nhật token bot mới vào `.env` và kiểm thử thực tế.

---

## 5. Cấu trúc thư mục dự án

```
rag_project/
├── ingestion/                  # PDF ➔ Chunk ➔ Embed ➔ Lưu trữ đa tầng có Rollback
│   ├── models.py              # FileRecord, ChunkDoc, EmbeddedChunk, StoreResult
│   ├── pipeline.py            # run_ingestion()
│   ├── services/
│   │   ├── chunker.py         # chunk_pdf (token & page strategy, overlap tracking)
│   │   └── embedder.py        # GeminiEmbedder, BaseEmbedder (embed_query, embed_chunks)
│   └── storage/
│       ├── store.py           # VectorStore (ChromaDB), MetadataStore (MongoDB), IngestionStore
│       └── object_store.py    # MinioObjectStore (S3-compatible)
│
├── retrieval/                  # Bộ tìm kiếm & sinh câu trả lời
│   ├── router.py              # QueryRouter (Phân loại 3 intents bằng Gemini Flash-Lite)
│   ├── pipeline.py            # search_and_generate(), expand_query_for_retrieval()
│   ├── retriever.py           # Retriever (Cosine Similarity trên ChromaDB)
│   ├── generator.py           # Generator (Gemini 3.5 Flash, format_references, generate_chitchat)
│   └── models.py              # QueryRequest, RetrievedChunk, RAGResponse (có trường intent)
│
├── api/
│   └── main.py                # FastAPI endpoints: /health, /ingest, /query, /files/{id}
│
├── telegram_bot/
│   └── main.py                # Telegram Bot handler (long-polling, text chunking 4096 chars)
│
├── evals/                      # Bộ đánh giá chất lượng Retrieval
│   ├── fifa_law11_basic.json  # Gold Dataset 3 ca test Luật 11 (Expected Pages)
│   ├── retrieval_eval.py      # Script tính Recall@k & Precision@k
│   ├── benchmark_baseline_law11_top5.json # Kết quả đo đối chứng Baseline chính thức
│   └── README.md              # Hướng dẫn chi tiết đánh giá
│
├── shared/
│   └── config.py              # Settings dataclass, auto-load .env
│
├── tests/                      # Bộ unit tests
│   ├── test_router.py         # Unit tests cho QueryRouter
│   ├── test_retrieval_eval.py # Unit tests cho Retrieval Evaluation
│   └── ...                    # Các unit tests khác
│
├── run_swagger.cmd            # Shortcut: Bật API + mở Swagger UI
├── run_eval.cmd               # Shortcut: Chạy benchmark đánh giá
├── view_logs.cmd              # Shortcut: Xem log realtime của Router & Pipeline
├── run_telegram.cmd           # Shortcut: Bật Telegram Bot
├── Dockerfile / docker-compose.yml
└── .env
```