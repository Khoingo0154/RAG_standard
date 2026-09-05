# Session Log — RAG Project

## Tiến độ: HOÀN THÀNH GIAI ĐOẠN 1 (full code + test)
## Ngày: 2026-08-02

### Đã hoàn thành

#### Module code (100%)
- `rag_project/ingestion/` — models, pipeline, chunker (page+token), embedder (Local/OpenAI/Ollama), store (Chroma+MongoDB rollback)
- `rag_project/retrieval/` — models, pipeline, retriever (Chroma cosine search), generator (Ollama/OpenAI LLM)
- `rag_project/api/` — FastAPI: /health, POST /ingest, DELETE /files/{id}, POST /query; CORS, file size limit 50MB
- `rag_project/shared/` — Settings dataclass auto-load từ .env và env vars

#### Infra
- `docker-compose.yml` — MongoDB + Chroma server + API
- `Dockerfile` — Python 3.11-slim
- `.env` — 21 biến cấu hình
- `requirements.txt` — 12 packages

#### Tests: 57 tests / 8 file — TẤT CẢ PASS
- test_models.py (4), test_chunker.py (9), test_chunker_edge.py (5), test_embedder.py (5)
- test_store.py (4), test_pipeline.py (3), test_config.py (2), test_config_env.py (4)
- test_retrieval.py (11), test_api.py (6)

#### Docs
- `PROJECT_REPORT.md` — Báo cáo tổng quan
- `rag_project/EXECUTION_TRACE.md` — Trace execution flow
- `rag_project/RUNTIME_LOG.md` — Log runtime
- `CLAUDE.md` — Hướng dẫn cho AI tools

### Runtime status
- Server: chạy OK trên port 8000
- Health: OK
- Query: OK (trả "Không tìm thấy" vì DB trống)
- Ingestion: FAIL — MongoDB chưa chạy (localhost:27017 refused)
- Ollama: khả dụng (100.71.230.7:11434)

### Kết quả 3 subagent review
- Kiến trúc: tốt, pattern rõ ràng. Cần tách embedder ra shared/
- Lỗi: đã fix hết critical + medium
- Test coverage: khá, thiếu integration test với DB thật

### Cách tiếp tục lần sau
Mở terminal, nói:
> "Đọc CLAUDE.md, tiếp tục RAG project"

Hoặc:
> "Đọc PROJECT_REPORT.md và RUNTIME_LOG.md, chạy MongoDB rồi test ingestion"

---

## Session Log — 2026-09-03: Nâng cấp Gemini 3.1 Flash-Lite, Fix Bug, Evals & Batch QA Suite

### 1. Đã hoàn thành

#### Model & Pipeline
- **Nâng cấp Model Gemini**: Chuyển `QueryRouter` và `Generator` sang `gemini-3.1-flash-lite`:
  - Khắc phục lỗi 404 NOT_FOUND của `gemini-1.5-flash-8b` trên QueryRouter.
  - Khắc phục lỗi cạn quota 20 lượt/ngày (429 RESOURCE_EXHAUSTED) của `gemini-3.5-flash`.
  - Tốc độ phản hồi trung bình giảm xuống ~3–8 giây / câu.
- **Cơ chế Auto-retry với Exponential Backoff**: Tích hợp trực tiếp vào `Generator._generate_gemini` để tự động phục hồi khi gặp rate limit 429 hoặc quá tải tạm thời 503.

#### Evals & Benchmark
- **Chuẩn hóa Evals**: Chuẩn hóa `evals/fifa_law11_basic.json` (4 ca test nghiệp vụ luật).
- **Xuất file đối chứng chính thức**: `evals/benchmark_baseline_law11_top5.json` (Recall@5: `1.0`, Precision@5: `0.70`).
- **Cập nhật script run_eval.cmd**: Bổ sung volume mount `-v "%cd%/evals:/app/evals"`.

#### Batch QA Test Suite (20 câu hỏi Luật FIFA)
- **Dataset**: Tạo `evals/fifa_qa_testset.json` gồm 20 câu hỏi tình huống chuyên sâu, phân loại theo 4 nhóm luật (Việt vị, Lỗi dùng tay, Thẻ phạt, Penalty, Ném biên, Phát bóng, VAR).
- **Runner**: Tạo `evals/run_qa_batch.py` tự động gửi câu hỏi qua API và xuất kết quả song song:
  - `evals/results_qa_testset.csv` (cho Excel / Google Sheets với mã hóa UTF-8 BOM `utf-8-sig`).
  - `evals/results_qa_testset.md` (bảng tổng hợp Markdown).
- **Shortcut tiện ích**: Tạo file `run_qa_test.cmd` để chạy test 1-click.
- **Kết quả thực nghiệm**: 20/20 câu thành công (100%), trích dẫn số trang PDF chuẩn xác.

#### Test Suite & Bảo vệ phạm vi
- **Fix lỗi test case**: Sửa `tests/test_retrieval.py` và cách ly mock `QueryRouter` trong unit tests. Toàn bộ **77/77 tests passed (100%)**.
- **Cô lập kiểm thử**: Tạo `pytest.ini` (`testpaths = tests`) và cờ `__test__ = False` trong `run_qa_batch.py` để đảm bảo bộ test 20 câu hỏi không bao giờ tự động chạy ngầm.

#### Chuẩn hóa tài liệu
- Chuyển các file tài liệu logic và hệ thống vào thư mục `docs/`:
  - `docs/PROJECT_REPORT.md`
  - `docs/EXECUTION_TRACE.md`
  - `docs/rag_ingestion_context.md`
  - `docs/SESSION.md`
  - `docs/RUNTIME_LOG.md`

---

## Session Log — 2026-09-03: Triển khai Advanced RAG (Hybrid Search BM25 + FlashRank Reranker)

### 1. Đã hoàn thành

#### Kế hoạch & Kiến trúc
- **Kế hoạch chuẩn hóa**: Lưu tại `docs/superpowers/plans/2026-09-03-advanced-rag-hybrid-rerank.md`.
- **Chuyển dịch thành công sang Advanced RAG**: Bổ sung tầng lọc Sparse BM25 + RRF Fusion + FlashRank Reranker giữa Retrieval và Generator.

#### Mã nguồn tính năng mới
- **`retrieval/bm25_retriever.py`**:
  - Tích hợp thuật toán `BM25Okapi` qua thư viện `rank-bm25`.
  - Xây dựng bộ lọc 35 stopwords tiếng Anh (`STOP_WORDS`) để loại bỏ các từ nhiễu xuất hiện ở chân trang PDF (`laws, game, fifa, page`).
  - Lấy Top-15 chunks từ khóa chính xác.
- **`retrieval/fusion.py`**:
  - Cài đặt thuật toán Reciprocal Rank Fusion (RRF, $k=60$).
  - Khử trùng lặp `chunk_id`, hợp nhất 15 dense hits + 15 sparse hits thành 20–25 candidates.
- **`retrieval/reranker.py`**:
  - Cài đặt Cross-Encoder model `ms-marco-MiniLM-L-12-v2` qua thư viện `flashrank` (chạy ONNX Runtime trên CPU < 100ms).
  - Áp dụng công thức nội suy điểm số: $\text{Score} = 0.6 \times \text{Normalized\_RRF} + 0.4 \times \text{FlashRank\_Score}$.
- **`retrieval/pipeline.py`**:
  - Xây dựng hàm `retrieve_hybrid_and_rerank()` điều phối luồng 4 bước.
  - Xử lý semantic gap đa ngữ với `get_search_terms()`: bóc tách `retrieval_query` (song ngữ cho vector) và `english_query` (từ khóa tiếng Anh cho BM25 và FlashRank).

#### Unit Tests
- Viết thêm 3 file test mới: `tests/test_bm25.py`, `tests/test_fusion.py`, `tests/test_reranker.py`.
- Toàn bộ test suite đạt **81/81 tests PASSED (100%)**.

#### Báo cáo & Cột Top_5_K
- Cập nhật `evals/run_qa_batch.py` bổ sung cột `Top_5_K` (hiển thị số trang, điểm score và trích đoạn 5 chunks).
- Sao lưu bản kết quả cũ thành `evals/results_qa_testset_v0.1.csv` và `evals/results_qa_testset_v0.1.md`.
- Xuất bản kết quả mới kèm `Top_5_K` vào `evals/results_qa_testset.csv` và `evals/results_qa_testset.md`.

#### Đóng gói Docker & Git
- Rebuild image `rag_project-api:latest` kèm `rank-bm25` và `flashrank`.
- Gắn tag phiên bản `v0.1` và push lên GitHub repository: `https://github.com/Khoingo0154/RAG_standard.git`.
