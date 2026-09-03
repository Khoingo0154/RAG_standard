# Kế hoạch Thực thi: Cập nhật Dockerfile, Tài liệu Evals và Chạy Benchmark Retrieval Baseline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đóng gói thư mục evals vào Docker image, cập nhật hướng dẫn và chạy đo mốc benchmark Recall@5 / Precision@5 baseline trên container.

**Architecture:** Sử dụng Dockerfile để đóng gói mã nguồn `evals/`, chạy `retrieval_eval.py` bên trong container `rag_api` (được kết nối tới volume `chroma_db` và MongoDB container), xuất file JSON lưu chỉ số baseline.

**Tech Stack:** Docker, Docker Compose, Python 3.11, ChromaDB, MongoDB, pytest.

## Global Constraints
- Phải đảm bảo toàn bộ unit tests hiện có trong `tests/` vượt qua (passed).
- Chạy eval bên trong Docker container để truy cập đúng volume `/data/chroma_db`.
- Không thay đổi logic tính điểm trong `evals/retrieval_eval.py`.
- Kết quả benchmark phải được lưu vào file `evals/benchmark_baseline_law11_top5.json`.

---

### Task 1: Cập nhật Dockerfile & evals/README.md

**Files:**
- Modify: `Dockerfile:12-14`
- Modify: `evals/README.md:1-23`
- Test: `tests/test_retrieval_eval.py`

**Interfaces:**
- Consumes: `evals/retrieval_eval.py`, `evals/fifa_law11_basic.json`
- Produces: Docker image chứa `/app/evals`, tài liệu `evals/README.md` cập nhật

- [ ] **Step 1: Cập nhật Dockerfile**
Thêm `COPY evals/ evals/` vào `Dockerfile`:
```dockerfile
COPY api/ api/
COPY telegram_bot/ telegram_bot/
COPY ingestion/ ingestion/
COPY retrieval/ retrieval/
COPY shared/ shared/
COPY evals/ evals/
```

- [ ] **Step 2: Cập nhật `evals/README.md`**
Cập nhật nội dung hướng dẫn chạy Docker command và xuất output JSON:
```markdown
# Retrieval evaluation

`fifa_law11_basic.json` là bộ **gold data** ban đầu: mỗi câu hỏi có các trang PDF đã được người đánh giá đọc và đánh dấu là cần thiết.

## Cách chạy benchmark trong Docker

Do ChromaDB dữ liệu nằm trong Docker volume (`chroma_db`), cần chạy script eval thông qua container:

```cmd
:: 1. Chạy đánh giá và in kết quả ra màn hình:
docker compose --profile swagger run --rm api python -m evals.retrieval_eval --file-id 8f525964-a864-43ef-b87b-34f6482d44f1 --top-k 5

:: 2. Chạy đánh giá và lưu kết quả JSON làm mốc đối chứng (baseline):
docker compose --profile swagger run --rm api python -m evals.retrieval_eval --file-id 8f525964-a864-43ef-b87b-34f6482d44f1 --top-k 5 --output evals/benchmark_baseline_law11_top5.json
```

## Kết quả quan trọng:

- `recall_at_k`: tỷ lệ trang gold xuất hiện trong top-k. Recall 1.0 nghĩa là không bỏ sót trang gold.
- `precision_at_k`: tỷ lệ chunk trong top-k thực sự bao phủ trang gold. Precision thấp nghĩa là có nhiều chunk nhiễu.

## Ba cách tạo test set

1. **Gắn nhãn thủ công (khuyến nghị lúc đầu):** đọc PDF, viết câu hỏi thật, rồi ghi `expected_pages` hoặc `expected_chunk_ids`. Đây là cách đáng tin nhất.
2. **LLM tạo nháp:** dùng Gemini tạo câu hỏi từ từng phần PDF, nhưng người làm dự án phải kiểm tra và sửa lại gold pages trước khi dùng làm benchmark.
3. **Log người dùng thật:** lưu câu hỏi thất bại, sửa lại expected pages sau khi review. Cách này giúp bộ test gần với nhu cầu sử dụng nhất.
```

- [ ] **Step 3: Chạy test kiểm tra unit tests**
Chạy: `.venv\Scripts\python.exe -m pytest tests/test_retrieval_eval.py -v`
Expected: PASS

---

### Task 2: Build lại Docker Image `api`

**Files:**
- Context: `Dockerfile`, `docker-compose.yml`

- [ ] **Step 1: Rebuild Docker image `api`**
Chạy lệnh: `docker compose build api`
Expected: Image build thành công với step `COPY evals/ evals/`.

---

### Task 3: Chạy Benchmark Retrieval Baseline (Top-k=5) & Ghi nhận kết quả

**Files:**
- Target: `evals/benchmark_baseline_law11_top5.json`

- [ ] **Step 1: Chạy benchmark trong container và xuất file JSON**
Chạy lệnh:
```cmd
docker compose --profile swagger run --rm api python -m evals.retrieval_eval --file-id 8f525964-a864-43ef-b87b-34f6482d44f1 --top-k 5 --output evals/benchmark_baseline_law11_top5.json
```
Expected: Lệnh hoàn thành, in kết quả JSON có `mean_recall_at_k` và `mean_precision_at_k`.

- [ ] **Step 2: Đọc file `evals/benchmark_baseline_law11_top5.json` để kiểm tra kết quả**
Xem nội dung và phân tích các chỉ số thu được.

---

### Task 4: Cập nhật AGENTS.md với kết quả Benchmark

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Cập nhật trạng thái các TODO và thêm kết quả baseline vào `AGENTS.md`**
Đánh dấu hoàn thành:
1. `Dockerfile` đã có `COPY evals/ evals/`.
2. Đã chạy benchmark Recall@5/Precision@5 baseline.
3. `evals/README.md` đã cập nhật lệnh Docker container.
Ghi lại kết quả Recall@5 / Precision@5 vào báo cáo của `AGENTS.md`.
