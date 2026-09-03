# Thiết kế: Cập nhật Dockerfile, Tài liệu Evals và Chạy Benchmark Retrieval Baseline

## 1. Bối cảnh & Mục tiêu

### 1.1 Bối cảnh
- Hệ thống RAG hiện tại đã chuyển sang dùng Gemini API (`gemini-3.5-flash` và `gemini-embedding-001`), ChromaDB persistent, MongoDB và MinIO thông qua Docker Compose.
- Dữ liệu `Law_fifa.pdf` đã được ingest vào hệ thống với `file_id`: `8f525964-a864-43ef-b87b-34f6482d44f1` (241 chunks).
- Bộ đánh giá retrieval (`evals/fifa_law11_basic.json` và `evals/retrieval_eval.py`) đã được viết để đo `Recall@k` và `Precision@k` theo số trang gold standard.
- Tuy nhiên, `Dockerfile` chưa copy thư mục `evals/` vào container, tài liệu `evals/README.md` còn ghi lệnh chạy local cũ (không kết nối được Chroma volume trong Docker), và mốc benchmark baseline chưa được chạy và lưu lại.

### 1.2 Mục tiêu
1. Đóng gói đầy đủ thư mục `evals/` vào Docker image để có thể thực thi trực tiếp trong môi trường container.
2. Chuẩn hóa tài liệu `evals/README.md` với các lệnh Docker Compose chính xác, hỗ trợ xuất kết quả JSON.
3. Chạy benchmark retrieval baseline với Top-K=5 trên bộ gold data Luật 11 FIFA và lưu kết quả vào `evals/benchmark_baseline_law11_top5.json`.
4. Cập nhật `AGENTS.md` để đồng bộ tiến độ dự án.

---

## 2. Chi tiết các thay đổi

### 2.1 Cập nhật `Dockerfile`
- **File:** `Dockerfile`
- **Nội dung thay đổi:** Thêm lệnh `COPY evals/ evals/` sau `COPY shared/ shared/`.
- **Mục đích:** Khi build image, toàn bộ script và dữ liệu đánh giá nằm sẵn trong container tại `/app/evals`.

### 2.2 Cập nhật `evals/README.md`
- **File:** `evals/README.md`
- **Nội dung thay đổi:**
  - Thay thế lệnh local python bằng lệnh `docker compose --profile swagger run ...` hoặc `docker compose exec ...`.
  - Hướng dẫn chi tiết cách chạy in kết quả ra terminal và cách dùng `--output` để ghi file JSON.
  - Giải thích ý nghĩa của `Recall@k` (độ bao phủ trang gold) và `Precision@k` (độ chính xác của top-k chunk).

### 2.3 Cập nhật `AGENTS.md`
- **File:** `AGENTS.md`
- **Nội dung thay đổi:** Đánh dấu hoàn thành các TODO dở dang từ phiên ngày 19/08 (Dockerfile, evals README, baseline benchmark).

---

## 3. Quy trình thực thi & Chạy Benchmark

1. **Kiểm tra unit tests:** Chạy `pytest` đảm bảo toàn bộ các test hiện có tiếp tục pass.
2. **Cập nhật mã nguồn & tài liệu:** Chỉnh sửa `Dockerfile`, `evals/README.md`, `AGENTS.md`.
3. **Build lại Docker image:** `docker compose build api` (hoặc `docker compose --profile swagger build`).
4. **Chạy benchmark baseline:**
   Thực thi lệnh đánh giá:
   ```cmd
   docker compose --profile swagger run --rm api python -m evals.retrieval_eval --file-id 8f525964-a864-43ef-b87b-34f6482d44f1 --top-k 5 --output evals/benchmark_baseline_law11_top5.json
   ```
5. **Kiểm tra & Xác nhận:** Đọc kết quả trong `evals/benchmark_baseline_law11_top5.json`, đối chiếu `mean_recall_at_k` và `mean_precision_at_k`.

---

## 4. Kế hoạch kiểm thử & Xác thực (Verification Plan)

- **Automated tests:** Chạy pytest kiểm tra tính tương thích của toàn bộ module (`tests/test_retrieval_eval.py`, `tests/test_pipeline.py`, v.v.).
- **Docker execution check:** Container build thành công và script `evals.retrieval_eval` chạy không phát sinh exception kết nối DB.
- **Output validation:** File `evals/benchmark_baseline_law11_top5.json` được tạo đầy đủ các trường `mean_recall_at_k`, `mean_precision_at_k`, và kết quả từng test case.
