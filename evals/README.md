# Retrieval evaluation

`fifa_law11_basic.json` là bộ **gold data** ban đầu: mỗi câu hỏi có các trang PDF đã được người đánh giá đọc và đánh dấu là cần thiết.

## Cách chạy benchmark nhanh

Bạn có thể chạy nhanh bằng file script có sẵn:

```cmd
run_eval.cmd
```

Hoặc chạy lệnh trực tiếp qua Docker Compose:

```cmd
:: 1. Chạy đánh giá và in kết quả ra màn hình:
docker compose --profile swagger run --rm api python -m evals.retrieval_eval --file-id 8f525964-a864-43ef-b87b-34f6482d44f1 --top-k 5

:: 2. Chạy đánh giá và lưu kết quả JSON làm mốc đối chứng (baseline):
docker compose --profile swagger run --rm api python -m evals.retrieval_eval --file-id 8f525964-a864-43ef-b87b-34f6482d44f1 --top-k 5 --output evals/benchmark_baseline_law11_top5.json
```

## Kết quả quan trọng

- `recall_at_k`: tỷ lệ trang gold xuất hiện trong top-k. Recall 1.0 nghĩa là không bỏ sót trang gold.
- `precision_at_k`: tỷ lệ chunk trong top-k thực sự bao phủ trang gold. Precision thấp nghĩa là có nhiều chunk nhiễu.

## Ba cách tạo test set

1. **Gắn nhãn thủ công (khuyến nghị lúc đầu):** đọc PDF, viết câu hỏi thật, rồi ghi `expected_pages` hoặc `expected_chunk_ids`. Đây là cách đáng tin nhất.
2. **LLM tạo nháp:** dùng Gemini tạo câu hỏi từ từng phần PDF, nhưng người làm dự án phải kiểm tra và sửa lại gold pages trước khi dùng làm benchmark.
3. **Log người dùng thật:** lưu câu hỏi thất bại, sửa lại expected pages sau khi review. Cách này giúp bộ test gần với nhu cầu sử dụng nhất.

Khi dự án lớn hơn, nên chuyển từ page-level sang `expected_chunk_ids`, thêm nhiều loại câu hỏi và theo dõi Recall@k, Precision@k, MRR trước/sau mỗi thay đổi embedding, chunk size, query expansion hoặc reranking.
