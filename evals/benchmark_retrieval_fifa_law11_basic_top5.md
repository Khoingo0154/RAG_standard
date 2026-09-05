# BÁO CÁO KẾT QUẢ BENCHMARK — FIFA LAW 11 - BASIC RETRIEVAL EVALUATION

- **Thời gian thực hiện:** 2026-09-05 15:04:00
- **Chế độ kiểm thử (Mode):** `RETRIEVAL`
- **Cấu hình:** Top-K = `5` | Hybrid Search = `True` | FlashRank Reranker = `True`
- **Tổng số ca:** 4 (Thành công: 4, Lỗi: 0)

## 1. BẢNG TỔNG KẾT CHỈ SỐ TOÀN CỤC

| Chỉ số đo lường | Giá trị đạt được | Ý nghĩa kỹ thuật |
|---|:---:|---|
| **Mean Recall@5** | **`83.3%`** | Tỷ lệ không bỏ sót trang luật cần tìm |
| **Mean Precision@5** | **`55.0%`** | Tỷ lệ chunks tinh khiết (không dính rác) |
| **Mean MRR** | **`0.8750`** | Thứ hạng của đoạn văn đúng đầu tiên |
| **Hit Rate@5** | **`100.0%`** | Tỷ lệ tìm trúng ít nhất 1 trang cần thiết |
| **Độ trễ trung bình** | **`5739.9 ms`** | Thời gian phản hồi trung bình |
| **Độ trễ P95** | **`8658.6 ms`** | 95% số ca phản hồi nhanh hơn mốc này |

---

## 2. BẢNG CHI TIẾT TỪNG CA KIỂM THỬ

| STT | ID | Câu hỏi | Recall | Precision | MRR | Latency | Chunks lấy về | Trích dẫn |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `offside_definition_vi` | Việt vị là gì? | `1.00` | `0.60` | `0.50` | `7212ms` | 49, 50, 51, 52, 80 | — |
| 2 | `offside_no_offence_vi` | Khi nào cầu thủ không bị phạt việt vị? | `1.00` | `0.40` | `1.00` | `8038ms` | 49, 50, 51, 52, 80 | — |
| 3 | `offside_sanction_vi` | Hình phạt khi phạm lỗi việt vị là gì? | `1.00` | `0.60` | `1.00` | `3859ms` | 50, 51, 52, 80, 81 | — |
| 4 | `foul_vi` | Hành vi cản người nào sẽ bị phạt thẻ ? | `0.33` | `0.60` | `1.00` | `3851ms` | 54, 55, 58 | — |