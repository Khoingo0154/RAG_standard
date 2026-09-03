# Kế hoạch Thực thi: Nâng cấp Advanced RAG (Hybrid Search BM25 + FlashRank Reranker)

> **Mục tiêu:** Nâng cấp hệ thống từ Standard RAG (chỉ Vector Search) lên Advanced RAG kết hợp Vector Search (Dense) + BM25 (Sparse) + Reranking (FlashRank Cross-Encoder), đẩy chỉ số Precision@5 từ 70% lên >90%.

---

## 1. Sơ đồ kiến trúc thực thi (Runtime Pipeline)

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

---

## 2. Các thành phần kỹ thuật chi tiết

### 2.1. Branch B: BM25 Sparse Search
- **Thư viện:** `rank-bm25` (thuần Python, siêu nhẹ, không phụ thuộc C++/GPU).
- **Corpus:** Nạp toàn bộ text các chunks từ CSDL (ChromaDB / MongoDB) theo `file_id`.
- **Tokenization:** Tokenizer tách từ tiếng Việt / Anh chuẩn hóa chữ thường, loại bỏ ký tự đặc biệt.
- **Cache:** Lưu BM25 index theo `file_id` trong memory (hoặc cache đệm) để các truy vấn sau đạt tốc độ dưới 1ms.

### 2.2. Reciprocal Rank Fusion (RRF)
- Kết hợp thứ hạng từ 2 danh sách Top-15:
  $$RRF\_Score(d) = \sum_{m \in \{Vector, BM25\}} \frac{1}{60 + Rank_m(d)}$$
- Khử trùng lặp theo `chunk_id`.
- Lấy tập ứng viên ~20–25 chunks chất lượng cao.

### 2.3. Reranking qua FlashRank
- **Thư viện:** `flashrank` (dựa trên ONNX Runtime, không cần PyTorch nặng nề).
- **Model:** `ms-marco-MiniLM-L-12-v2` (~40MB).
- **Hiệu năng:** Chấm điểm 20–25 chunks trong < 100ms trên CPU.
- **Cắt lọc:** Chọn lọc Top-5 chunks tinh khiết nhất đưa vào LLM Generator.

---

## 3. Lộ trình phát triển tiếp theo (Next Milestones)

1. **Milestone 1:** Conversational Memory & Multi-turn Chat (cho Telegram Bot và API).
2. **Milestone 2:** Parent-Document / Small-to-Big Retrieval.
3. **Milestone 3:** Query Transformation & HyDE.
4. **Milestone 4:** RAG Triad Evaluation (Faithfulness & Relevance).
5. **Milestone 5:** GraphRAG (Neo4j).
