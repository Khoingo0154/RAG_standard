# Ingestion Module — Design

## Vị trí trong project

```
rag_project/
├── ingestion/    ← MODULE NÀY
├── retrieval/    ← chưa build
├── api/          ← chưa build
└── shared/       ← chưa build
```

## Pipeline

```
PDF bytes → FileRecord
          → chunk_pdf()    → [ChunkDoc]
          → embed_chunks() → [EmbeddedChunk]
          → store.save()   → StoreResult
```

## Data models (`models.py`)

| Class | Stage | Fields chính |
|---|---|---|
| `FileRecord` | Đầu vào | file_id, filename, minio_path, size_bytes |
| `ChunkDoc` | Sau chunk | chunk_id, file_id, text, page_start, page_end, chunk_index |
| `EmbeddedChunk` | Sau embed | kế thừa ChunkDoc + vector: list[float], model_name |
| `StoreResult` | Sau store | file_id, total_chunks, chroma_ids, mongo_ids |

## Chunker (`services/chunker.py`) ✅

**Strategy mặc định**: token (sliding window) — gom text toàn bộ PDF, cắt đều với chunk_size và chunk_overlap. Ghi nhận page_start / page_end dựa trên vị trí ký tự trong boundaries.

Có hỗ trợ page strategy (1 trang = 1 chunk) như một phương án thay thế.

**Clean data**: text được làm sạch qua `_clean_text()` (xuống dòng, header/footer "1/100", "Trang 1") và lọc chunk rác qua `_is_valid_chunk()` (dưới 50 ký tự hoặc tỷ lệ chữ <30%).

## Embedder (`services/embedder.py`)

Interface `BaseEmbedder` với 3 implementation:

| Embedder | Model | Chiều | Ghi chú |
|---|---|---|---|
| `OpenAIEmbedder` | text-embedding-3-small | 1536 | Cần API key |
| `LocalEmbedder` | all-MiniLM-L6-v2 | 384 | Chạy local, free |
| `OllamaEmbedder` | phi4-mini | tùy model | Gọi server 100.71.230.7:11434 |

Factory function `get_embedder(provider)` trả về embedder tương ứng.

## Store (`storage/store.py`) ✅

- `VectorStore` — ChromaDB: lưu vector + metadata tối thiểu
- `MetadataStore` — MongoDB: lưu text gốc + metadata đầy đủ
- `IngestionStore` — Facade: upsert Chroma trước, insert MongoDB sau, rollback Chroma nếu MongoDB fail

## Pipeline entry (`pipeline.py`)

`run_ingestion(pdf_bytes, filename, config, embedder)`:
1. Sinh file_id → FileRecord
2. chunk_pdf() → list[ChunkDoc]
3. embedder.embed_chunks() → list[EmbeddedChunk]
4. IngestionStore.save() → StoreResult

Config qua `IngestionConfig`: chunk_strategy, chunk_size, chunk_overlap, embed_provider, chroma_path, mongo_uri.

## Design decisions đã chốt

- `file_id` là UUID sinh ngay đầu pipeline, truyền xuyên suốt
- `BaseEmbedder` interface → đổi embedder bằng config, không sửa pipeline
- Chroma lưu vector + minimal metadata; MongoDB lưu text đầy đủ
- Rollback: Chroma trước → MongoDB sau. Nếu MongoDB fail, xóa Chroma
- Token chunking mặc định với sliding window
- Ollama embedder mặc định dùng phi4-mini trên server 100.71.230.7:11434

## Chưa làm (TODO)

- [x] Tạo models.py ✅
- [x] Tạo storage/store.py ✅
- [ ] Tạo pipeline.py
- [ ] Cài dependencies: pypdf, chromadb, pymongo, sentence-transformers, requests
- [ ] Kiểm thử pipeline với file PDF thật
