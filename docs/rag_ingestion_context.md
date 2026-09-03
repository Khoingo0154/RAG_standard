# RAG Project — Ingestion Module Context

## Vị trí trong project

Module này là `ingestion/` — một trong các module của RAG project lớn hơn:

```
rag_project/
├── docker-compose.yml
├── .env
├── ingestion/          ← MODULE NÀY (đã build)
├── retrieval/          ← chưa build
├── api/                ← chưa build (FastAPI layer)
└── shared/             ← chưa build (config, logging)
```

`ingestion/` và `retrieval/` là **pure Python, không biết FastAPI**. Chúng chỉ nhận bytes/string, trả dataclass. `api/` sẽ là lớp mỏng bọc ngoài, chuyển HTTP request thành function call vào các module này.

---

## Pipeline ingestion

```
PDF bytes → chunk_pdf() → embed_chunks() → IngestionStore.save()
              ChunkDoc[]    EmbeddedChunk[]   StoreResult
```

`file_id` (UUID) được sinh ra lúc đầu và truyền xuyên suốt — Chroma, MongoDB đều dùng cùng key này. Đây là foreign key để retrieval stage lookup về đúng file/page.

---

## Cấu trúc file

```
ingestion/
├── models.py              # data contracts
├── pipeline.py            # entry point: run_ingestion()
├── services/
│   ├── chunker.py         # PDF → list[ChunkDoc]
│   └── embedder.py        # list[ChunkDoc] → list[EmbeddedChunk]
└── storage/
    └── store.py           # list[EmbeddedChunk] → Chroma + MongoDB
```

---

## Data models (`models.py`)

```python
from dataclasses import dataclass, field
from datetime import datetime
import uuid

def new_id() -> str:
    return str(uuid.uuid4())

@dataclass
class FileRecord:
    """Output của Upload stage."""
    file_id: str
    filename: str
    minio_path: str          # bucket/file_id/filename
    size_bytes: int
    uploaded_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ChunkDoc:
    """Output của Chunk stage. Một chunk = một đoạn text có thể embed."""
    chunk_id: str
    file_id: str             # FK về FileRecord
    text: str
    page_start: int
    page_end: int
    chunk_index: int         # thứ tự chunk trong file, bắt đầu từ 0
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)

@dataclass
class EmbeddedChunk:
    """Output của Embed stage. ChunkDoc + vector."""
    chunk_id: str
    file_id: str
    text: str
    page_start: int
    page_end: int
    chunk_index: int
    vector: list[float]      # 1536 dims (OpenAI) hoặc 384 dims (local)
    model_name: str

    @classmethod
    def from_chunk(cls, chunk: ChunkDoc, vector: list[float], model_name: str) -> "EmbeddedChunk":
        return cls(
            chunk_id=chunk.chunk_id,
            file_id=chunk.file_id,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_index=chunk.chunk_index,
            vector=vector,
            model_name=model_name,
        )

@dataclass
class StoreResult:
    """Output của Store stage."""
    file_id: str
    total_chunks: int
    chroma_ids: list[str]
    mongo_ids: list[str]
    stored_at: datetime = field(default_factory=datetime.utcnow)
```

---

## Pipeline entry point (`pipeline.py`)

```python
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from ingestion.models import FileRecord, StoreResult
from ingestion.services.chunker import ChunkStrategy, chunk_pdf
from ingestion.services.embedder import BaseEmbedder, get_embedder
from ingestion.storage.store import IngestionStore, MetadataStore, VectorStore

logger = logging.getLogger(__name__)

@dataclass
class IngestionConfig:
    chunk_strategy: ChunkStrategy = "token"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embed_provider: str = "local"   # "local" hoặc "openai"
    embed_batch_size: int = 32
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "rag_chunks"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "rag_db"

@dataclass
class IngestionResult:
    file_record: FileRecord
    total_chunks: int
    store_result: StoreResult
    duration_seconds: float

def run_ingestion(
    pdf_bytes: bytes,
    filename: str,
    config: IngestionConfig | None = None,
    embedder: BaseEmbedder | None = None,
) -> IngestionResult:
    cfg = config or IngestionConfig()
    start = datetime.utcnow()

    # Stage 0: FileRecord
    file_id = str(uuid.uuid4())
    file_record = FileRecord(
        file_id=file_id,
        filename=filename,
        minio_path=f"documents/{file_id}/{filename}",
        size_bytes=len(pdf_bytes),
    )

    # Stage 1: Chunk
    chunks = chunk_pdf(
        pdf_bytes=pdf_bytes,
        file_id=file_id,
        strategy=cfg.chunk_strategy,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError(f"PDF '{filename}' không extract được text")

    # Stage 2: Embed
    emb = embedder or get_embedder(cfg.embed_provider)
    embedded_chunks = emb.embed_chunks(chunks, batch_size=cfg.embed_batch_size)

    # Stage 3: Store
    store = IngestionStore(
        vector_store=VectorStore(cfg.chroma_path, cfg.chroma_collection),
        metadata_store=MetadataStore(cfg.mongo_uri, cfg.mongo_db),
    )
    store_result = store.save(embedded_chunks)

    duration = (datetime.utcnow() - start).total_seconds()
    return IngestionResult(
        file_record=file_record,
        total_chunks=len(chunks),
        store_result=store_result,
        duration_seconds=duration,
    )
```

---

## Chunker (`services/chunker.py`)

```python
import io
import uuid
from typing import Literal
from pypdf import PdfReader
from ingestion.models import ChunkDoc

ChunkStrategy = Literal["page", "token"]

def chunk_pdf(
    pdf_bytes: bytes,
    file_id: str,
    strategy: ChunkStrategy = "token",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[ChunkDoc]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = _extract_pages(reader)
    if strategy == "page":
        return _chunk_by_page(pages, file_id)
    else:
        return _chunk_by_token(pages, file_id, chunk_size, chunk_overlap)

def _extract_pages(reader: PdfReader) -> list[dict]:
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page_num": i + 1, "text": text})
    return pages

def _chunk_by_page(pages: list[dict], file_id: str) -> list[ChunkDoc]:
    return [
        ChunkDoc(
            chunk_id=str(uuid.uuid4()),
            file_id=file_id,
            text=p["text"],
            page_start=p["page_num"],
            page_end=p["page_num"],
            chunk_index=i,
        )
        for i, p in enumerate(pages)
    ]

def _chunk_by_token(
    pages: list[dict],
    file_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkDoc]:
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap phải nhỏ hơn chunk_size")

    full_text = ""
    boundaries: list[tuple[int, int, int]] = []
    for page in pages:
        start = len(full_text)
        full_text += page["text"] + "\n"
        boundaries.append((start, len(full_text), page["page_num"]))

    chunks = []
    step = chunk_size - chunk_overlap
    pos = 0
    chunk_index = 0
    while pos < len(full_text):
        end_pos = min(pos + chunk_size, len(full_text))
        text = full_text[pos:end_pos].strip()
        if text:
            page_start, page_end = _find_page_range(pos, end_pos, boundaries)
            chunks.append(ChunkDoc(
                chunk_id=str(uuid.uuid4()),
                file_id=file_id,
                text=text,
                page_start=page_start,
                page_end=page_end,
                chunk_index=chunk_index,
            ))
            chunk_index += 1
        pos += step
    return chunks

def _find_page_range(
    char_start: int,
    char_end: int,
    boundaries: list[tuple[int, int, int]],
) -> tuple[int, int]:
    page_start = None
    page_end = None
    for b_start, b_end, page_num in boundaries:
        if b_end > char_start and b_start < char_end:
            if page_start is None:
                page_start = page_num
            page_end = page_num
    return (page_start or 1, page_end or 1)
```

---

## Embedder (`services/embedder.py`)

```python
import os
import time
from abc import ABC, abstractmethod
from ingestion.models import ChunkDoc, EmbeddedChunk

class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    def embed_chunks(self, chunks: list[ChunkDoc], batch_size: int = 32) -> list[EmbeddedChunk]:
        results = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = self._embed_batch([c.text for c in batch])
            for chunk, vector in zip(batch, vectors):
                results.append(EmbeddedChunk.from_chunk(chunk, vector, self.model_name))
            if i + batch_size < len(chunks):
                time.sleep(0.1)
        return results

class OpenAIEmbedder(BaseEmbedder):
    """text-embedding-3-small, 1536 dims. Cần OPENAI_API_KEY."""
    MODEL = "text-embedding-3-small"

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    @property
    def model_name(self) -> str:
        return self.MODEL

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.MODEL, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

class LocalEmbedder(BaseEmbedder):
    """all-MiniLM-L6-v2, 384 dims. Không cần API key."""
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.encode(texts, convert_to_numpy=True)]

def get_embedder(provider: str = "local") -> BaseEmbedder:
    if provider == "openai":
        return OpenAIEmbedder()
    elif provider == "local":
        return LocalEmbedder()
    raise ValueError(f"Unknown provider: {provider}")
```

---

## Store (`storage/store.py`)

```python
import logging
from datetime import datetime
from ingestion.models import EmbeddedChunk, StoreResult

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB wrapper. pip install chromadb"""

    def __init__(self, persist_path: str = "./chroma_db", collection_name: str = "rag_chunks"):
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[EmbeddedChunk]) -> list[str]:
        if not chunks:
            return []
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[c.vector for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{
                "file_id": c.file_id,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "chunk_index": c.chunk_index,
                "model_name": c.model_name,
            } for c in chunks],
        )
        return [c.chunk_id for c in chunks]

    def delete_by_file(self, file_id: str) -> None:
        results = self._collection.get(where={"file_id": file_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

class MetadataStore:
    """MongoDB wrapper. pip install pymongo"""

    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", db_name: str = "rag_db", collection_name: str = "chunks"):
        from pymongo import MongoClient
        self._col = MongoClient(mongo_uri)[db_name][collection_name]
        self._col.create_index("chunk_id", unique=True)
        self._col.create_index("file_id")

    def insert_chunks(self, chunks: list[EmbeddedChunk]) -> list[str]:
        if not chunks:
            return []
        docs = [{
            "chunk_id": c.chunk_id,
            "file_id": c.file_id,
            "text": c.text,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "chunk_index": c.chunk_index,
            "model_name": c.model_name,
            "inserted_at": datetime.utcnow(),
        } for c in chunks]
        result = self._col.insert_many(docs, ordered=False)
        return [str(oid) for oid in result.inserted_ids]

    def delete_by_file(self, file_id: str) -> int:
        return self._col.delete_many({"file_id": file_id}).deleted_count

class IngestionStore:
    """
    Facade duy nhất cho Store stage.
    Rollback: Chroma upsert trước → MongoDB insert sau.
    Nếu MongoDB fail → xóa Chroma để tránh split-brain.
    """

    def __init__(self, vector_store: VectorStore, metadata_store: MetadataStore):
        self._vector = vector_store
        self._metadata = metadata_store

    def save(self, chunks: list[EmbeddedChunk]) -> StoreResult:
        if not chunks:
            raise ValueError("Không có chunk nào để lưu")

        file_id = chunks[0].file_id

        # Step 1: Chroma
        try:
            chroma_ids = self._vector.upsert(chunks)
        except Exception as e:
            raise RuntimeError(f"Chroma upsert failed: {e}") from e

        # Step 2: MongoDB — rollback Chroma nếu fail
        try:
            mongo_ids = self._metadata.insert_chunks(chunks)
        except Exception as e:
            try:
                self._vector.delete_by_file(file_id)
            except Exception:
                pass
            raise RuntimeError(f"MongoDB insert failed (Chroma rolled back): {e}") from e

        return StoreResult(
            file_id=file_id,
            total_chunks=len(chunks),
            chroma_ids=chroma_ids,
            mongo_ids=mongo_ids,
        )
```

---

## Requirements

```
pypdf>=4.0.0
openai>=1.0.0
sentence-transformers>=2.0.0
chromadb>=0.5.0
pymongo>=4.0.0
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
python-multipart>=0.0.6
reportlab>=4.0.0
pytest>=7.0.0
```

---

## Những gì chưa build (TODO)

### `retrieval/` module

Pipeline: `query string → embed → similarity search Chroma → lookup MongoDB → assemble context → LLM call → response`

Data contracts cần build:
- `QueryRequest` — query string + optional file_id filter + top_k
- `RetrievedChunk` — chunk_id, text, score, page_start, page_end, filename
- `RAGResponse` — answer string + list[RetrievedChunk] làm sources

### `api/` layer (FastAPI)

Hai endpoints chính:
- `POST /ingest` — nhận file upload, gọi `run_ingestion()`, trả `IngestionResult`
- `POST /query` — nhận query string, gọi retrieval pipeline, stream response qua SSE

### `shared/` utilities

- `config.py` — load `.env` một chỗ, expose typed config object
- `logging.py` — structured logging với `request_id` tracking

### MinIO integration

`minio_path` trong `FileRecord` hiện là placeholder string. Cần thêm `storage/minio_client.py` để thực sự upload bytes lên MinIO trước khi chunk.

---

## Design decisions đã chốt

**`file_id` sinh UUID ngay lúc tạo `FileRecord`**, không phải sau khi lưu DB. Lý do: các stage sau cần `file_id` làm FK nhưng chưa có DB connection.

**`BaseEmbedder` interface** tách biệt hoàn toàn khỏi pipeline logic. Swap `OpenAIEmbedder` ↔ `LocalEmbedder` bằng config, không sửa code pipeline.

**Chroma không lưu full text**, chỉ lưu vector + minimal metadata. Full text lưu trong MongoDB. Lý do: tránh duplicate data lớn, MongoDB search/filter linh hoạt hơn.

**Rollback strategy**: Chroma upsert trước, MongoDB sau. Nếu MongoDB fail thì xóa Chroma. Không phải distributed transaction nhưng đủ cho dev/staging. Production nên dùng outbox pattern hoặc message queue.
