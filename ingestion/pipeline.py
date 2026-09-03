import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from ingestion.models import FileRecord, StoreResult
from ingestion.services.chunker import ChunkStrategy, chunk_pdf
from ingestion.services.embedder import BaseEmbedder, get_embedder
from ingestion.storage.object_store import MinioObjectStore
from ingestion.storage.store import IngestionStore, MetadataStore, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    chunk_strategy: ChunkStrategy = "token"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embed_provider: str = "local"
    embed_batch_size: int = 32
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "rag_chunks"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "rag_db"
    minio_enabled: bool = False
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-documents"
    minio_secure: bool = False


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
    object_store: MinioObjectStore | None = None,
) -> IngestionResult:
    cfg = config or IngestionConfig()
    start = datetime.now(timezone.utc)

    file_id = str(uuid.uuid4())
    file_record = FileRecord(
        file_id=file_id,
        filename=filename,
        minio_path=f"documents/{file_id}/{filename}",
        size_bytes=len(pdf_bytes),
    )
    logger.info(f"[{file_id}] Starting ingestion: {filename} ({len(pdf_bytes):,} bytes)")

    storage = object_store
    uploaded_to_minio = False
    if cfg.minio_enabled:
        storage = storage or MinioObjectStore(
            endpoint=cfg.minio_endpoint,
            access_key=cfg.minio_access_key,
            secret_key=cfg.minio_secret_key,
            bucket=cfg.minio_bucket,
            secure=cfg.minio_secure,
        )

    try:
        if storage:
            file_record.minio_path = storage.upload_pdf(file_id, filename, pdf_bytes)
            uploaded_to_minio = True
            logger.info(f"[{file_id}] Original PDF stored in MinIO: {file_record.minio_path}")

        logger.info(f"[{file_id}] Chunking (strategy={cfg.chunk_strategy})")
        chunks = chunk_pdf(
            pdf_bytes=pdf_bytes,
            file_id=file_id,
            filename=filename,
            strategy=cfg.chunk_strategy,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        logger.info(f"[{file_id}] Got {len(chunks)} chunks")

        if not chunks:
            raise RuntimeError(f"PDF '{filename}' không extract được text")

        emb = embedder or get_embedder(cfg.embed_provider, task_type="RETRIEVAL_DOCUMENT")
        logger.info(f"[{file_id}] Embedding với {emb.model_name}, batch_size={cfg.embed_batch_size}")
        embedded_chunks = emb.embed_chunks(chunks, batch_size=cfg.embed_batch_size)
        logger.info(f"[{file_id}] Embedded {len(embedded_chunks)} chunks")

        logger.info(f"[{file_id}] Storing to Chroma + MongoDB")
        store = IngestionStore(
            vector_store=VectorStore(cfg.chroma_path, cfg.chroma_collection),
            metadata_store=MetadataStore(cfg.mongo_uri, cfg.mongo_db),
        )
        store_result = store.save(embedded_chunks)
    except Exception:
        if uploaded_to_minio and storage:
            try:
                storage.delete(file_record.minio_path)
                logger.info(f"[{file_id}] Rolled back original PDF in MinIO")
            except Exception as rollback_error:
                logger.warning(f"[{file_id}] MinIO rollback failed: {rollback_error}")
        raise

    duration = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(f"[{file_id}] Done in {duration:.2f}s — {len(chunks)} chunks stored")

    return IngestionResult(
        file_record=file_record,
        total_chunks=len(chunks),
        store_result=store_result,
        duration_seconds=duration,
    )
