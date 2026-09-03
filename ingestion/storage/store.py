import logging
from ingestion.models import EmbeddedChunk, StoreResult

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Lớp bọc ChromaDB — nơi lưu VECTOR.
    Dùng để search similarity sau này (retrieval).

    Cài đặt: pip install chromadb
    """

    def __init__(self, persist_path: str = "./chroma_db", collection_name: str = "rag_chunks"):
        """
        Mở (hoặc tự tạo) database Chroma.
        persist_path: nơi Chroma lưu file xuống đĩa
        collection_name: tên "bảng" chứa chunks
        """
        import chromadb
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # đo độ tương đồng bằng cosine
        )

    def upsert(self, chunks: list[EmbeddedChunk]) -> list[str]:
        """
        Ghi (hoặc cập nhật) chunks vào Chroma.
        upSert = update + insert: có id rồi thì cập nhật, chưa có thì thêm mới.
        Chỉ lưu vector + metadata tối thiểu (KHÔNG lưu text đầy đủ).
        Trả về danh sách chunk_id đã ghi.
        """
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
                "filename": c.filename,
            } for c in chunks],
        )
        return [c.chunk_id for c in chunks]

    def delete_by_file(self, file_id: str) -> None:
        """
        Xóa toàn bộ chunk của 1 file khỏi Chroma.
        Dùng cho rollback khi MongoDB fail.
        """
        results = self._collection.get(where={"file_id": file_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])


class MetadataStore:
    """
    Lớp bọc MongoDB — nơi lưu TEXT gốc + metadata đầy đủ.
    Dùng để đọc lại nội dung, filter, trả source cho người dùng.

    Cài đặt: pip install pymongo
    """

    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", db_name: str = "rag_db", collection_name: str = "chunks"):
        """
        Kết nối MongoDB.
        mongo_uri: địa chỉ server MongoDB
        db_name: tên database
        collection_name: tên "bảng" chứa chunks
        """
        from pymongo import MongoClient
        self._col = MongoClient(mongo_uri)[db_name][collection_name]
        self._col.create_index("chunk_id", unique=True)  # cấm 2 chunk trùng id
        self._col.create_index("file_id")                # tìm nhanh theo file

    def insert_chunks(self, chunks: list[EmbeddedChunk]) -> list[str]:
        """
        Chèn chunks vào MongoDB dưới dạng document (kiểu JSON).
        Trả về danh sách id MongoDB đã tạo.
        """
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
            "filename": c.filename,
        } for c in chunks]
        result = self._col.insert_many(docs, ordered=False)
        return [str(oid) for oid in result.inserted_ids]

    def delete_by_file(self, file_id: str) -> int:
        """
        Xóa toàn bộ chunk của 1 file khỏi MongoDB.
        Trả về số document đã xóa.
        """
        return self._col.delete_many({"file_id": file_id}).deleted_count


class IngestionStore:
    """
    Facade duy nhất cho Store stage.
    pipeline chỉ cần gọi IngestionStore.save() là xong.

    Rollback strategy:
    1. Upsert Chroma TRƯỚC
    2. Insert MongoDB SAU
    3. Nếu MongoDB fail → xóa Chroma để tránh split-brain
       (2 DB không đồng bộ: có vector nhưng không có text)
    """

    def __init__(self, vector_store: VectorStore, metadata_store: MetadataStore):
        self._vector = vector_store
        self._metadata = metadata_store

    def save(self, chunks: list[EmbeddedChunk]) -> StoreResult:
        """
        Lưu chunks vào cả 2 DB.
        Nếu fail ở giữa chừng → rollback về trạng thái như chưa làm gì.
        """
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
