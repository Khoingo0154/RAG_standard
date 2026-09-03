import logging
import re
from typing import Optional

from shared.config import settings

logger = logging.getLogger(__name__)


STOP_WORDS = {
    "the", "of", "and", "in", "to", "a", "is", "for", "on", "that", "by", "this",
    "with", "it", "not", "or", "be", "are", "from", "at", "as", "all", "have",
    "an", "was", "we", "will", "can", "if", "page", "laws", "game", "fifa"
}


def tokenize(text: str) -> list[str]:
    """Tách từ đơn giản, loại bỏ stopwords phổ biến, giữ lại từ khóa nội dung."""
    if not text:
        return []
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

class BM25Retriever:
    """Tìm kiếm từ khóa chính xác (Sparse Keyword Search) sử dụng thuật toán BM25Okapi."""

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        mongo_db: Optional[str] = None,
    ):
        self._chroma_path = chroma_path or settings.CHROMA_PATH
        self._collection_name = collection_name or settings.CHROMA_COLLECTION
        self._mongo_uri = mongo_uri or settings.MONGO_URI
        self._mongo_db = mongo_db or settings.MONGO_DB

        self._bm25 = None
        self._indexed_chunks: list[dict] = []
        self._indexed_file_id: Optional[str] = None

    def _load_chunks_from_chroma(self, file_id: Optional[str] = None) -> list[dict]:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self._chroma_path)
            collection = client.get_collection(self._collection_name)

            where_filter = {"file_id": file_id} if file_id else None
            data = collection.get(where=where_filter, include=["documents", "metadatas"])

            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            chunks = []
            for cid, doc, meta in zip(ids, documents, metadatas):
                meta = meta or {}
                chunks.append({
                    "chunk_id": cid,
                    "file_id": meta.get("file_id", file_id or ""),
                    "text": doc or "",
                    "page_start": meta.get("page_start", 1),
                    "page_end": meta.get("page_end", 1),
                    "chunk_index": meta.get("chunk_index", 0),
                })
            return chunks
        except Exception as e:
            logger.debug(f"Không thể nạp chunks từ ChromaDB cho BM25: {e}")
            return []

    def _load_chunks_from_mongo(self, file_id: Optional[str] = None) -> list[dict]:
        try:
            from pymongo import MongoClient

            client = MongoClient(self._mongo_uri, serverSelectionTimeoutMS=2000)
            col = client[self._mongo_db]["chunks"]

            query = {"file_id": file_id} if file_id else {}
            cursor = col.find(query)

            chunks = []
            for doc in cursor:
                chunks.append({
                    "chunk_id": doc.get("chunk_id", str(doc.get("_id"))),
                    "file_id": doc.get("file_id", ""),
                    "text": doc.get("text", ""),
                    "page_start": doc.get("page_start", 1),
                    "page_end": doc.get("page_end", 1),
                    "chunk_index": doc.get("chunk_index", 0),
                })
            return chunks
        except Exception as e:
            logger.debug(f"Không thể nạp chunks từ MongoDB cho BM25: {e}")
            return []

    def build_index(self, chunks: Optional[list[dict]] = None, file_id: Optional[str] = None) -> bool:
        """Xây dựng BM25 index từ danh sách chunks truyền vào hoặc tự động nạp từ CSDL."""
        from rank_bm25 import BM25Okapi

        if chunks is None:
            chunks = self._load_chunks_from_chroma(file_id)
            if not chunks:
                chunks = self._load_chunks_from_mongo(file_id)

        if not chunks:
            logger.warning("BM25Retriever không tìm thấy chunks nào để index.")
            self._bm25 = None
            self._indexed_chunks = []
            self._indexed_file_id = None
            return False

        tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._indexed_chunks = chunks
        self._indexed_file_id = file_id
        logger.info(f"BM25Retriever đã đánh chỉ mục {len(chunks)} chunks (file_id={file_id}).")
        return True

    def search(
        self,
        query: str,
        top_k: int = 15,
        file_id: Optional[str] = None,
    ) -> list[dict]:
        """Tìm kiếm các chunks khớp từ khóa BM25 tốt nhất."""
        if not query or not query.strip():
            return []

        # Kiểm tra nếu index chưa nạp hoặc file_id thay đổi
        if self._bm25 is None or self._indexed_file_id != file_id:
            success = self.build_index(file_id=file_id)
            if not success or not self._bm25:
                return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        scored_chunks = []
        for idx, score in enumerate(scores):
            if score > 0:
                chunk = dict(self._indexed_chunks[idx])
                chunk["score"] = float(score)
                scored_chunks.append(chunk)

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
