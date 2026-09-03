import logging
from typing import Optional

from retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, chroma_path: str = "./chroma_db", collection_name: str = "rag_chunks"):
        import chromadb
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        file_id: Optional[str] = None,
    ) -> list[dict]:
        where_filter = None
        if file_id:
            where_filter = {"file_id": file_id}

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                doc = results["documents"][0][i] if results["documents"] else ""
                dist = results["distances"][0][i] if results["distances"] else 0.0
                hits.append({
                    "chunk_id": chunk_id,
                    "text": doc,
                    "file_id": meta.get("file_id", ""),
                    "page_start": meta.get("page_start", 0),
                    "page_end": meta.get("page_end", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": dist,
                    "score": 1.0 - dist,
                })
        return hits
