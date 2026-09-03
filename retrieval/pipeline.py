import logging
import time
from typing import Optional

from ingestion.services.embedder import get_embedder
from retrieval.bm25_retriever import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.generator import Generator
from retrieval.models import RAGResponse, RetrievedChunk
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever
from retrieval.router import QueryIntent, QueryRouter
from shared.config import settings

logger = logging.getLogger(__name__)

_GLOBAL_BM25 = None
_GLOBAL_RERANKER = None


def get_bm25_retriever() -> BM25Retriever:
    global _GLOBAL_BM25
    if _GLOBAL_BM25 is None:
        _GLOBAL_BM25 = BM25Retriever()
    return _GLOBAL_BM25


def get_reranker() -> Reranker:
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        _GLOBAL_RERANKER = Reranker()
    return _GLOBAL_RERANKER
logger = logging.getLogger(__name__)


# Glossary theo domain FIFA. Mở rộng truy vấn ngắn bằng thuật ngữ xuất hiện trong PDF
# để embedding có đủ tín hiệu tìm đúng Luật, nhưng không thay đổi câu hỏi người dùng thấy.
QUERY_EXPANSIONS = {
    "không bị phạt việt vị": "no offside offence exception goal kick corner kick throw-in Law 11",
    "không việt vị": "no offside offence exception goal kick corner kick throw-in Law 11",
    "hình phạt khi phạm lỗi việt vị": "sanction offside offence indirect free kick Law 11",
    "hình phạt việt vị": "sanction offside offence indirect free kick Law 11",
    "việt vị": "offside position offside offence Law 11 Offside",
    "phạt đền": "penalty kick Law 14 The Penalty Kick",
    "penalty": "penalty kick Law 14 The Penalty Kick",
    "phạt góc": "corner kick Law 17 The Corner Kick",
    "ném biên": "throw-in Law 15 The Throw-in",
    "phát bóng": "goal kick Law 16 The Goal Kick",
    "thẻ đỏ": "red card sending-off offence Law 12 Fouls and Misconduct",
    "thẻ vàng": "yellow card cautionable offence Law 12 Fouls and Misconduct",
    "phạt thẻ": "disciplinary action caution yellow card red card Law 12 Fouls and Misconduct",
    "cản người": "impeding progress of opponent Law 12 Fouls and Misconduct",
    "chạm tay": "handball offence Law 12 Fouls and Misconduct",
    "dùng tay": "handball offence Law 12 Fouls and Misconduct",
}


def get_search_terms(query: str) -> tuple[str, str, str]:
    """Tách câu hỏi thành (retrieval_query cho embedding, sparse_query cho BM25, rerank_query cho FlashRank)."""
    normalized = query.casefold()
    additions = []
    for phrase, term in QUERY_EXPANSIONS.items():
        if phrase in normalized:
            additions.append(term)
    if not additions:
        return query, query, query
    english_terms = "; ".join(additions)
    retrieval_query = f"{query}\nRelated FIFA terminology: {english_terms}"
    sparse_query = english_terms
    rerank_query = f"{query} {english_terms}"
    return retrieval_query, sparse_query, rerank_query


def expand_query_for_retrieval(query: str) -> str:
    retrieval_query, _, _ = get_search_terms(query)
    return retrieval_query


def retrieve_hybrid_and_rerank(
    query: str,
    query_vector: list[float],
    sparse_query: Optional[str] = None,
    rerank_query: Optional[str] = None,
    file_id: Optional[str] = None,
    top_k: int = 5,
    top_candidates: int = 15,
    use_hybrid: bool = True,
    use_rerank: bool = True,
    retriever: Optional[Retriever] = None,
    bm25_retriever: Optional[BM25Retriever] = None,
    reranker: Optional[Reranker] = None,
) -> list[dict]:
    """Thực thi luồng tìm kiếm nâng cao: Dense (Vector) + Sparse (BM25) -> RRF Fusion -> FlashRank Reranker."""
    dense_retriever = retriever or Retriever(
        chroma_path=settings.CHROMA_PATH,
        collection_name=settings.CHROMA_COLLECTION,
    )
    fetch_k = top_candidates if (use_hybrid or use_rerank) else top_k

    # 1. Branch A: Dense Vector Search
    vector_hits = dense_retriever.search(
        query_vector=query_vector,
        top_k=fetch_k,
        file_id=file_id,
    )

    if not use_hybrid:
        candidates = vector_hits
    else:
        # 2. Branch B: Sparse BM25 Search
        sparse_retriever = bm25_retriever or get_bm25_retriever()
        bm25_hits = sparse_retriever.search(
            query=sparse_query or query,
            top_k=fetch_k,
            file_id=file_id,
        )
        logger.info(
            f"🔀 [BRANCH: HYBRID SEARCH] Dense hits: {len(vector_hits)} | Sparse BM25 hits: {len(bm25_hits)}"
        )

        # 3. RRF Fusion
        candidates = reciprocal_rank_fusion(vector_hits, bm25_hits, k=60, top_n=25)
        logger.info(f"🧬 [HÀM: RRF Fusion] Hợp nhất được {len(candidates)} chunks ứng viên độc nhất")

    # 4. Reranking Stage
    if not use_rerank or not candidates:
        final_hits = candidates[:top_k]
    else:
        cross_reranker = reranker or get_reranker()
        final_hits = cross_reranker.rerank(query=rerank_query or query, candidates=candidates, top_k=top_k)
        logger.info(f"🎯 [HÀM: FlashRank Reranker] Đã lọc lấy Top-{len(final_hits)} chunks tinh khiết")

    return final_hits


def search_and_generate(
    query: str,
    file_id: Optional[str] = None,
    top_k: int = 5,
    embed_provider: Optional[str] = None,
    llm_provider: Optional[str] = None,
    router: Optional[QueryRouter] = None,
    retriever: Optional[Retriever] = None,
    bm25_retriever: Optional[BM25Retriever] = None,
    reranker: Optional[Reranker] = None,
    use_hybrid: bool = True,
    use_rerank: bool = True,
) -> RAGResponse:
    start = time.time()
    logger.info("=" * 60)
    logger.info(f"📥 [STEP 1: NHẬN CÂU HỎI] Query: \"{query}\" (file_id={file_id}, top_k={top_k})")

    # 1. Phân loại intent của câu hỏi
    logger.info("🧠 [STEP 2: QUERY ROUTING] Đang phân loại ý định câu hỏi bằng QueryRouter...")
    query_router = router or QueryRouter()
    intent = query_router.classify(query)
    logger.info(f"🎯 [ROUTER DECISION] Ý định được xác định: === {intent.value} ===")

    generator = Generator(provider=llm_provider or settings.LLM_PROVIDER)

    # 2. Xử lý câu hỏi xã giao / chào hỏi (CHITCHAT)
    if intent == QueryIntent.CHITCHAT:
        logger.info("⚡ [BRANCH: CHITCHAT] -> Gọi hàm: Generator.generate_chitchat() [Bỏ qua Retriever.search / ChromaDB]")
        answer = generator.generate_chitchat(query)
        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"✅ [HOÀN THÀNH: CHITCHAT] Trả lời trong {elapsed_ms:.1f}ms (sources: 0 chunks)")
        logger.info("=" * 60)
        return RAGResponse(
            answer=answer,
            sources=[],
            query_time_ms=elapsed_ms,
            intent=intent.value,
        )

    # 3. Xử lý câu hỏi ngoài phạm vi bóng đá (OUT_OF_SCOPE)
    if intent == QueryIntent.OUT_OF_SCOPE:
        logger.info("🚫 [BRANCH: OUT_OF_SCOPE] -> Trả về thông báo từ chối trực tiếp [Bỏ qua Retriever.search và Generator]")
        answer = (
            "Xin lỗi bạn, tôi là trợ lý chuyên môn về tài liệu Luật bóng đá FIFA (Law_fifa.pdf). "
            "Tôi chỉ có thể giải đáp các câu hỏi liên quan đến luật và quy định bóng đá. "
            "Bạn có câu hỏi nào về luật thi đấu không?"
        )
        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"✅ [HOÀN THÀNH: OUT_OF_SCOPE] Phản hồi trong {elapsed_ms:.1f}ms")
        logger.info("=" * 60)
        return RAGResponse(
            answer=answer,
            sources=[],
            query_time_ms=elapsed_ms,
            intent=intent.value,
        )

    # 4. Với DOCUMENT_QUERY -> Thực thi luồng RAG đầy đủ
    logger.info("📚 [BRANCH: DOCUMENT_QUERY] -> Kích hoạt chuỗi hàm: expand_query_for_retrieval() ➔ BaseEmbedder.embed_query() ➔ Retriever.search() ➔ Generator.generate()")
    emb_provider = embed_provider or settings.EMBED_PROVIDER
    embedder = get_embedder(emb_provider, task_type="RETRIEVAL_QUERY")
    
    retrieval_query, sparse_query, rerank_query = get_search_terms(query)
    if retrieval_query != query:
        logger.info(f"🔍 [HÀM: get_search_terms()] Đã mở rộng truy vấn:\n{retrieval_query}")
    logger.info(f"🔢 [HÀM: {embedder.__class__.__name__}.embed_query()] Tạo vector với model '{embedder.model_name}'...")
    query_vector = embedder.embed_query(retrieval_query)

    logger.info("🔎 [STEP 3: ADVANCED RETRIEVAL] Kích hoạt Hybrid Search (BM25 + Vector) và FlashRank Reranker...")
    hits = retrieve_hybrid_and_rerank(
        query=query,
        query_vector=query_vector,
        sparse_query=sparse_query,
        rerank_query=rerank_query,
        file_id=file_id,
        top_k=top_k,
        top_candidates=15,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        retriever=retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
    )
    scores = [round(h['score'], 4) for h in hits]
    logger.info(f"📦 [ADVANCED RETRIEVAL] Kết quả cuối: Lấy được {len(hits)} chunks (Scores: {scores})")

    logger.info(f"🤖 [HÀM: Generator.generate()] Gửi ngữ cảnh tới {generator._provider.upper()} ({generator._gemini_model}) để sinh câu trả lời...")
    answer, sources = generator.generate(query, hits)

    elapsed_ms = (time.time() - start) * 1000
    logger.info(f"✅ [HOÀN THÀNH: RAG] Hoàn tất toàn bộ chu trình trong {elapsed_ms:.1f}ms")
    logger.info("=" * 60)
    return RAGResponse(
        answer=answer,
        sources=sources,
        query_time_ms=elapsed_ms,
        intent=intent.value,
    )
