"""Mô hình Cross-Encoder Reranking sử dụng FlashRank (tối ưu ONNX trên CPU)."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_GLOBAL_RANKER = None


def get_ranker(model_name: str = "ms-marco-MiniLM-L-12-v2"):
    global _GLOBAL_RANKER
    if _GLOBAL_RANKER is None:
        try:
            from flashrank import Ranker
            _GLOBAL_RANKER = Ranker(model_name=model_name)
            logger.info(f"FlashRank Ranker đã khởi tạo thành công với model '{model_name}'.")
        except Exception as e:
            logger.error(f"Không thể khởi tạo FlashRank Ranker: {e}")
            _GLOBAL_RANKER = None
    return _GLOBAL_RANKER


class Reranker:
    """Chấm điểm chéo (Cross-Encoder) giữa câu hỏi và các đoạn văn bản ứng viên."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self._model_name = model_name
        self._ranker = get_ranker(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Chấm điểm lại danh sách candidates và trả về top_k chunks có điểm tương quan cao nhất."""
        if not candidates or not query:
            return []

        if len(candidates) <= 1 or self._ranker is None:
            return candidates[:top_k]

        try:
            from flashrank import RerankRequest

            # Map từng candidate thành passage
            passages = [
                {"id": idx, "text": c.get("text", "")}
                for idx, c in enumerate(candidates)
            ]

            req = RerankRequest(query=query, passages=passages)
            ranked_results = self._ranker.rerank(req)

            # Chuẩn hóa điểm ban đầu (RRF / Vector score) về [0, 1]
            initial_scores = [float(c.get("score", 0.0)) for c in candidates]
            max_s = max(initial_scores) if initial_scores else 1.0
            min_s = min(initial_scores) if initial_scores else 0.0
            range_s = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0

            rerank_dict = {r["id"]: float(r["score"]) for r in ranked_results}

            reranked_chunks = []
            for idx, candidate in enumerate(candidates):
                norm_initial = (float(candidate.get("score", 0.0)) - min_s) / range_s
                rerank_score = rerank_dict.get(idx, 0.0)

                # Kết hợp thông minh: 60% Dense Semantic / RRF + 40% Cross-Encoder
                combined_score = round(0.6 * norm_initial + 0.4 * rerank_score, 4)

                chunk_copy = dict(candidate)
                chunk_copy["score"] = combined_score
                chunk_copy["rerank_score"] = round(rerank_score, 4)
                reranked_chunks.append(chunk_copy)

            reranked_chunks.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"Reranker: Đã chấm điểm lại {len(candidates)} chunks -> Lấy Top-{top_k}")
            return reranked_chunks[:top_k]

        except Exception as e:
            logger.warning(f"Lỗi khi chạy Reranker ({e}). Giữ nguyên thứ tự ban đầu.")
            return candidates[:top_k]
