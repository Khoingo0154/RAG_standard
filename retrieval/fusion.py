"""Hợp nhất kết quả tìm kiếm (Reciprocal Rank Fusion - RRF)."""

from typing import Optional


def reciprocal_rank_fusion(
    vector_hits: list[dict],
    bm25_hits: list[dict],
    k: int = 60,
    top_n: Optional[int] = 25,
) -> list[dict]:
    """Hợp nhất 2 danh sách kết quả Dense (Vector) và Sparse (BM25) bằng thuật toán RRF.

    RRF_score(d) = sum(1 / (k + rank_i(d)))
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for hit_list in [vector_hits, bm25_hits]:
        for rank, hit in enumerate(hit_list, 1):
            cid = hit.get("chunk_id")
            if not cid:
                continue

            if cid not in docs:
                docs[cid] = dict(hit)
                scores[cid] = 0.0

            scores[cid] += 1.0 / (k + rank)

    merged = []
    for cid, doc in docs.items():
        doc_copy = dict(doc)
        doc_copy["score"] = round(scores[cid], 6)
        merged.append(doc_copy)

    merged.sort(key=lambda x: x["score"], reverse=True)

    if top_n is not None:
        return merged[:top_n]
    return merged
