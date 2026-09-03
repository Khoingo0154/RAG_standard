"""Chạy đánh giá Recall@k và Precision@k cho retrieval.

Ví dụ:
    python -m evals.retrieval_eval --file-id 8f525964-a864-43ef-b87b-34f6482d44f1
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

from ingestion.services.embedder import get_embedder
from retrieval.pipeline import expand_query_for_retrieval
from retrieval.retriever import Retriever
from retrieval.router import QueryRouter
from shared.config import settings

DEFAULT_DATASET = Path(__file__).with_name("fifa_law11_basic.json")


def pages_in_range(page_start: int, page_end: int) -> set[int]:
    """Trả các trang mà một chunk bao phủ, ví dụ 49–51 → {49, 50, 51}."""
    return set(range(page_start, page_end + 1))


def score_retrieval(hits: Iterable[dict], expected_pages: Iterable[int]) -> dict:
    """Tính page-level Recall@k và Precision@k cho một query có gold pages."""
    expected = set(expected_pages)
    hits = list(hits)
    retrieved_correct = 0
    covered_pages: set[int] = set()

    for hit in hits:
        hit_pages = pages_in_range(hit["page_start"], hit["page_end"])
        if hit_pages & expected:
            retrieved_correct += 1
            covered_pages.update(hit_pages & expected)

    return {
        "recall_at_k": len(covered_pages) / len(expected) if expected else 0.0,
        "precision_at_k": retrieved_correct / len(hits) if hits else 0.0,
        "covered_pages": sorted(covered_pages),
        "retrieved_pages": [
            {"page_start": hit["page_start"], "page_end": hit["page_end"], "score": round(hit["score"], 4)}
            for hit in hits
        ],
    }


def retrieve(
    query: str,
    file_id: str,
    top_k: int,
    use_hybrid: bool = True,
    use_rerank: bool = True,
) -> list[dict]:
    """Chạy Advanced Retrieval (Dense Vector + BM25 + FlashRank Reranker)."""
    from retrieval.pipeline import get_search_terms, retrieve_hybrid_and_rerank
    embedder = get_embedder(settings.EMBED_PROVIDER, task_type="RETRIEVAL_QUERY")
    retrieval_query, sparse_query, rerank_query = get_search_terms(query)
    vector = embedder.embed_query(retrieval_query)
    return retrieve_hybrid_and_rerank(
        query=query,
        query_vector=vector,
        sparse_query=sparse_query,
        rerank_query=rerank_query,
        file_id=file_id,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
    )


def evaluate_dataset(
    file_id: str,
    dataset_path: Path,
    top_k: int,
    use_hybrid: bool = True,
    use_rerank: bool = True,
) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    results = []
    router = QueryRouter()

    for case in dataset["cases"]:
        intent = router.classify(case["query"]).value
        hits = retrieve(
            case["query"],
            file_id=file_id,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
        )
        metrics = score_retrieval(hits, case["expected_pages"])
        results.append({
            "id": case["id"],
            "query": case["query"],
            "intent": intent,
            **metrics,
        })

    return {
        "dataset": dataset["name"],
        "file_id": file_id,
        "top_k": top_k,
        "mean_recall_at_k": sum(item["recall_at_k"] for item in results) / len(results) if results else 0.0,
        "mean_precision_at_k": sum(item["precision_at_k"] for item in results) / len(results) if results else 0.0,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá RAG retrieval bằng gold pages")
    parser.add_argument("--file-id", required=True, help="file_id PDF đã ingest")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, help="Tùy chọn: lưu kết quả JSON vào file")
    parser.add_argument("--no-hybrid", action="store_true", help="Chỉ dùng Vector Search (tắt BM25)")
    parser.add_argument("--no-rerank", action="store_true", help="Tắt bước Reranking FlashRank")
    args = parser.parse_args()

    result = evaluate_dataset(
        file_id=args.file_id,
        dataset_path=args.dataset,
        top_k=args.top_k,
        use_hybrid=not args.no_hybrid,
        use_rerank=not args.no_rerank,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print("____________________________________________________")
    
    print(rendered)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
