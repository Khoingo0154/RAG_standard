"""Các hàm tính toán chỉ số đo lường Retrieval và Generation cho RAG Benchmark."""

import re
from typing import Iterable, Optional


def pages_in_range(page_start: int, page_end: int) -> set[int]:
    """Trả về tập hợp các trang mà một chunk bao phủ (ví dụ: 49-51 -> {49, 50, 51})."""
    if page_start <= 0 or page_end <= 0:
        return set()
    start = min(page_start, page_end)
    end = max(page_start, page_end)
    return set(range(start, end + 1))


def recall_at_k(hits: list[dict], expected_pages: Iterable[int]) -> float:
    """Tính Recall@K: Tỷ lệ trang gold standard xuất hiện trong Top-K chunks."""
    expected = set(expected_pages)
    if not expected:
        return 1.0

    covered: set[int] = set()
    for hit in hits:
        p_start = hit.get("page_start", 0)
        p_end = hit.get("page_end", 0)
        chunk_pages = pages_in_range(p_start, p_end)
        covered.update(chunk_pages & expected)

    return len(covered) / len(expected)


def precision_at_k(hits: list[dict], expected_pages: Iterable[int]) -> float:
    """Tính Precision@K: Tỷ lệ chunks trong Top-K thực sự chứa ít nhất 1 trang gold standard."""
    if not hits:
        return 0.0

    expected = set(expected_pages)
    if not expected:
        return 0.0

    correct_chunks = 0
    for hit in hits:
        p_start = hit.get("page_start", 0)
        p_end = hit.get("page_end", 0)
        chunk_pages = pages_in_range(p_start, p_end)
        if chunk_pages & expected:
            correct_chunks += 1

    return correct_chunks / len(hits)


def hit_at_k(hits: list[dict], expected_pages: Iterable[int]) -> float:
    """Hit@K: Trả về 1.0 nếu có ít nhất 1 trang gold standard xuất hiện trong Top-K, ngược lại 0.0."""
    expected = set(expected_pages)
    if not expected:
        return 1.0

    for hit in hits:
        p_start = hit.get("page_start", 0)
        p_end = hit.get("page_end", 0)
        if pages_in_range(p_start, p_end) & expected:
            return 1.0
    return 0.0


def reciprocal_rank(hits: list[dict], expected_pages: Iterable[int]) -> float:
    """Reciprocal Rank (MRR): Nghịch đảo của thứ hạng (1/rank) của chunk đầu tiên chứa trang gold standard."""
    expected = set(expected_pages)
    if not expected:
        return 1.0

    for rank, hit in enumerate(hits, 1):
        p_start = hit.get("page_start", 0)
        p_end = hit.get("page_end", 0)
        if pages_in_range(p_start, p_end) & expected:
            return 1.0 / rank
    return 0.0


def extract_cited_pages(answer: str) -> list[int]:
    """Trích xuất danh sách các số trang PDF được trích dẫn trong văn bản câu trả lời."""
    if not answer:
        return []

    # Tìm cụm: "(Tham khảo tại trang 51, 80–81...)"
    match = re.search(r"\(Tham khảo tại (.*?)\.?\)", answer)
    target_text = match.group(1) if match else answer

    pages: set[int] = set()
    # Bắt các mẫu: "trang 51", "trang 49-51", "trang 49–51", "51", "80–81"
    # Tách theo dấu phẩy hoặc chữ "và"
    tokens = re.findall(r"(\d+)(?:\s*[-–—]\s*(\d+))?", target_text)
    for start_str, end_str in tokens:
        try:
            start = int(start_str)
            end = int(end_str) if end_str else start
            if 1 <= start <= 500:
                pages.update(pages_in_range(start, end))
        except ValueError:
            continue

    return sorted(pages)


def verify_citation_faithfulness(answer: str, hits: list[dict]) -> dict:
    """Xác thực tính trung thực của trích dẫn: Trích dẫn có thực sự nằm trong Top-K chunks được nạp hay không."""
    cited = extract_cited_pages(answer)
    retrieved_pages: set[int] = set()

    for hit in hits:
        p_start = hit.get("page_start", 0)
        p_end = hit.get("page_end", 0)
        retrieved_pages.update(pages_in_range(p_start, p_end))

    valid = [p for p in cited if p in retrieved_pages]
    hallucinated = [p for p in cited if p not in retrieved_pages]

    return {
        "cited_pages": cited,
        "retrieved_pages": sorted(retrieved_pages),
        "valid_citations": valid,
        "hallucinated_citations": hallucinated,
        "is_faithful": len(hallucinated) == 0 if cited else True,
        "citation_count": len(cited),
    }


def keyword_coverage(answer: str, key_fact: str) -> float:
    """Đo lường tỷ lệ các từ khóa cốt lõi từ key_fact xuất hiện trong câu trả lời."""
    if not key_fact:
        return 1.0
    if not answer:
        return 0.0

    # Lọc lấy các từ khóa có độ dài >= 3 ký tự (bỏ stopwords ngắn)
    words = re.findall(r"\w+", key_fact.lower())
    stop_words = {"các", "của", "cho", "khi", "nào", "được", "trong", "hoặc", "trên", "dưới", "không", "thì", "với", "như", "này", "phải", "đến"}
    keywords = [w for w in words if len(w) >= 3 and w not in stop_words]

    if not keywords:
        return 1.0

    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw in answer_lower)
    return round(found / len(keywords), 4)
