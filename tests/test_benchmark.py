import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from evals.benchmark import BenchmarkCase, BenchmarkSummary, CaseResult, RAGBenchmark
from evals.metrics import (
    extract_cited_pages,
    hit_at_k,
    keyword_coverage,
    pages_in_range,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    verify_citation_faithfulness,
)


# === 1. TESTS CHO EVALS/METRICS.PY ===

def test_pages_in_range():
    assert pages_in_range(49, 51) == {49, 50, 51}
    assert pages_in_range(51, 51) == {51}
    assert pages_in_range(0, 0) == set()
    assert pages_in_range(52, 50) == {50, 51, 52}


def test_retrieval_metrics():
    hits = [
        {"page_start": 49, "page_end": 50, "score": 0.9},
        {"page_start": 88, "page_end": 89, "score": 0.8},
        {"page_start": 51, "page_end": 51, "score": 0.7},
    ]
    expected = [49, 50, 51]

    # Recall: trang 49, 50, 51 đều xuất hiện -> 3/3 = 1.0
    assert recall_at_k(hits, expected) == 1.0

    # Precision: 2/3 chunks đúng (hit 1 và hit 3) -> 2/3 ≈ 0.6667
    assert round(precision_at_k(hits, expected), 4) == round(2 / 3, 4)

    # Hit@K: Có xuất hiện -> 1.0
    assert hit_at_k(hits, expected) == 1.0

    # MRR: Hit đầu tiên (rank 1) trúng trang 49-50 -> 1/1 = 1.0
    assert reciprocal_rank(hits, expected) == 1.0

    # Trường hợp không trúng trang nào
    assert recall_at_k(hits, [99, 100]) == 0.0
    assert precision_at_k(hits, [99, 100]) == 0.0
    assert hit_at_k(hits, [99, 100]) == 0.0
    assert reciprocal_rank(hits, [99, 100]) == 0.0


def test_extract_cited_pages():
    text1 = "Việt vị là gì? (Tham khảo tại trang 49–51, 80–81 của PDF Law_fifa.pdf.)"
    assert extract_cited_pages(text1) == [49, 50, 51, 80, 81]

    text2 = "Không có lỗi việt vị (Tham khảo tại trang 51 của PDF)."
    assert extract_cited_pages(text2) == [51]

    text3 = "Câu trả lời không có trích dẫn nào cả."
    assert extract_cited_pages(text3) == []


def test_verify_citation_faithfulness():
    answer = "Câu trả lời đúng. (Tham khảo tại trang 49-51, 99 của PDF)."
    hits = [
        {"page_start": 49, "page_end": 51},
        {"page_start": 52, "page_end": 53},
    ]

    res = verify_citation_faithfulness(answer, hits)
    # Trang 49, 50, 51 là hợp lệ; trang 99 là bịa (hallucinated)
    assert 49 in res["valid_citations"]
    assert 50 in res["valid_citations"]
    assert 51 in res["valid_citations"]
    assert 99 in res["hallucinated_citations"]
    assert res["is_faithful"] is False


def test_keyword_coverage():
    fact = "Tối thiểu 7 cầu thủ trên sân, nếu ít hơn trận đấu sẽ bị dừng"
    answer_good = "Số lượng cầu thủ tối thiểu là 7 cầu thủ, nếu ít hơn trận đấu sẽ dừng lại"
    answer_bad = "Thời tiết hôm nay rất đẹp và có nắng ấm"

    assert keyword_coverage(answer_good, fact) > 0.5
    assert keyword_coverage(answer_bad, fact) == 0.0


# === 2. TESTS CHO EVALS/BENCHMARK.PY ===

def test_benchmark_case_and_summary_exports(tmp_path):
    case = BenchmarkCase.from_dict({
        "id": "c1",
        "query": "Việt vị là gì?",
        "expected_pages": [49, 50, 51],
        "key_fact": "Vị trí việt vị và thời điểm chuyền",
        "category": "Luật 11",
    })

    assert case.id == "c1"
    assert case.expected_pages == [49, 50, 51]

    res = CaseResult(
        case_id=case.id,
        query=case.query,
        intent="DOCUMENT_QUERY",
        latency_ms=150.0,
        recall_at_k=1.0,
        precision_at_k=0.8,
        mrr=1.0,
        hit_at_k=1.0,
        top_score=0.92,
        retrieved_pages=[49, 50, 51],
        answer="Việt vị là...",
        key_fact=case.key_fact,
        fact_coverage=0.9,
    )

    summary = BenchmarkSummary(
        dataset_name="Test Dataset",
        file_id="f1",
        mode="retrieval",
        top_k=5,
        use_hybrid=True,
        use_rerank=True,
        timestamp="2026-09-05 12:00:00",
        total_cases=1,
        successful_cases=1,
        failed_cases=0,
        mean_recall_at_k=1.0,
        mean_precision_at_k=0.8,
        mean_mrr=1.0,
        mean_hit_rate=1.0,
        mean_latency_ms=150.0,
        p95_latency_ms=150.0,
        mean_fact_coverage=0.9,
        citation_faithfulness_rate=1.0,
        cases=[res],
    )

    # Test export JSON
    json_path = tmp_path / "summary.json"
    summary.save_json(json_path)
    assert json_path.exists()
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["mean_recall_at_k"] == 1.0

    # Test export CSV
    csv_path = tmp_path / "summary.csv"
    summary.save_csv(csv_path)
    assert csv_path.exists()

    # Test export Markdown
    md_path = tmp_path / "summary.md"
    summary.save_markdown(md_path)
    assert md_path.exists()
    assert "BÁO CÁO KẾT QUẢ BENCHMARK" in md_path.read_text(encoding="utf-8")


def test_rag_benchmark_run_mocked(tmp_path):
    # Tạo file test dataset mẫu
    ds_file = tmp_path / "sample_dataset.json"
    ds_data = {
        "name": "Sample Test",
        "cases": [
            {
                "id": "test_1",
                "query": "Việt vị là gì?",
                "expected_pages": [49, 50],
                "key_fact": "offside offence",
            }
        ],
    }
    ds_file.write_text(json.dumps(ds_data), encoding="utf-8")

    bench = RAGBenchmark(file_id="mock_file", top_k=5)

    with patch("retrieval.router.QueryRouter.classify") as mock_classify, \
         patch("ingestion.services.embedder.get_embedder") as mock_embedder, \
         patch("retrieval.pipeline.retrieve_hybrid_and_rerank") as mock_retrieve:

        mock_classify.return_value = MagicMock(value="DOCUMENT_QUERY")
        mock_emb_inst = MagicMock()
        mock_emb_inst.embed_query.return_value = [0.1, 0.2]
        mock_embedder.return_value = mock_emb_inst

        mock_retrieve.return_value = [
            {"chunk_id": "c1", "page_start": 49, "page_end": 50, "score": 0.95, "text": "Law 11 text"}
        ]

        summary = bench.run(dataset_path=ds_file, mode="retrieval", output_dir=tmp_path)

        assert summary.total_cases == 1
        assert summary.successful_cases == 1
        assert summary.mean_recall_at_k == 1.0
        assert summary.mean_precision_at_k == 1.0
        assert summary.mean_hit_rate == 1.0
        assert summary.mean_mrr == 1.0
