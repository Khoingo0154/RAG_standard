from evals.retrieval_eval import pages_in_range, score_retrieval


def test_pages_in_range_covers_all_pages():
    assert pages_in_range(49, 51) == {49, 50, 51}


def test_retrieval_metrics_calculate_recall_and_precision():
    hits = [
        {"page_start": 49, "page_end": 50, "score": 0.9},
        {"page_start": 88, "page_end": 89, "score": 0.8},
        {"page_start": 51, "page_end": 51, "score": 0.7},
    ]

    result = score_retrieval(hits, expected_pages=[49, 50, 51])

    assert result["recall_at_k"] == 1.0
    assert result["precision_at_k"] == 2 / 3
    assert result["covered_pages"] == [49, 50, 51]
