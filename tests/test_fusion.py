from retrieval.fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_merges_and_deduplicates():
    vector_hits = [
        {"chunk_id": "c1", "text": "doc1", "score": 0.9},
        {"chunk_id": "c2", "text": "doc2", "score": 0.8},
        {"chunk_id": "c3", "text": "doc3", "score": 0.7},
    ]

    bm25_hits = [
        {"chunk_id": "c2", "text": "doc2", "score": 10.5},
        {"chunk_id": "c4", "text": "doc4", "score": 8.0},
        {"chunk_id": "c1", "text": "doc1", "score": 5.0},
    ]

    fused = reciprocal_rank_fusion(vector_hits, bm25_hits, k=60, top_n=10)

    # Tổng cộng có 4 chunk độc nhất (c1, c2, c3, c4)
    assert len(fused) == 4

    # c1 xuất hiện ở rank 1 (vector) và rank 3 (bm25): 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ≈ 0.01639 + 0.01587 ≈ 0.03226
    # c2 xuất hiện ở rank 2 (vector) và rank 1 (bm25): 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.01613 + 0.01639 ≈ 0.03252
    # c2 có điểm cao nhất
    assert fused[0]["chunk_id"] == "c2"
    assert fused[1]["chunk_id"] == "c1"
    assert fused[0]["score"] > fused[1]["score"]
