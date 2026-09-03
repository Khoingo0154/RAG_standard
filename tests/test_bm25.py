from retrieval.bm25_retriever import BM25Retriever, tokenize


def test_tokenize_cleans_and_splits_text():
    tokens = tokenize("Luật 11: Việt vị trong bóng đá! 11m")
    assert "luật" in tokens
    assert "11" in tokens
    assert "việt" in tokens
    assert "vị" in tokens
    assert "11m" in tokens


def test_bm25_retriever_build_and_search():
    retriever = BM25Retriever()
    sample_chunks = [
        {
            "chunk_id": "c1",
            "file_id": "f1",
            "text": "Luật 11 quy định chi tiết về lỗi việt vị trong trận đấu bóng đá.",
            "page_start": 49,
            "page_end": 50,
            "chunk_index": 0,
        },
        {
            "chunk_id": "c2",
            "file_id": "f1",
            "text": "Luật 14 quy định về quả phạt đền penalty từ cự ly 11m.",
            "page_start": 61,
            "page_end": 62,
            "chunk_index": 1,
        },
        {
            "chunk_id": "c3",
            "file_id": "f1",
            "text": "Luật 15 quy định về quả ném biên khi bóng đi hết đường biên dọc.",
            "page_start": 64,
            "page_end": 65,
            "chunk_index": 2,
        },
    ]

    retriever.build_index(chunks=sample_chunks, file_id="f1")

    # Tìm kiếm việt vị
    hits = retriever.search("việt vị là gì", top_k=2, file_id="f1")
    assert len(hits) >= 1
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["score"] > 0

    # Tìm kiếm penalty 11m
    hits_penalty = retriever.search("phạt đền 11m", top_k=2, file_id="f1")
    assert len(hits_penalty) >= 1
    assert hits_penalty[0]["chunk_id"] == "c2"
