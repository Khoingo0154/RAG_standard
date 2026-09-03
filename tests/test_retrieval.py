import uuid
from unittest.mock import MagicMock, patch

from retrieval.models import QueryRequest, RetrievedChunk, RAGResponse


class TestRetrievalModels:
    def test_query_request_defaults(self):
        q = QueryRequest(query="What is AI?")
        assert q.query == "What is AI?"
        assert q.file_id is None
        assert q.top_k == 5

    def test_query_request_with_file_id(self):
        q = QueryRequest(query="test", file_id="file-123", top_k=10)
        assert q.file_id == "file-123"
        assert q.top_k == 10

    def test_retrieved_chunk(self):
        rc = RetrievedChunk(
            chunk_id="c1", file_id="f1", text="hello",
            page_start=1, page_end=2, chunk_index=0,
            score=0.95, filename="doc.pdf",
        )
        assert rc.score == 0.95
        assert rc.filename == "doc.pdf"

    def test_rag_response_default_sources(self):
        resp = RAGResponse(answer="Hello", query_time_ms=123.0)
        assert resp.answer == "Hello"
        assert resp.sources == []
        assert resp.query_time_ms == 123.0


class TestRetriever:
    def test_search_returns_hits(self):
        with patch("chromadb.PersistentClient") as mock_client:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["c1", "c2"]],
                "documents": [["text1", "text2"]],
                "metadatas": [[
                    {"file_id": "f1", "page_start": 1, "page_end": 1, "chunk_index": 0},
                    {"file_id": "f1", "page_start": 2, "page_end": 2, "chunk_index": 1},
                ]],
                "distances": [[0.1, 0.3]],
            }
            mock_client.return_value.get_or_create_collection.return_value = mock_collection

            from retrieval.retriever import Retriever
            retriever = Retriever()
            retriever._client = mock_client.return_value
            retriever._collection = mock_collection

            hits = retriever.search([0.1, 0.2, 0.3, 0.4])
            assert len(hits) == 2
            assert hits[0]["chunk_id"] == "c1"
            assert hits[0]["score"] == 0.9

    def test_search_with_file_filter(self):
        with patch("chromadb.PersistentClient") as mock_client:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            mock_client.return_value.get_or_create_collection.return_value = mock_collection

            from retrieval.retriever import Retriever
            retriever = Retriever()
            retriever._client = mock_client.return_value
            retriever._collection = mock_collection

            hits = retriever.search([0.1, 0.2, 0.3, 0.4], file_id="f1")
            mock_collection.query.assert_called_once()
            assert mock_collection.query.call_args[1]["where"] == {"file_id": "f1"}


class TestGenerator:
    def test_build_prompt_includes_context(self):
        from retrieval.generator import Generator
        gen = Generator(provider="ollama")
        hits = [
            {"text": "Context passage 1 about AI"},
            {"text": "Context passage 2 about ML"},
        ]
        prompt = gen._build_prompt("What is AI?", hits)
        assert "Context passage 1" in prompt
        assert "Context passage 2" in prompt
        assert "What is AI?" in prompt

    def test_generate_no_hits_returns_default(self):
        from retrieval.generator import Generator
        gen = Generator(provider="ollama")
        answer, sources = gen.generate("test", [])
        assert "Không tìm thấy" in answer
        assert sources == []

    def test_generate_with_ollama(self):
        with patch("retrieval.generator.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": "AI là trí tuệ nhân tạo."}
            mock_post.return_value = mock_resp

            with patch("retrieval.generator.Generator._lookup_metadata", return_value={"filename": "doc.pdf"}):
                from retrieval.generator import Generator
                gen = Generator(provider="ollama", mongo_uri="mongodb://dummy")
                answer, sources = gen.generate("What is AI?", [
                    {"chunk_id": "c1", "file_id": "f1", "text": "AI stands for...", "page_start": 1, "page_end": 1, "chunk_index": 0, "score": 0.9}
                ])

            assert "AI" in answer or "trí tuệ" in answer
            assert len(sources) == 1
            assert sources[0].filename == "doc.pdf"

    def test_generate_with_gemini(self):
        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value.text = "AI là trí tuệ nhân tạo."
            with patch("retrieval.generator.Generator._lookup_metadata", return_value={"filename": "doc.pdf"}):
                from retrieval.generator import Generator
                gen = Generator(provider="gemini", gemini_api_key="test-key", mongo_uri="mongodb://dummy")
                answer, sources = gen.generate("What is AI?", [
                    {"chunk_id": "c1", "file_id": "f1", "text": "AI stands for...", "page_start": 1, "page_end": 1, "chunk_index": 0, "score": 0.9}
                ])

            assert answer.startswith("AI là trí tuệ nhân tạo.")
            assert "trang 1 của PDF doc.pdf" in answer
            assert len(sources) == 1
            mock_client.return_value.models.generate_content.assert_called_once()

    def test_generate_with_gemini_requires_api_key(self):
        from retrieval.generator import Generator
        gen = Generator(provider="gemini", gemini_api_key="")
        answer, _ = gen.generate("What is AI?", [
            {"chunk_id": "c1", "file_id": "f1", "text": "AI stands for...", "page_start": 1, "page_end": 1, "chunk_index": 0, "score": 0.9}
        ])
        assert "GEMINI_API_KEY" in answer

    def test_generate_appends_pdf_page_references(self):
        with patch("retrieval.generator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "Câu trả lời."}
            with patch("retrieval.generator.Generator._lookup_metadata", return_value={"filename": "Law_fifa.pdf"}):
                from retrieval.generator import Generator
                answer, _ = Generator(provider="ollama").generate("Câu hỏi", [
                    {"chunk_id": "c1", "file_id": "f1", "text": "nội dung 1", "page_start": 10, "page_end": 10, "chunk_index": 0, "score": 0.9},
                    {"chunk_id": "c2", "file_id": "f1", "text": "nội dung 2", "page_start": 12, "page_end": 13, "chunk_index": 1, "score": 0.8},
                ])

        assert answer.endswith("(Tham khảo tại trang 10, 12–13 của PDF Law_fifa.pdf.)")


class TestRetrievalPipeline:
    def test_expand_query_for_fifa_vietnamese_term(self):
        from retrieval.pipeline import expand_query_for_retrieval

        expanded = expand_query_for_retrieval("Việt vị là gì?")
        assert "offside" in expanded
        assert "Law 11" in expanded

    def test_expand_query_leaves_unrelated_query_unchanged(self):
        from retrieval.pipeline import expand_query_for_retrieval

        assert expand_query_for_retrieval("Thời tiết hôm nay thế nào?") == "Thời tiết hôm nay thế nào?"
        assert "corner kick" in expand_query_for_retrieval("Quả phạt góc là gì?")

    def test_search_and_generate(self):
        with patch("retrieval.pipeline.get_embedder") as mock_get_embedder:
            mock_embedder = MagicMock()
            mock_embedder._embed_batch.return_value = [[0.1, 0.2, 0.3]]
            mock_get_embedder.return_value = mock_embedder

            with patch("retrieval.retriever.Retriever.search") as mock_search:
                mock_search.return_value = [
                    {"chunk_id": "c1", "file_id": "f1", "text": "AI context", "page_start": 1, "page_end": 1, "chunk_index": 0, "score": 0.95}
                ]

                with patch("retrieval.generator.Generator.generate") as mock_gen:
                    mock_gen.return_value = ("AI là trí tuệ nhân tạo", [
                        RetrievedChunk(chunk_id="c1", file_id="f1", text="AI context", page_start=1, page_end=1, chunk_index=0, score=0.95, filename="doc.pdf")
                    ])

                    from retrieval.pipeline import search_and_generate
                    from retrieval.router import QueryIntent
                    mock_router = MagicMock()
                    mock_router.classify.return_value = QueryIntent.DOCUMENT_QUERY

                    result = search_and_generate("What is AI?", router=mock_router)

                    assert "AI" in result.answer
                    assert len(result.sources) == 1
                    assert result.query_time_ms > 0
    def test_search_and_generate_with_file_filter(self):
        with patch("retrieval.pipeline.get_embedder") as mock_embedder:
            mock_embedder.return_value._embed_batch.return_value = [[0.1, 0.2, 0.3]]
            with patch("retrieval.retriever.Retriever.search") as mock_search:
                mock_search.return_value = [
                    {"chunk_id": "c1", "file_id": "f1", "text": "text", "page_start": 1, "page_end": 1, "chunk_index": 0, "score": 0.9}
                ]
                with patch("retrieval.generator.Generator.generate") as mock_gen:
                    mock_gen.return_value = ("Answer", [
                        RetrievedChunk(chunk_id="c1", file_id="f1", text="text", page_start=1, page_end=1, chunk_index=0, score=0.9, filename="doc.pdf")
                    ])

                    from retrieval.pipeline import search_and_generate
                    from retrieval.router import QueryIntent
                    mock_router = MagicMock()
                    mock_router.classify.return_value = QueryIntent.DOCUMENT_QUERY

                    result = search_and_generate("query", file_id="f1", router=mock_router)

                    mock_search.assert_called_once()
                    assert mock_search.call_args[1]["file_id"] == "f1"
