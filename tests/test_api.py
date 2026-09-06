import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app
from tests.conftest import _make_sample_pdf

client = TestClient(app)


class TestAPI:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ingest_non_pdf_rejected(self):
        response = client.post("/ingest", files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")})
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_ingest_empty_file_rejected(self):
        response = client.post("/ingest", files={"file": ("test.pdf", io.BytesIO(b""), "application/pdf")})
        assert response.status_code == 400
        assert "rỗng" in response.json()["detail"]

    def test_ingest_pdf_success(self):
        pdf_bytes = _make_sample_pdf(num_pages=2)
        mock_store = MagicMock()
        from ingestion.models import StoreResult
        mock_store.save.return_value = StoreResult(
            file_id="test-id", total_chunks=2,
            chroma_ids=["id1", "id2"], mongo_ids=["m1", "m2"],
        )

        with patch("ingestion.pipeline.IngestionStore", return_value=mock_store):
            with patch("ingestion.pipeline.VectorStore"):
                with patch("ingestion.pipeline.MetadataStore"):
                    with patch("ingestion.pipeline.MinioObjectStore"):
                        with patch("ingestion.pipeline.get_embedder") as mock_embed:
                            mock_embedder = MagicMock()
                            mock_embedder.model_name = "mock"
                            mock_embedder.embed_chunks.return_value = []
                            mock_embed.return_value = mock_embedder

                            response = client.post(
                                "/ingest",
                                files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
                            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "file_id" in data
        assert data["total_chunks"] > 0

    def test_query_endpoint(self):
        with patch("api.main.search_and_generate") as mock_sg:
            from retrieval.models import RAGResponse, RetrievedChunk
            mock_sg.return_value = RAGResponse(
                answer="AI là trí tuệ nhân tạo",
                sources=[
                    RetrievedChunk(
                        chunk_id="c1", file_id="f1", text="AI context",
                        page_start=1, page_end=1, chunk_index=0,
                        score=0.95, filename="doc.pdf",
                    )
                ],
                query_time_ms=123.0,
            )

            response = client.post("/query", json={"query": "What is AI?"})

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["sources"]) == 1
        assert data["query_time_ms"] > 0

    def test_query_endpoint_with_file_filter(self):
        with patch("api.main.search_and_generate") as mock_sg:
            from retrieval.models import RAGResponse
            mock_sg.return_value = RAGResponse(answer="Answer", sources=[], query_time_ms=50.0)

            response = client.post("/query", json={
                "query": "test", "file_id": "f1", "top_k": 3,
            })

        assert response.status_code == 200
        mock_sg.assert_called_once()
        assert mock_sg.call_args[1]["file_id"] == "f1"
        assert mock_sg.call_args[1]["top_k"] == 3

    def test_list_files_endpoint(self):
        with patch("api.main.MinioObjectStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store.list_files.return_value = [
                {
                    "file_id": "file-123",
                    "filename": "Law_fifa.pdf",
                    "object_key": "documents/file-123/Law_fifa.pdf",
                    "size_bytes": 1024,
                    "last_modified": "2026-09-05T00:00:00",
                }
            ]
            mock_store_cls.return_value = mock_store

            response = client.get("/files")

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 1
        assert data["file_ids"] == ["file-123"]
        assert data["files"][0]["filename"] == "Law_fifa.pdf"
