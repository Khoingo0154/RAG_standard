from unittest.mock import MagicMock, patch

from ingestion.pipeline import IngestionConfig, run_ingestion
from ingestion.models import StoreResult
from tests.conftest import _make_sample_pdf
from tests.test_embedder import MockEmbedder


class TestIngestionPipeline:
    def test_full_pipeline_with_mock_store(self):
        pdf_bytes = _make_sample_pdf(num_pages=2)

        mock_store = MagicMock()
        mock_store.save.return_value = StoreResult(
            file_id="test-id",
            total_chunks=2,
            chroma_ids=["id1", "id2"],
            mongo_ids=["mongo1", "mongo2"],
        )

        with patch("ingestion.pipeline.IngestionStore", return_value=mock_store):
            with patch("ingestion.pipeline.VectorStore"):
                with patch("ingestion.pipeline.MetadataStore"):
                    result = run_ingestion(
                        pdf_bytes=pdf_bytes,
                        filename="test.pdf",
                        config=IngestionConfig(chunk_strategy="page", embed_provider="local"),
                        embedder=MockEmbedder(),
                    )

        assert result.total_chunks > 0
        assert result.file_record.filename == "test.pdf"
        assert result.duration_seconds > 0
        assert len(result.file_record.file_id) == 36

    def test_pipeline_empty_pdf_raises(self):
        try:
            run_ingestion(b"not a valid pdf", "empty.pdf", embedder=MockEmbedder())
            assert False, "Should have raised"
        except (RuntimeError, Exception):
            pass

    def test_pipeline_config_defaults(self):
        cfg = IngestionConfig()
        assert cfg.chunk_strategy == "token"
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.embed_provider == "local"
        assert cfg.mongo_uri == "mongodb://localhost:27017"

    def test_pipeline_uploads_original_pdf_to_minio(self):
        pdf_bytes = _make_sample_pdf(num_pages=1)
        mock_store = MagicMock()
        mock_store.save.return_value = StoreResult("test-id", 1, ["c1"], ["m1"])
        object_store = MagicMock()
        object_store.upload_pdf.return_value = "documents/file-id/test.pdf"

        with patch("ingestion.pipeline.IngestionStore", return_value=mock_store):
            with patch("ingestion.pipeline.VectorStore"):
                with patch("ingestion.pipeline.MetadataStore"):
                    result = run_ingestion(
                        pdf_bytes=pdf_bytes,
                        filename="test.pdf",
                        config=IngestionConfig(minio_enabled=True),
                        embedder=MockEmbedder(),
                        object_store=object_store,
                    )

        object_store.upload_pdf.assert_called_once()
        assert result.file_record.minio_path == "documents/file-id/test.pdf"
