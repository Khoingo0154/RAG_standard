import uuid
import pytest
from unittest.mock import MagicMock, patch

from ingestion.models import ChunkDoc
from ingestion.services.embedder import BaseEmbedder, GeminiEmbedder, get_embedder


class MockEmbedder(BaseEmbedder):
    DIMS = 4

    @property
    def model_name(self) -> str:
        return "mock-embedder"

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import random
        return [[random.random() for _ in range(self.DIMS)] for _ in texts]


class TestEmbedder:
    def test_mock_embedder_returns_correct_count(self):
        chunks = [
            ChunkDoc(
                chunk_id=str(uuid.uuid4()), file_id=str(uuid.uuid4()),
                text=f"Sample text {i}", page_start=i + 1, page_end=i + 1,
                chunk_index=i,
            )
            for i in range(5)
        ]
        embedder = MockEmbedder()
        embedded = embedder.embed_chunks(chunks, batch_size=2)

        assert len(embedded) == 5
        for ec in embedded:
            assert ec.chunk_id in [c.chunk_id for c in chunks]
            assert len(ec.vector) == MockEmbedder.DIMS
            assert ec.model_name == "mock-embedder"

    def test_embed_chunks_preserves_metadata(self):
        chunk = ChunkDoc(
            chunk_id=str(uuid.uuid4()), file_id=str(uuid.uuid4()),
            text="Test content", page_start=1, page_end=1, chunk_index=0,
        )
        embedder = MockEmbedder()
        embedded = embedder.embed_chunks([chunk])[0]

        assert embedded.text == "Test content"
        assert embedded.page_start == 1
        assert embedded.page_end == 1
        assert embedded.chunk_index == 0

    def test_embed_chunks_empty_list(self):
        embedder = MockEmbedder()
        result = embedder.embed_chunks([], batch_size=32)
        assert len(result) == 0

    def test_embed_chunks_rejects_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            MockEmbedder().embed_chunks([], batch_size=0)

    def test_embed_chunks_rejects_mismatched_vector_count(self):
        class BrokenEmbedder(MockEmbedder):
            def _embed_batch(self, texts: list[str]) -> list[list[float]]:
                return []

        chunk = ChunkDoc(
            chunk_id=str(uuid.uuid4()), file_id=str(uuid.uuid4()),
            text="Test content", page_start=1, page_end=1, chunk_index=0,
        )
        with pytest.raises(RuntimeError, match="vector"):
            BrokenEmbedder().embed_chunks([chunk])

    def test_get_embedder_gemini(self):
        with patch("google.genai.Client"):
            embedder = get_embedder("gemini")
        assert isinstance(embedder, GeminiEmbedder)
        assert embedder.model_name == "gemini-embedding-001"

    def test_gemini_embedder_uses_retrieval_config(self):
        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.embed_content.return_value.embeddings = [
                MagicMock(values=[0.1, 0.2])
            ]
            vectors = GeminiEmbedder(
                api_key="test-key", dimensions=2, task_type="RETRIEVAL_QUERY"
            )._embed_batch(["việt vị"])

        assert vectors == [[0.1, 0.2]]
        assert mock_client.return_value.models.embed_content.call_args[1]["config"].task_type == "RETRIEVAL_QUERY"

    def test_gemini_embedder_retries_quota_error(self):
        with patch("google.genai.Client") as mock_client:
            client = mock_client.return_value
            client.models.embed_content.side_effect = [
                RuntimeError("429 RESOURCE_EXHAUSTED: retry in 2s"),
                MagicMock(embeddings=[MagicMock(values=[0.5])]),
            ]
            with patch("ingestion.services.embedder.time.sleep") as mock_sleep:
                vectors = GeminiEmbedder(api_key="test-key", dimensions=1)._embed_batch(["test"])

        assert vectors == [[0.5]]
        mock_sleep.assert_called_once_with(3.0)

    def test_get_embedder_unknown_provider(self):
        try:
            get_embedder("unknown_provider_xyz")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
