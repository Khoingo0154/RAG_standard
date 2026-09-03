import uuid
from unittest.mock import MagicMock

from ingestion.models import EmbeddedChunk, StoreResult
from ingestion.storage.store import IngestionStore


class TestIngestionStore:
    def _make_chunks(self, count: int = 3) -> list[EmbeddedChunk]:
        file_id = str(uuid.uuid4())
        return [
            EmbeddedChunk(
                chunk_id=str(uuid.uuid4()), file_id=file_id,
                text=f"Test chunk {i}", page_start=i + 1, page_end=i + 1,
                chunk_index=i, vector=[0.1, 0.2, 0.3, 0.4], model_name="test-model",
            )
            for i in range(count)
        ]

    def test_save_success(self):
        chunks = self._make_chunks(3)
        mock_vector = MagicMock()
        mock_vector.upsert.return_value = [c.chunk_id for c in chunks]
        mock_meta = MagicMock()
        mock_meta.insert_chunks.return_value = ["id1", "id2", "id3"]

        store = IngestionStore(vector_store=mock_vector, metadata_store=mock_meta)
        result = store.save(chunks)

        assert result.file_id == chunks[0].file_id
        assert result.total_chunks == 3
        assert len(result.chroma_ids) == 3
        assert len(result.mongo_ids) == 3
        mock_vector.upsert.assert_called_once()
        mock_meta.insert_chunks.assert_called_once()

    def test_save_rollback_on_mongo_failure(self):
        chunks = self._make_chunks(2)
        mock_vector = MagicMock()
        mock_vector.upsert.return_value = [c.chunk_id for c in chunks]
        mock_meta = MagicMock()
        mock_meta.insert_chunks.side_effect = RuntimeError("Mongo down")

        store = IngestionStore(vector_store=mock_vector, metadata_store=mock_meta)

        try:
            store.save(chunks)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Chroma rolled back" in str(e)

        mock_vector.delete_by_file.assert_called_once_with(chunks[0].file_id)

    def test_save_empty_chunks_raises(self):
        store = IngestionStore(vector_store=MagicMock(), metadata_store=MagicMock())
        try:
            store.save([])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_save_chroma_failure_raises(self):
        chunks = self._make_chunks(2)
        mock_vector = MagicMock()
        mock_vector.upsert.side_effect = RuntimeError("Chroma down")
        mock_meta = MagicMock()

        store = IngestionStore(vector_store=mock_vector, metadata_store=mock_meta)

        try:
            store.save(chunks)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Chroma upsert failed" in str(e)
