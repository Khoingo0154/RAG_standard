import io

from ingestion.models import ChunkDoc, EmbeddedChunk, FileRecord, StoreResult


class TestModels:
    def test_file_record_creation(self):
        fr = FileRecord(
            file_id="abc-123", filename="test.pdf",
            minio_path="documents/abc-123/test.pdf", size_bytes=1024,
        )
        assert fr.file_id == "abc-123"
        assert fr.filename == "test.pdf"
        assert fr.size_bytes == 1024
        assert fr.uploaded_at is not None

    def test_chunk_doc_char_count_auto(self):
        chunk = ChunkDoc(
            chunk_id="c1", file_id="f1", text="Hello World",
            page_start=1, page_end=1, chunk_index=0,
        )
        assert chunk.char_count == 11

    def test_embedded_chunk_from_chunk(self):
        chunk = ChunkDoc(
            chunk_id="c1", file_id="f1", text="Test text",
            page_start=2, page_end=3, chunk_index=5, filename="test.pdf",
        )
        ec = EmbeddedChunk.from_chunk(chunk, vector=[0.1, 0.2], model_name="test-model")

        assert ec.chunk_id == chunk.chunk_id
        assert ec.file_id == chunk.file_id
        assert ec.text == chunk.text
        assert ec.page_start == chunk.page_start
        assert ec.page_end == chunk.page_end
        assert ec.chunk_index == chunk.chunk_index
        assert ec.vector == [0.1, 0.2]
        assert ec.model_name == "test-model"
        assert ec.filename == "test.pdf"

    def test_store_result_creation(self):
        sr = StoreResult(
            file_id="f1", total_chunks=10,
            chroma_ids=["c1", "c2"], mongo_ids=["m1", "m2"],
        )
        assert sr.total_chunks == 10
        assert len(sr.chroma_ids) == 2
        assert len(sr.mongo_ids) == 2
        assert sr.stored_at is not None
