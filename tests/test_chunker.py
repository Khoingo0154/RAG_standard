import io
import uuid

from ingestion.services.chunker import chunk_pdf, _clean_text, _is_valid_chunk
from tests.conftest import _make_sample_pdf


class TestChunker:
    def test_chunk_by_page_basic(self):
        pdf_bytes = _make_sample_pdf(num_pages=3)
        file_id = str(uuid.uuid4())

        chunks = chunk_pdf(pdf_bytes, file_id, strategy="page")

        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert chunk.file_id == file_id
            assert chunk.chunk_index == i
            assert chunk.page_start == chunk.page_end
            assert chunk.page_start == i + 1
            assert len(chunk.text) > 0
            assert chunk.char_count == len(chunk.text)

    def test_chunk_by_token_sliding_window(self):
        pdf_bytes = _make_sample_pdf(num_pages=5)
        file_id = str(uuid.uuid4())

        chunks = chunk_pdf(pdf_bytes, file_id, strategy="token", chunk_size=300, chunk_overlap=50)

        assert len(chunks) > 0
        for i in range(len(chunks) - 1):
            assert chunks[i].chunk_index < chunks[i + 1].chunk_index

    def test_chunk_overlap_validation(self):
        pdf_bytes = _make_sample_pdf(num_pages=1)
        file_id = str(uuid.uuid4())

        try:
            chunk_pdf(pdf_bytes, file_id, strategy="token", chunk_size=100, chunk_overlap=200)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_clean_text_removes_header_footer(self):
        raw = "Trang 1\nSome content\n1/3"
        cleaned = _clean_text(raw)
        assert "Trang" not in cleaned
        assert "1/3" not in cleaned
        assert "Some content" in cleaned

    def test_clean_text_removes_page_header(self):
        raw = "Page 1\nActual text here"
        cleaned = _clean_text(raw)
        assert "Page" not in cleaned
        assert "Actual text" in cleaned

    def test_is_valid_chunk_below_min_length(self):
        assert not _is_valid_chunk("Hi")

    def test_is_valid_chunk_low_letter_ratio(self):
        assert not _is_valid_chunk("12345 67890 12345 67890 12345 67890 12345 67890 12345 67890 12345")

    def test_is_valid_chunk_valid(self):
        assert _is_valid_chunk("This is a valid chunk with enough text and mostly letters " * 3)

    def test_empty_pdf_returns_empty_chunks(self):
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.showPage()
        c.save()
        pdf_bytes = buf.getvalue()

        chunks = chunk_pdf(pdf_bytes, str(uuid.uuid4()), strategy="page")
        assert len(chunks) == 0
