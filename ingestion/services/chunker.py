import re
import io
import uuid
from typing import Literal
from pypdf import PdfReader
from ingestion.models import ChunkDoc

ChunkStrategy = Literal["page", "token"]


def _clean_text(raw: str) -> str:
    """
    Làm sạch text raw từ PDF trước khi chunk.
    1. Strip khoảng trắng đầu/cuối
    2. Gom nhiều dòng trống (3+) thành 2 dòng
    3. Xoá header/footer dạng "1/100" (số trang / tổng số)
    4. Xoá dòng "Trang 1" hoặc "Page 1" ở đầu mỗi trang
    5. Chuẩn hóa tất cả khoảng trắng (space, tab, newline) về 1 space
    """
    text = raw.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n\d+\s*/\s*\d+", "", text)
    text = re.sub(r"(?i)^(trang|page)\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = " ".join(text.split())
    return text


def _is_valid_chunk(text: str) -> bool:
    """
    Kiểm tra chunk có đủ "chất lượng" để giữ lại không.
    - Quá ngắn (< 50 ký tự) → rác do PDF lỗi, không có ý nghĩa
    - Tỷ lệ chữ cái < 30% → toàn số/ký hiệu/bảng bị lỗi
    """
    if len(text) < 50:
        return False
    letter_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return letter_ratio > 0.3


def chunk_pdf(
    pdf_bytes: bytes,
    file_id: str,
    filename: str = "",
    strategy: ChunkStrategy = "token",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[ChunkDoc]:
    """
    Hàm CHÍNH của chunker.
    Đầu vào: PDF dạng byte, file_id, chiến lược cắt, kích thước chunk, overlap.
    Đầu ra: list[ChunkDoc] - danh sách các đoạn nhỏ.

    Cách hoạt động:
    1. Đọc PDF bằng thư viện pypdf
    2. Trích text từng trang
    3. Cắt nhỏ theo 1 trong 2 chiến lược: "page" hoặc "token"
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = _extract_pages(reader)
    if strategy == "page":
        return _chunk_by_page(pages, file_id, filename)
    else:
        return _chunk_by_token(pages, file_id, filename, chunk_size, chunk_overlap)


def _extract_pages(reader: PdfReader) -> list[dict]:
    """
    Đọc từng trang PDF, lấy text, làm sạch.
    Bỏ qua các trang trắng (không có text).
    Kết quả: list of dict, mỗi dict chứa page_num (số trang) và text đã clean.
    """
    pages = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = _clean_text(raw)
        if text:
            pages.append({"page_num": i + 1, "text": text})
    return pages


def _chunk_by_page(pages: list[dict], file_id: str, filename: str = "") -> list[ChunkDoc]:
    """
    Chiến lược PAGE: mỗi trang PDF là 1 chunk.
    Đơn giản nhất, giữ nguyên cấu trúc trang.
    Lọc bỏ các trang rác qua _is_valid_chunk().
    Nhược điểm: nếu 1 trang quá dài (>2000 chữ) thì không tốt cho embedding.
    """
    chunks = []
    chunk_index = 0
    for p in pages:
        if not _is_valid_chunk(p["text"]):
            continue
        chunks.append(ChunkDoc(
            chunk_id=str(uuid.uuid4()),
            file_id=file_id,
            text=p["text"],
            page_start=p["page_num"],
            page_end=p["page_num"],
            chunk_index=chunk_index,
            filename=filename,
        ))
        chunk_index += 1
    return chunks


def _chunk_by_token(
    pages: list[dict],
    file_id: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkDoc]:
    """
    Chiến lược TOKEN (sliding window):
    - Gom text tất cả trang thành 1 chuỗi dài
    - Dùng cửa sổ trượt cắt thành từng chunk đều nhau
    - Có overlap để không mất context ở chỗ nối
    - Lọc bỏ các chunk rác qua _is_valid_chunk()

    Ví dụ: chunk_size=1000, overlap=200
    Window 1: [0 ────────── 1000]
    Window 2:      [800 ────────── 1800]
    Window 3:           [1600 ────────── 2500]
    """
    # Kiểm tra: overlap phải nhỏ hơn chunk_size
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap phải nhỏ hơn chunk_size")

    # --- Bước 1: Gom text + đánh dấu ranh giới trang ---
    # boundaries = list of (vị_trí_bắt_đầu, vị_trí_kết_thúc, số_trang)
    # Giúp sau này biết một chunk nằm trong những trang nào
    full_text = ""
    boundaries: list[tuple[int, int, int]] = []
    for page in pages:
        start = len(full_text)
        full_text += page["text"] + "\n"
        boundaries.append((start, len(full_text), page["page_num"]))

    # --- Bước 2: Sliding window cắt ---
    chunks = []
    step = chunk_size - chunk_overlap  # mỗi lần tiến khoảng 800 (nếu size=1000, overlap=200)
    pos = 0
    chunk_index = 0
    while pos < len(full_text):
        end_pos = min(pos + chunk_size, len(full_text))
        text = full_text[pos:end_pos].strip()
        if text and _is_valid_chunk(text):
            page_start, page_end = _find_page_range(pos, end_pos, boundaries)
            chunks.append(ChunkDoc(
                chunk_id=str(uuid.uuid4()),
                file_id=file_id,
                text=text,
                page_start=page_start,
                page_end=page_end,
                chunk_index=chunk_index,
                filename=filename,
            ))
            chunk_index += 1
        pos += step
    return chunks


def _find_page_range(
    char_start: int,
    char_end: int,
    boundaries: list[tuple[int, int, int]],
) -> tuple[int, int]:
    """
    Xác định đoạn text từ char_start đến char_end nằm trong những trang nào.
    Duyệt qua boundaries để tìm trang_start và trang_end tương ứng.

    Ví dụ:
    - boundaries = [(0, 7, 1), (7, 14, 2)] (2 trang, mỗi trang 7 ký tự)
    - char_start=5, char_end=12
    - Kết quả: page_start=1, page_end=2 (chunk này chạm cả 2 trang)
    """
    page_start = None
    page_end = None
    for b_start, b_end, page_num in boundaries:
        if b_end > char_start and b_start < char_end:
            if page_start is None:
                page_start = page_num
            page_end = page_num
    return (page_start or 1, page_end or 1)
