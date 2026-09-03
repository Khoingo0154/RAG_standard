from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


def new_id() -> str:
    """
    Sinh mã UUID ngẫu nhiên.
    Dùng để gán file_id cho FileRecord và chunk_id cho ChunkDoc.
    """
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Trả về thời gian UTC có timezone để tránh timestamp mơ hồ."""
    return datetime.now(timezone.utc)


@dataclass
class FileRecord:
    """
    Thông tin file PDF gốc.
    Sinh ra ở đầu pipeline, trước khi chunk.
    """
    file_id: str            # UUID duy nhất, sinh = new_id()
    filename: str           # tên file gốc, vd "hopdong.pdf"
    minio_path: str         # đường dẫn trên MinIO (nếu có)
    size_bytes: int         # dung lượng file
    uploaded_at: datetime = field(default_factory=utc_now)
    # ↑ tự động lấy thời gian hiện tại khi tạo object


@dataclass
class ChunkDoc:
    """
    Một đoạn text đã cắt từ PDF.
    Output của chunker, input của embedder.
    """
    chunk_id: str           # UUID riêng của chunk này
    file_id: str            # FK: chunk này thuộc file nào
    text: str               # nội dung text
    page_start: int         # trang bắt đầu
    page_end: int           # trang kết thúc
    chunk_index: int        # thứ tự chunk trong file (0, 1, 2...)
    char_count: int = 0     # tự động tính sau
    filename: str = ""       # tên file gốc, để trả source cho retrieval

    def __post_init__(self):
        """
        Tự động chạy sau khi tạo xong object.
        Đếm số ký tự của text.
        """
        self.char_count = len(self.text)


@dataclass
class EmbeddedChunk:
    """
    ChunkDoc + vector embedding.
    Output của embedder, input của store.
    """
    chunk_id: str           # UUID riêng
    file_id: str            # FK về FileRecord
    text: str               # nội dung text
    page_start: int         # trang bắt đầu
    page_end: int           # trang kết thúc
    chunk_index: int        # thứ tự chunk
    vector: list[float]     # mảng số, vd [0.12, -0.45, ...], 384 hoặc 1536 chiều
    model_name: str         # tên model đã dùng để embed
    filename: str = ""       # tên file gốc

    @classmethod
    def from_chunk(cls, chunk: ChunkDoc, vector: list[float], model_name: str) -> "EmbeddedChunk":
        """
        Tiện ích: copy dữ liệu từ ChunkDoc sang EmbeddedChunk,
        chỉ thêm vector + model_name.
        """
        return cls(
            chunk_id=chunk.chunk_id,
            file_id=chunk.file_id,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_index=chunk.chunk_index,
            vector=vector,
            model_name=model_name,
            filename=chunk.filename,
        )


@dataclass
class StoreResult:
    """
    Kết quả sau khi lưu xuống Chroma + MongoDB.
    Output của store, trả về cho pipeline.
    """
    file_id: str            # file đã lưu
    total_chunks: int       # tổng số chunk
    chroma_ids: list[str]   # id trên Chroma (để rollback nếu cần)
    mongo_ids: list[str]    # id trên MongoDB
    stored_at: datetime = field(default_factory=utc_now)
    # ↑ tự động lấy thời gian lưu
