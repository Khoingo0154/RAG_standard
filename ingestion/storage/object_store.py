"""Lớp lưu file gốc vào MinIO (dịch vụ tương thích Amazon S3)."""

from io import BytesIO
from pathlib import Path


class MinioObjectStore:
    """Lưu/xóa PDF gốc. Chunks và vectors vẫn do MongoDB/ChromaDB quản lý."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        from minio import Minio

        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket = bucket

    def upload_pdf(self, file_id: str, filename: str, pdf_bytes: bytes) -> str:
        """Tạo bucket nếu cần rồi ghi PDF, trả về object key để lưu làm metadata."""
        safe_filename = Path(filename).name
        object_key = f"documents/{file_id}/{safe_filename}"

        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)

        self._client.put_object(
            self.bucket,
            object_key,
            BytesIO(pdf_bytes),
            length=len(pdf_bytes),
            content_type="application/pdf",
        )
        return object_key

    def delete(self, object_key: str) -> None:
        """Xóa PDF gốc theo object key đã lưu trong FileRecord."""
        self._client.remove_object(self.bucket, object_key)

    def delete_file(self, file_id: str) -> int:
        """Xóa mọi object của một file_id; phù hợp cả với tên file gốc bất kỳ."""
        prefix = f"documents/{file_id}/"
        deleted = 0
        for obj in self._client.list_objects(self.bucket, prefix=prefix, recursive=True):
            self._client.remove_object(self.bucket, obj.object_name)
            deleted += 1
        return deleted
