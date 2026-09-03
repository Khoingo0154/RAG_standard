import os
import re
import time
import logging
from abc import ABC, abstractmethod
from ingestion.models import ChunkDoc, EmbeddedChunk

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """
    Khuôn mẫu (abstract class) cho tất cả embedder.
    Bất kỳ embedder nào cũng phải có:
    - model_name: tên model
    - _embed_batch(texts): lõi xử lý (gọi API hoặc chạy local)
    - embed_chunks(chunks): code dùng chung, đã viết sẵn ở đây
    """

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    def embed_chunks(self, chunks: list[ChunkDoc], batch_size: int = 32) -> list[EmbeddedChunk]:
        """
        Nhận list ChunkDoc, trả list EmbeddedChunk (chunk + vector).
        Tự động xử lý theo batch, không cần quan tâm số lượng chunk lớn hay nhỏ.
        """
        if batch_size <= 0:
            raise ValueError("batch_size phải lớn hơn 0")

        results = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = self._embed_batch([c.text for c in batch])
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedder trả về {len(vectors)} vector cho {len(batch)} chunk"
                )
            for chunk, vector in zip(batch, vectors):
                results.append(EmbeddedChunk.from_chunk(chunk, vector, self.model_name))
            if i + batch_size < len(chunks):
                time.sleep(0.1)
        return results

    def embed_query(self, query: str) -> list[float]:
        """Tạo vector embedding cho một chuỗi truy vấn (query)."""
        vectors = self._embed_batch([query])
        if not vectors:
            raise RuntimeError("Embedder không trả về vector cho query")
        return vectors[0]


class OpenAIEmbedder(BaseEmbedder):
    """
    Embed bằng OpenAI API.
    Model: text-embedding-3-small (1536 chiều)
    Yêu cầu: biến môi trường OPENAI_API_KEY hoặc truyền api_key.
    """

    MODEL = "text-embedding-3-small"

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    @property
    def model_name(self) -> str:
        return self.MODEL

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.MODEL, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


class GeminiEmbedder(BaseEmbedder):
    """Embed qua Gemini API; không tải hoặc chạy model embedding trên laptop."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        from google import genai
        from shared.config import settings

        self._model_name = model or settings.GEMINI_EMBED_MODEL
        self._dimensions = dimensions or settings.GEMINI_EMBED_DIMENSIONS
        self._task_type = task_type
        key = api_key or settings.GEMINI_API_KEY
        if not key:
            raise ValueError("Thiếu GEMINI_API_KEY để tạo embedding")
        self._client = genai.Client(api_key=key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        # Free Tier có quota theo phút. Khi Gemini trả RetryInfo, chờ đúng thời gian
        # được khuyến nghị rồi tiếp tục batch hiện tại thay vì làm hỏng cả ingest.
        for attempt in range(3):
            try:
                response = self._client.models.embed_content(
                    model=self._model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=self._task_type,
                        output_dimensionality=self._dimensions,
                    ),
                )
                return [embedding.values for embedding in response.embeddings]
            except Exception as error:
                match = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", str(error), re.IGNORECASE)
                if "RESOURCE_EXHAUSTED" not in str(error) or not match or attempt == 2:
                    raise
                wait_seconds = float(match.group(1)) + 1
                logger.warning(
                    "Gemini embedding quota reached; retrying this batch in %.0f seconds",
                    wait_seconds,
                )
                time.sleep(wait_seconds)

        raise RuntimeError("Không thể tạo Gemini embedding")


class LocalEmbedder(BaseEmbedder):
    """
    Embed chạy ngay trên máy local.
    Provider cũ, chỉ dùng nếu người học tự cài thêm sentence-transformers.
    Không cần API key, không cần mạng.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = SentenceTransformer(self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.encode(texts, convert_to_numpy=True)]


class OllamaEmbedder(BaseEmbedder):
    """
    Embed bằng Ollama server chạy trên máy ảo (hoặc local).
    Gọi API POST /api/embed của Ollama.

    Mặc định: phi4-mini (nhẹ, nhanh, phù hợp newbie)

    Nếu muốn đổi model:
      OllamaEmbedder(base_url="http://100.71.230.7:11434", model="qwen3:8b")
    """

    def __init__(
        self,
        base_url: str = "",
        model: str = "",
    ):
        from shared.config import settings
        import requests
        self._requests = requests
        self._base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self._model = model or settings.OLLAMA_MODEL

    @property
    def model_name(self) -> str:
        return self._model

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._requests.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


def get_embedder(provider: str = "gemini", task_type: str = "RETRIEVAL_DOCUMENT") -> BaseEmbedder:
    """
    Nhà máy sản xuất embedder.
    Truyền "gemini" → GeminiEmbedder (gọi Gemini API)
    Truyền "local" → LocalEmbedder (cần cài thêm sentence-transformers)
    Truyền "openai" → OpenAIEmbedder (gọi API)
    Truyền "ollama" → OllamaEmbedder (gọi server Ollama)
    """
    if provider == "openai":
        return OpenAIEmbedder()
    elif provider == "gemini":
        return GeminiEmbedder(task_type=task_type)
    elif provider == "local":
        return LocalEmbedder()
    elif provider == "ollama":
        return OllamaEmbedder()
    raise ValueError(f"Unknown provider: {provider}")
