import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class Settings:
    CHUNK_STRATEGY: str = "token"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    EMBED_PROVIDER: str = "gemini"
    EMBED_BATCH_SIZE: int = 32
    GEMINI_EMBED_MODEL: str = "gemini-embedding-001"
    GEMINI_EMBED_DIMENSIONS: int = 768

    # === Telegram bot (tùy chọn) ===
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_DEFAULT_FILE_ID: Optional[str] = None
    TELEGRAM_CONNECT_TIMEOUT: int = 30
    TELEGRAM_READ_TIMEOUT: int = 45
    TELEGRAM_WRITE_TIMEOUT: int = 45
    TELEGRAM_POOL_TIMEOUT: int = 30

    CHROMA_PATH: str = "./chroma_db"
    CHROMA_COLLECTION: str = "rag_chunks"

    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "rag_db"

    # === Object storage (MinIO / S3-compatible) ===
    MINIO_ENABLED: bool = True
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "rag-documents"
    MINIO_SECURE: bool = False

    OLLAMA_BASE_URL: str = "http://100.71.230.7:11434"
    OLLAMA_MODEL: str = "phi4-mini"

    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    RETRIEVAL_TOP_K: int = 5

    LLM_PROVIDER: str = "gemini"
    LLM_OLLAMA_MODEL: str = "phi4-mini"
    LLM_GEMINI_MODEL: str = "gemini-3.1-flash-lite"
    ROUTER_GEMINI_MODEL: str = "gemini-3.1-flash-lite"

    UPLOAD_DIR: str = "./uploads"

    def __post_init__(self):
        for key, definition in self.__dataclass_fields__.items():
            env_val = os.getenv(key)
            # Cấu hình truyền trực tiếp vào Settings() phải luôn được ưu tiên.
            if env_val is None or getattr(self, key) != definition.default:
                continue

            if isinstance(definition.default, bool):
                setattr(self, key, env_val.strip().lower() in {"1", "true", "yes", "on"})
            elif isinstance(definition.default, int):
                setattr(self, key, int(env_val))
            else:
                setattr(self, key, env_val)


settings = Settings()
