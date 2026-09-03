import os
from shared.config import Settings


class TestSettingsEnvOverride:
    def test_env_override_string(self):
        os.environ["CHUNK_STRATEGY"] = "page"
        s = Settings()
        assert s.CHUNK_STRATEGY == "page"
        del os.environ["CHUNK_STRATEGY"]

    def test_env_override_int_auto_cast(self):
        os.environ["CHUNK_SIZE"] = "500"
        s = Settings()
        assert s.CHUNK_SIZE == 500
        del os.environ["CHUNK_SIZE"]

    def test_env_override_top_k(self):
        os.environ["RETRIEVAL_TOP_K"] = "10"
        s = Settings()
        assert s.RETRIEVAL_TOP_K == 10
        del os.environ["RETRIEVAL_TOP_K"]

    def test_env_override_with_defaults(self):
        os.environ["EMBED_PROVIDER"] = "ollama"
        os.environ["EMBED_BATCH_SIZE"] = "64"
        s = Settings()
        assert s.EMBED_PROVIDER == "ollama"
        assert s.EMBED_BATCH_SIZE == 64
        assert s.CHROMA_PATH == "./chroma_db"
        assert s.MONGO_URI == "mongodb://localhost:27017"
        del os.environ["EMBED_PROVIDER"]
        del os.environ["EMBED_BATCH_SIZE"]
