from shared.config import Settings


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.CHUNK_SIZE == 1000
        assert s.CHUNK_OVERLAP == 200
        assert s.EMBED_PROVIDER == "gemini"
        assert s.RETRIEVAL_TOP_K == 5
        assert s.MONGO_URI == "mongodb://localhost:27017"

    def test_custom_values(self):
        s = Settings(
            CHUNK_SIZE=500,
            EMBED_PROVIDER="ollama",
            MONGO_URI="mongodb://custom:27017",
        )
        assert s.CHUNK_SIZE == 500
        assert s.EMBED_PROVIDER == "ollama"
        assert s.MONGO_URI == "mongodb://custom:27017"
