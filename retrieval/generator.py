import logging
import requests
from typing import Optional

from retrieval.models import RetrievedChunk
from shared.config import settings

logger = logging.getLogger(__name__)

_MONGO_CLIENT = None
_MONGO_COL = None


def _get_mongo_col(mongo_uri: str, mongo_db: str):
    global _MONGO_CLIENT, _MONGO_COL
    if _MONGO_COL is None:
        from pymongo import MongoClient
        _MONGO_CLIENT = MongoClient(mongo_uri)
        _MONGO_COL = _MONGO_CLIENT[mongo_db]["chunks"]
    return _MONGO_COL


class Generator:
    def __init__(
        self,
        provider: str = "ollama",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        mongo_db: Optional[str] = None,
    ):
        self._provider = provider
        self._base_url = base_url or settings.OLLAMA_BASE_URL
        self._ollama_model = model or settings.LLM_OLLAMA_MODEL
        self._gemini_model = model or settings.LLM_GEMINI_MODEL
        # None nghĩa là dùng cấu hình mặc định; chuỗi rỗng nghĩa là caller
        # chủ động không cung cấp key và phải được báo lỗi rõ ràng.
        self._gemini_api_key = settings.GEMINI_API_KEY if gemini_api_key is None else gemini_api_key
        self._mongo_uri = mongo_uri or settings.MONGO_URI
        self._mongo_db = mongo_db or settings.MONGO_DB

    def _lookup_metadata(self, chunk_id: str) -> Optional[dict]:
        try:
            col = _get_mongo_col(self._mongo_uri, self._mongo_db)
            doc = col.find_one({"chunk_id": chunk_id})
            return doc
        except Exception:
            return None

    def _enrich_sources(self, hits: list[dict]) -> list[RetrievedChunk]:
        sources = []
        for hit in hits:
            meta = self._lookup_metadata(hit["chunk_id"])
            filename = "unknown"
            if meta:
                filename = meta.get("filename", "unknown")
            sources.append(RetrievedChunk(
                chunk_id=hit["chunk_id"],
                file_id=hit["file_id"],
                text=hit["text"],
                page_start=hit.get("page_start", 0),
                page_end=hit.get("page_end", 0),
                chunk_index=hit.get("chunk_index", 0),
                score=hit["score"],
                filename=filename,
            ))
        return sources

    def _build_prompt(self, query: str, hits: list[dict]) -> str:
        context_parts = []
        for i, hit in enumerate(hits):
            context_parts.append(f"[{i+1}] {hit['text']}")
        context = "\n\n".join(context_parts)
        prompt = (
            f"Dựa vào các đoạn văn bản sau đây, hãy trả lời câu hỏi.\n\n"
            f"=== NGỮ CẢNH ===\n{context}\n\n"
            f"=== CÂU HỎI ===\n{query}\n\n"
            f"Câu trả lời:"
        )
        return prompt

    @staticmethod
    def _format_references(sources: list[RetrievedChunk]) -> str:
        """Tạo chú thích trang PDF từ các chunks thực sự được đưa vào ngữ cảnh."""
        pages_by_file: dict[str, list[str]] = {}
        seen: set[tuple[str, int, int]] = set()

        for source in sources:
            if source.page_start <= 0:
                continue
            key = (source.filename, source.page_start, source.page_end)
            if key in seen:
                continue
            seen.add(key)

            page_label = (
                str(source.page_start)
                if source.page_start == source.page_end
                else f"{source.page_start}–{source.page_end}"
            )
            pages_by_file.setdefault(source.filename, []).append(page_label)

        if not pages_by_file:
            return ""

        references = "; ".join(
            f"trang {', '.join(pages)} của PDF {filename}"
            for filename, pages in pages_by_file.items()
        )
        return f"(Tham khảo tại {references}.)"

    def generate(
        self,
        query: str,
        hits: list[dict],
        system_prompt: Optional[str] = None,
    ) -> tuple[str, list[RetrievedChunk]]:
        if not hits:
            return "Không tìm thấy thông tin liên quan.", []

        prompt = self._build_prompt(query, hits)
        sources = self._enrich_sources(hits)

        if self._provider == "gemini":
            answer = self._generate_gemini(prompt, system_prompt)
        elif self._provider == "openai":
            answer = self._generate_openai(prompt, system_prompt)
        elif self._provider == "ollama":
            answer = self._generate_ollama(prompt, system_prompt)
        else:
            answer = f"LLM provider không được hỗ trợ: {self._provider}"

        is_error = answer.startswith(("Lỗi ", "Thiếu ", "LLM provider không được hỗ trợ"))
        if not is_error:
            references = self._format_references(sources)
            if references:
                answer = f"{answer.rstrip()}\n\n{references}"

        return answer, sources

    def generate_chitchat(self, query: str) -> str:
        """Sinh câu trả lời xã giao/chào hỏi trực tiếp mà không cần ngữ cảnh tài liệu."""
        system_prompt = (
            "Bạn là trợ lý AI chuyên nghiệp về Luật bóng đá FIFA. "
            "Hãy chào hỏi hoặc phản hồi lời xã giao của người dùng một cách thân thiện, ngắn gọn bằng tiếng Việt, "
            "và gợi ý bạn sẵn sàng hỗ trợ giải đáp các thắc mắc về luật thi đấu bóng đá."
        )
        if self._provider == "gemini":
            return self._generate_gemini(query, system_prompt=system_prompt)
        elif self._provider == "openai":
            return self._generate_openai(query, system_prompt=system_prompt)
        elif self._provider == "ollama":
            return self._generate_ollama(query, system_prompt=system_prompt)
        return "Chào bạn! Tôi là trợ lý Luật bóng đá FIFA. Tôi có thể giúp gì cho bạn hôm nay?"

    def _generate_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        else:
            payload["system"] = "Bạn là trợ lý trả lời câu hỏi dựa trên ngữ cảnh được cung cấp. Trả lời bằng tiếng Việt."

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generate failed: {e}")
            return f"Lỗi khi tạo câu trả lời: {e}"

    def _generate_gemini(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Gọi Gemini Generate Content API bằng SDK google-genai chính thức, có auto-retry khi gặp rate limit."""
        if not self._gemini_api_key:
            return "Thiếu GEMINI_API_KEY. Hãy đặt khóa API trong file .env rồi khởi động lại API."

        instruction = system_prompt or (
            "Bạn là trợ lý trả lời câu hỏi dựa trên ngữ cảnh được cung cấp. "
            "Trả lời bằng tiếng Việt."
        )
        import re
        import time
        from google import genai

        client = genai.Client(api_key=self._gemini_api_key)
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(
                    model=self._gemini_model,
                    contents=f"{instruction}\n\n{prompt}",
                )
                return (response.text or "").strip()
            except Exception as e:
                err_str = str(e)
                # Xử lý Rate Limit (429 RESOURCE_EXHAUSTED) hoặc Tạm thời quá tải (503 UNAVAILABLE)
                is_rate_limit = any(marker in err_str for marker in ["RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503"])
                if is_rate_limit and attempt < max_attempts - 1:
                    match = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                    if match:
                        wait_seconds = float(match.group(1)) + 1.0
                    else:
                        wait_seconds = (attempt + 1) * 4.0
                    logger.warning(
                        f"Gemini API gặp giới hạn tạm thời ({err_str[:80]}...). Thử lại lần {attempt + 1}/{max_attempts - 1} sau {wait_seconds:.1f}s..."
                    )
                    time.sleep(wait_seconds)
                    continue

                logger.error(f"Gemini generate failed: {e}")
                return f"Lỗi khi tạo câu trả lời bằng Gemini: {e}"
    def _generate_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        from openai import OpenAI
        api_key = settings.OPENAI_API_KEY
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt or "Bạn là trợ lý trả lời câu hỏi. Trả lời bằng tiếng Việt."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI generate failed: {e}")
            return f"Lỗi khi tạo câu trả lời: {e}"
