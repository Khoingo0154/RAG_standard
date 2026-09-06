import logging
from enum import Enum
from typing import Optional

from shared.config import settings

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    DOCUMENT_QUERY = "DOCUMENT_QUERY"
    FOOTBALL_LIVE_DATA = "FOOTBALL_LIVE_DATA"
    CHITCHAT = "CHITCHAT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


ROUTER_SYSTEM_PROMPT = """Bạn là bộ phân loại ý định người dùng (Query Intent Router) cho hệ thống trợ lý bóng đá thông minh.
Nhiệm vụ của bạn là đọc câu hỏi của người dùng và phân loại vào đúng 1 trong 4 nhóm sau:

1. DOCUMENT_QUERY:
   - Câu hỏi về luật thi đấu bóng đá, quy định trận đấu theo sách Luật FIFA (Law_fifa.pdf), lỗi vi phạm, thẻ phạt, trọng tài, kích thước sân, bóng, quả phạt đền, việt vị, ném biên, phạt góc, quy định thay người, v.v.
   - Ví dụ: "Việt vị là gì?", "Khi nào trọng tài rút thẻ đỏ?", "Khoảng cách chấm 11m là bao nhiêu?"

2. FOOTBALL_LIVE_DATA:
   - Câu hỏi về dữ liệu bóng đá thực tế: tỷ số trận đấu, kết quả thi đấu, tỷ số trực tiếp, lịch thi đấu, bảng xếp hạng, thông tin câu lạc bộ / đội bóng (thành lập năm nào, sân vận động), thông tin cầu thủ (sinh năm bao nhiêu, quốc tịch, chiều cao, cân nặng, thành tích, ảnh).
   - Ví dụ: "Tỷ số Arsenal hôm nay", "Có trận nào đang đá trực tiếp không?", "Thông tin câu lạc bộ Real Madrid", "Messi sinh năm bao nhiêu quốc tịch gì?", "Xem bảng xếp hạng Ngoại Hạng Anh", "Ronaldo bao nhiêu tuổi?".

3. CHITCHAT:
   - Lời chào hỏi, cảm ơn, tạm biệt, hỏi thăm sức khỏe, câu hỏi về danh tính, vai trò, chức năng của trợ lý ảo.
   - Ví dụ: "Xin chào", "Hello", "Bạn là ai?", "Bạn tên gì?", "Bạn có thể làm được gì?", "Cảm ơn bạn nhé", "Tạm biệt".

4. OUT_OF_SCOPE:
   - Câu hỏi hoàn toàn không liên quan đến bóng đá hoặc luật thi đấu (ví dụ: thời tiết, công thức nấu ăn, viết code lập trình, chính trị, toán học, thơ ca, v.v.).
   - Ví dụ: "Thời tiết hôm nay thế nào?", "Chỉ tôi cách làm bánh mì", "Viết code Python sắp xếp mảng".

CHỈ TRẢ VỀ DUY NHẤT 1 TỪ KHÓA trong 4 từ: DOCUMENT_QUERY, FOOTBALL_LIVE_DATA, CHITCHAT, OUT_OF_SCOPE. Không thêm bất kỳ từ ngữ nào khác."""

class QueryRouter:
    """Bộ định tuyến phân loại ý định câu hỏi sử dụng Gemini Flash."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self._api_key = settings.GEMINI_API_KEY if gemini_api_key is None else gemini_api_key
        self._model = model or settings.ROUTER_GEMINI_MODEL

    def classify(self, query: str) -> QueryIntent:
        """Phân loại câu hỏi thành DOCUMENT_QUERY, CHITCHAT hoặc OUT_OF_SCOPE."""
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return QueryIntent.CHITCHAT

        if not self._api_key:
            logger.warning("Không có GEMINI_API_KEY, fallback về DOCUMENT_QUERY")
            return QueryIntent.DOCUMENT_QUERY

        try:
            from google import genai

            client = genai.Client(api_key=self._api_key)
            response = client.models.generate_content(
                model=self._model,
                contents=f"{ROUTER_SYSTEM_PROMPT}\n\nCâu hỏi người dùng: \"{cleaned_query}\"\nÝ định:",
            )

            raw_result = (response.text or "").strip().upper()
            logger.info(f"QueryRouter phân loại: '{cleaned_query}' -> {raw_result}")

            if "CHITCHAT" in raw_result:
                return QueryIntent.CHITCHAT
            elif "FOOTBALL_LIVE_DATA" in raw_result:
                return QueryIntent.FOOTBALL_LIVE_DATA
            elif "OUT_OF_SCOPE" in raw_result:
                return QueryIntent.OUT_OF_SCOPE
            elif "DOCUMENT_QUERY" in raw_result:
                return QueryIntent.DOCUMENT_QUERY
            else:
                logger.warning(f"Router trả về kết quả không khớp ({raw_result}), fallback về DOCUMENT_QUERY")
                return QueryIntent.DOCUMENT_QUERY
        except Exception as e:
            logger.error(f"Lỗi khi phân loại câu hỏi: {e}. Fallback về DOCUMENT_QUERY.")
            return QueryIntent.DOCUMENT_QUERY
