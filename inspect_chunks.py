"""Script thống kê và đo đếm chính xác số lượng Ký tự, Từ, và Token của các chunks trong CSDL.

Cách chạy:
    docker compose --profile swagger run --rm api python inspect_chunks.py --file-id 8f525964-a864-43ef-b87b-34f6482d44f1
"""

import argparse
import json
import statistics
from pymongo import MongoClient

from shared.config import settings


def count_tokens_accurate(text: str, client=None, model: str = "gemini-1.5-flash") -> int:
    """Đếm token chính xác thông qua Gemini API count_tokens, fallback sang ước lượng."""
    if client:
        try:
            resp = client.models.count_tokens(model=model, contents=text)
            return resp.total_tokens
        except Exception:
            pass
    # Ước lượng chuẩn cho tiếng Anh/Việt (1 token ~ 4 ký tự hoặc ~ 0.75 từ)
    return max(1, int(len(text) / 4))


def inspect_chunks(file_id: str, sample_count: int = 3):
    print("=" * 70)
    print(f"📊 ĐANG PHÂN TÍCH CHUNKS TRONG CSDL CHO FILE_ID: {file_id}")
    print("=" * 70)

    client_mongo = MongoClient(settings.MONGO_URI)
    db = client_mongo[settings.MONGO_DB]
    col = db["chunks"]

    filter_query = {"file_id": file_id} if file_id else {}
    cursor = col.find(filter_query).sort("chunk_index", 1)
    chunks = list(cursor)

    if not chunks:
        print(f"❌ Không tìm thấy chunks nào cho file_id: '{file_id}' trong MongoDB.")
        return

    # Khởi tạo client Gemini để đếm token chính xác nếu có API key
    gemini_client = None
    if settings.GEMINI_API_KEY:
        try:
            from google import genai
            gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            print(" Connected to Gemini Tokenizer API (Đo đếm token bằng model chính hãng).")
        except Exception:
            print("⚠️ Dùng bộ ước lượng chuẩn Token (1 token ≈ 4 ký tự).")

    total_chunks = len(chunks)
    char_lengths = []
    word_lengths = []
    token_lengths = []

    print(f"\n⏳ Đang phân tích chi tiết {total_chunks} chunks...")

    for i, c in enumerate(chunks):
        text = c.get("text", "")
        chars = len(text)
        words = len(text.split())
        tokens = count_tokens_accurate(text, client=gemini_client)

        char_lengths.append(chars)
        word_lengths.append(words)
        token_lengths.append(tokens)

    # Thống kê tổng hợp
    print("\n" + "=" * 70)
    print("📈 BẢNG THỐNG KÊ TỔNG QUAN")
    print("=" * 70)
    print(f"• Tổng số Chunks:                {total_chunks}")
    print(f"• Tên file:                       {chunks[0].get('filename', 'Unknown')}")
    print(f"• Cấu hình chunk_size (.env):     {settings.CHUNK_SIZE} ký tự")
    print(f"• Cấu hình chunk_overlap (.env):  {settings.CHUNK_OVERLAP} ký tự")
    print("-" * 70)
    print(f"• Số KÝ TỰ (Characters):")
    print(f"    - Trung bình (Mean):          {statistics.mean(char_lengths):.1f} ký tự")
    print(f"    - Trung vị (Median):          {statistics.median(char_lengths):.1f} ký tự")
    print(f"    - Nhỏ nhất (Min):             {min(char_lengths)} ký tự")
    print(f"    - Lớn nhất (Max):             {max(char_lengths)} ký tự")
    print("-" * 70)
    print(f"• Số TỪ (Words):")
    print(f"    - Trung bình (Mean):          {statistics.mean(word_lengths):.1f} từ")
    print(f"    - Nhỏ nhất (Min):             {min(word_lengths)} từ")
    print(f"    - Lớn nhất (Max):             {max(word_lengths)} từ")
    print("-" * 70)
    print(f"• Số TOKEN CHÍNH XÁC (Tokens):")
    print(f"    - Trung bình (Mean):          {statistics.mean(token_lengths):.1f} TOKENS / CHUNK")
    print(f"    - Trung vị (Median):          {statistics.median(token_lengths):.1f} TOKENS / CHUNK")
    print(f"    - Nhỏ nhất (Min):             {min(token_lengths)} tokens")
    print(f"    - Lớn nhất (Max):             {max(token_lengths)} tokens")
    print("=" * 70)

    # Phân bố độ dài token
    bucket_under_100 = sum(1 for t in token_lengths if t < 100)
    bucket_100_200 = sum(1 for t in token_lengths if 100 <= t < 200)
    bucket_200_250 = sum(1 for t in token_lengths if 200 <= t <= 250)
    bucket_over_250 = sum(1 for t in token_lengths if t > 250)

    print("\n📊 PHÂN BỐ KÍCH THƯỚC TOKEN TRÊN TOÀN BỘ TÀI LIỆU:")
    print(f"  • < 100 tokens (đoạn ngắn/tiêu đề):     {bucket_under_100:>3} chunks ({bucket_under_100/total_chunks*100:.1f}%)")
    print(f"  • 100 - 200 tokens:                     {bucket_100_200:>3} chunks ({bucket_100_200/total_chunks*100:.1f}%)")
    print(f"  • 200 - 250 tokens (vùng tối ưu chính): {bucket_200_250:>3} chunks ({bucket_200_250/total_chunks*100:.1f}%)")
    print(f"  • > 250 tokens:                         {bucket_over_250:>3} chunks ({bucket_over_250/total_chunks*100:.1f}%)")

    # In mẫu một vài chunks thực tế
    print("\n" + "=" * 70)
    print(f"🔍 XEM MẪU {sample_count} CHUNKS THỰC TẾ TRONG DATABASE:")
    print("=" * 70)

    sample_indices = [0, total_chunks // 2, total_chunks - 1]
    for idx in sample_indices[:sample_count]:
        c = chunks[idx]
        text = c.get("text", "")
        tokens = token_lengths[idx]
        print(f"\n📄 [CHUNK #{c.get('chunk_index')} | Trang {c.get('page_start')}–{c.get('page_end')}]")
        print(f"• ID: {c.get('chunk_id')}")
        print(f"• Độ dài: {len(text)} ký tự | {len(text.split())} từ | 👉 {tokens} TOKENS")
        print(f"• Nội dung xem trước: \"{text[:200]}...\"")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra chi tiết số lượng ký tự và token của Chunks trong DB")
    parser.add_argument("--file-id", default="8f525964-a864-43ef-b87b-34f6482d44f1", help="ID của file PDF cần kiểm tra")
    parser.add_argument("--samples", type=int, default=3, help="Số lượng mẫu chunk cần in chi tiết")
    args = parser.parse_args()

    inspect_chunks(file_id=args.file_id, sample_count=args.samples)


if __name__ == "__main__":
    main()
