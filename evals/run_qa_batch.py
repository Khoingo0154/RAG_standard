"""Script chạy kiểm thử tự động 20 câu hỏi Luật bóng đá qua API và xuất báo cáo CSV + Markdown.

Cách chạy:
    python -m evals.run_qa_batch
hoặc click đúp file:
    run_qa_test.cmd
"""

__test__ = False  # Đảm bảo pytest không bao giờ tự động quét hoặc chạy file này

import csv
import json
import re
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

ROOT_DIR = Path(__file__).resolve().parent.parent
TESTSET_PATH = ROOT_DIR / "evals" / "fifa_qa_testset.json"
OUTPUT_CSV_PATH = ROOT_DIR / "evals" / "results_qa_testset.csv"
OUTPUT_MD_PATH = ROOT_DIR / "evals" / "results_qa_testset.md"
API_URL = "http://localhost:8000/query"
DEFAULT_FILE_ID = "8f525964-a864-43ef-b87b-34f6482d44f1"


def extract_citations(answer: str, sources: list) -> str:
    """Trích xuất thông tin trang tham khảo từ câu trả lời hoặc sources."""
    match = re.search(r"\(Tham khảo tại (.*?)\.?\)", answer)
    if match:
        return match.group(1).strip()
    
    # Fallback từ sources nếu câu trả lời không có text citation
    pages = set()
    for s in sources:
        start = s.get("page_start")
        end = s.get("page_end")
        if start and end:
            pages.add(f"trang {start}–{end}" if start != end else f"trang {start}")
    return ", ".join(sorted(pages)) if pages else "Không có trích dẫn"


def clean_for_csv(text: str) -> str:
    """Làm sạch chuỗi cho ô tính CSV, gom khoảng trắng và dòng mới."""
    if not text:
        return ""
    return " ".join(text.split())


def clean_for_md(text: str) -> str:
    """Làm sạch chuỗi cho ô Markdown, thay newline bằng <br> và escape ký tự |."""
    if not text:
        return ""
    escaped = text.replace("|", "\\|")
    return "<br>".join(line.strip() for line in escaped.splitlines() if line.strip())


def run_batch_tests():
    if not TESTSET_PATH.exists():
        print(f"[ERROR] Không tìm thấy file testset tại {TESTSET_PATH}")
        sys.exit(1)

    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    total = len(cases)
    print("=" * 75)
    print(f"BẮT ĐẦU CHẠY KIỂM THỬ BATCH QA ({total} CÂU HỎI LUẬT BÓNG ĐÁ)")
    print(f"Target API: {API_URL}")
    print("=" * 75)

    results = []
    success_count = 0

    for idx, case in enumerate(cases, 1):
        q_id = case.get("id")
        law_group = case.get("law_group", "N/A")
        query = case.get("query", "")
        key_fact = case.get("key_fact", "")

        print(f"[{idx:02d}/{total:02d}] [{law_group}]")
        print(f"     Hỏi: \"{query}\"")

        payload = {
            "query": query,
            "file_id": DEFAULT_FILE_ID,
            "top_k": 5,
        }

        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                elapsed_total_ms = (time.time() - t0) * 1000
                res_json = json.loads(resp.read().decode("utf-8"))

                answer = res_json.get("answer", "").strip()
                sources = res_json.get("sources", [])
                api_time_ms = res_json.get("query_time_ms", elapsed_total_ms)
                intent = res_json.get("intent", "N/A")
                citation = extract_citations(answer, sources)

                top_score = 0.0
                if sources and "score" in sources[0]:
                    top_score = round(sources[0]["score"], 4)

                print(f"     -> Phản hồi trong {elapsed_total_ms:.0f}ms | Top Score: {top_score} | Citation: {citation}")
                print(f"     -> Intent: {intent}")
                print("-" * 75)

                results.append({
                    "stt": idx,
                    "law_group": law_group,
                    "query": query,
                    "answer": answer,
                    "citation": citation,
                    "top_score": top_score,
                    "time_ms": round(elapsed_total_ms, 1),
                    "intent": intent,
                    "key_fact": key_fact,
                    "status": "OK",
                })
                success_count += 1
        except Exception as e:
            elapsed_total_ms = (time.time() - t0) * 1000
            print(f"     [!] LỖI KHI GỌI API: {e} ({elapsed_total_ms:.0f}ms)")
            print("-" * 75)
            results.append({
                "stt": idx,
                "law_group": law_group,
                "query": query,
                "answer": f"LỖI: {e}",
                "citation": "N/A",
                "top_score": 0.0,
                "time_ms": round(elapsed_total_ms, 1),
                "intent": "ERROR",
                "key_fact": key_fact,
                "status": "ERROR",
            })

        # Delay nhẹ giữa các câu để tránh spike rate limit của Gemini Free Tier
        if idx < total:
            time.sleep(2.5)

    # 1. Xuất file CSV (với BOM utf-8-sig để Excel mở không lỗi font tiếng Việt)
    fieldnames = [
        "STT",
        "Nhóm luật",
        "Câu hỏi",
        "Câu trả lời RAG",
        "Trang PDF trích dẫn",
        "Điểm Top Chunk",
        "Thời gian (ms)",
        "Ý định (Intent)",
        "Điểm mấu chốt đối chiếu",
    ]

    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "STT": r["stt"],
                "Nhóm luật": r["law_group"],
                "Câu hỏi": r["query"],
                "Câu trả lời RAG": clean_for_csv(r["answer"]),
                "Trang PDF trích dẫn": r["citation"],
                "Điểm Top Chunk": r["top_score"],
                "Thời gian (ms)": r["time_ms"],
                "Ý định (Intent)": r["intent"],
                "Điểm mấu chốt đối chiếu": clean_for_csv(r["key_fact"]),
            })
    print(f"\n[XONG] Đã xuất file CSV: {OUTPUT_CSV_PATH}")

    # 2. Xuất file Markdown
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO KẾT QUẢ KIỂM THỬ 20 CÂU HỎI RAG - LUẬT BÓNG ĐÁ FIFA\n\n")
        f.write(f"- **Ngày chạy kiểm thử:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Tổng số câu hỏi:** {total}\n")
        f.write(f"- **Số câu thành công:** {success_count}/{total}\n")
        avg_time = sum(r['time_ms'] for r in results) / len(results) if results else 0
        f.write(f"- **Thời gian phản hồi trung bình:** {avg_time:.1f} ms\n\n")
        f.write("---\n\n")
        f.write("## BẢNG KẾT QUẢ ĐỐI CHIẾU CHI TIẾT\n\n")
        f.write("| STT | Nhóm luật | Câu hỏi | Câu trả lời RAG | Trang PDF trích dẫn | Điểm Top Chunk | Thời gian (ms) | Điểm mấu chốt đối chiếu |\n")
        f.write("|:---:|---|---|---|:---:|:---:|:---:|---|\n")
        for r in results:
            f.write(
                f"| {r['stt']} | {r['law_group']} | {clean_for_md(r['query'])} | "
                f"{clean_for_md(r['answer'])} | {r['citation']} | {r['top_score']} | "
                f"{r['time_ms']} | {clean_for_md(r['key_fact'])} |\n"
            )

    print(f"[XONG] Đã xuất file Markdown: {OUTPUT_MD_PATH}")
    print("=" * 75)


if __name__ == "__main__":
    run_batch_tests()
