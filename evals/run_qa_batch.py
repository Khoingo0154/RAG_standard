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

def format_top_5_k(sources: list) -> str:
    """Định dạng danh sách Top-5 K chunks thành chuỗi text cho CSV."""
    if not sources:
        return "Không có chunks"
    items = []
    for i, s in enumerate(sources[:5], 1):
        p_start = s.get("page_start")
        p_end = s.get("page_end")
        page_str = f"Trang {p_start}–{p_end}" if p_start != p_end else f"Trang {p_start}"
        score = round(float(s.get("score", 0.0)), 4)
        snippet = s.get("text", "").replace("\n", " ")[:70].strip()
        items.append(f"[{i}] {page_str} (score {score}): \"{snippet}...\"")
    return " | ".join(items)


def format_top_5_k_md(sources: list) -> str:
    """Định dạng danh sách Top-5 K chunks thành danh sách bullet có HTML <br> cho Markdown."""
    if not sources:
        return "Không có chunks"
    items = []
    for i, s in enumerate(sources[:5], 1):
        p_start = s.get("page_start")
        p_end = s.get("page_end")
        page_str = f"Trang {p_start}–{p_end}" if p_start != p_end else f"Trang {p_start}"
        score = round(float(s.get("score", 0.0)), 4)
        snippet = s.get("text", "").replace("\n", " ")[:60].strip().replace("|", "\\|")
        items.append(f"[{i}] {page_str} (`{score}`): <i>{snippet}...</i>")
    return "<br>".join(items)


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
        query = case.get("query", "")
        key_fact = case.get("key_fact", "")

        print(f"[{idx:02d}/{total:02d}] Hỏi: \"{query}\"")
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

                top_5_k_csv = format_top_5_k(sources)
                top_5_k_md = format_top_5_k_md(sources)

                print(f"     -> Phản hồi trong {elapsed_total_ms:.0f}ms | Top Score: {top_score} | Citation: {citation}")
                print(f"     -> Intent: {intent}")
                print("     -> Top-5 K Chunks được truy xuất:")
                for s_i, s in enumerate(sources[:5], 1):
                    p_start = s.get("page_start")
                    p_end = s.get("page_end")
                    page_str = f"Trang {p_start}–{p_end}" if p_start != p_end else f"Trang {p_start}"
                    s_score = round(float(s.get("score", 0.0)), 4)
                    snippet = s.get("text", "").replace("\n", " ")[:75].strip()
                    print(f"        [{s_i}] {page_str} (Score: {s_score}): \"{snippet}...\"")
                print("-" * 75)
                results.append({
                    "stt": idx,
                    "query": query,
                    "answer": answer,
                    "citation": citation,
                    "top_score": top_score,
                    "top_5_k": top_5_k_csv,
                    "top_5_k_md": top_5_k_md,
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
                "query": query,
                "answer": f"LỖI: {e}",
                "citation": "N/A",
                "top_score": 0.0,
                "top_5_k": "N/A",
                "top_5_k_md": "N/A",
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
        "Câu hỏi",
        "Câu trả lời RAG",
        "Trang PDF trích dẫn",
        "Top_5_K",
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
                "Câu hỏi": r["query"],
                "Câu trả lời RAG": clean_for_csv(r["answer"]),
                "Trang PDF trích dẫn": r["citation"],
                "Top_5_K": clean_for_csv(r.get("top_5_k", "")),
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
        f.write("| STT | Câu hỏi | Câu trả lời RAG | Trang PDF trích dẫn | Top 5 Chunks (Top-5 K) | Điểm Top Chunk | Thời gian (ms) | Điểm mấu chốt đối chiếu |\n")
        f.write("|:---:|---|---|:---:|---|:---:|:---:|---|\n")
        for r in results:
            f.write(
                f"| {r['stt']} | {clean_for_md(r['query'])} | "
                f"{clean_for_md(r['answer'])} | {r['citation']} | {r.get('top_5_k_md', '')} | {r['top_score']} | "
                f"{r['time_ms']} | {clean_for_md(r['key_fact'])} |\n"
            )

    print(f"[XONG] Đã xuất file Markdown: {OUTPUT_MD_PATH}")
    print("=" * 75)


if __name__ == "__main__":
    run_batch_tests()
