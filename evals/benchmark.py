"""Lớp RAGBenchmark chuyên nghiệp phục vụ kiểm thử, đo đạc và benchmark hệ thống RAG."""

import csv
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from evals.metrics import (
    extract_cited_pages,
    hit_at_k,
    keyword_coverage,
    pages_in_range,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    verify_citation_faithfulness,
)
from shared.config import settings

logger = logging.getLogger(__name__)

__test__ = False  # Ngăn pytest quét nhầm class Benchmark này như test suite pytest


@dataclass
class BenchmarkCase:
    """Đại diện cho một ca kiểm thử benchmark."""
    id: str
    query: str
    expected_pages: list[int] = field(default_factory=list)
    key_fact: str = ""
    category: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkCase":
        return cls(
            id=str(data.get("id", "")),
            query=str(data.get("query", "")),
            expected_pages=list(data.get("expected_pages", [])),
            key_fact=str(data.get("key_fact", "")),
            category=str(data.get("category", data.get("law_group", ""))),
        )


@dataclass
class CaseResult:
    """Kết quả kiểm thử chi tiết của 1 ca."""
    case_id: str
    query: str
    intent: str = "N/A"
    latency_ms: float = 0.0

    # Chỉ số Retrieval
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    hit_at_k: float = 0.0
    top_score: float = 0.0
    retrieved_pages: list[int] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)

    # Chỉ số Generation (nếu chạy e2e)
    answer: str = ""
    key_fact: str = ""
    fact_coverage: float = 0.0
    citation_info: dict = field(default_factory=dict)
    status: str = "OK"
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Bản tổng hợp kết quả toàn bộ đợt chạy benchmark."""
    dataset_name: str
    file_id: str
    mode: Literal["retrieval", "e2e"]
    top_k: int
    use_hybrid: bool
    use_rerank: bool
    timestamp: str
    total_cases: int
    successful_cases: int
    failed_cases: int

    # Trung bình các chỉ số
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_mrr: float
    mean_hit_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_fact_coverage: float
    citation_faithfulness_rate: float
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cases"] = [c.to_dict() for c in self.cases]
        return data

    def save_json(self, path: Path | str) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def save_csv(self, path: Path | str) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "STT",
            "ID",
            "Câu hỏi",
            "Intent",
            "Recall@K",
            "Precision@K",
            "MRR",
            "Hit@K",
            "Top_Score",
            "Thời gian (ms)",
            "Trang trích xuất",
            "Trang trích dẫn",
            "Độ bao phủ Fact",
            "Trích dẫn trung thực?",
            "Câu trả lời RAG",
            "Điểm mấu chốt đối chiếu",
            "Trạng thái",
        ]

        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, c in enumerate(self.cases, 1):
                cited_str = ", ".join(str(p) for p in c.citation_info.get("cited_pages", []))
                retrieved_str = ", ".join(str(p) for p in c.retrieved_pages)
                writer.writerow({
                    "STT": idx,
                    "ID": c.case_id,
                    "Câu hỏi": c.query,
                    "Intent": c.intent,
                    "Recall@K": round(c.recall_at_k, 4),
                    "Precision@K": round(c.precision_at_k, 4),
                    "MRR": round(c.mrr, 4),
                    "Hit@K": round(c.hit_at_k, 4),
                    "Top_Score": c.top_score,
                    "Thời gian (ms)": c.latency_ms,
                    "Trang trích xuất": retrieved_str,
                    "Trang trích dẫn": cited_str,
                    "Độ bao phủ Fact": round(c.fact_coverage, 4),
                    "Trích dẫn trung thực?": "ĐÚNG" if c.citation_info.get("is_faithful", True) else "ẢO GIÁC",
                    "Câu trả lời RAG": " ".join(c.answer.split()),
                    "Điểm mấu chốt đối chiếu": " ".join(c.key_fact.split()),
                    "Trạng thái": c.status,
                })
        return out_path

    def save_markdown(self, path: Path | str) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# BÁO CÁO KẾT QUẢ BENCHMARK — {self.dataset_name.upper()}\n",
            f"- **Thời gian thực hiện:** {self.timestamp}",
            f"- **Chế độ kiểm thử (Mode):** `{self.mode.upper()}`",
            f"- **Cấu hình:** Top-K = `{self.top_k}` | Hybrid Search = `{self.use_hybrid}` | FlashRank Reranker = `{self.use_rerank}`",
            f"- **Tổng số ca:** {self.total_cases} (Thành công: {self.successful_cases}, Lỗi: {self.failed_cases})\n",
            "## 1. BẢNG TỔNG KẾT CHỈ SỐ TOÀN CỤC\n",
            "| Chỉ số đo lường | Giá trị đạt được | Ý nghĩa kỹ thuật |",
            "|---|:---:|---|",
            f"| **Mean Recall@{self.top_k}** | **`{self.mean_recall_at_k:.1%}`** | Tỷ lệ không bỏ sót trang luật cần tìm |",
            f"| **Mean Precision@{self.top_k}** | **`{self.mean_precision_at_k:.1%}`** | Tỷ lệ chunks tinh khiết (không dính rác) |",
            f"| **Mean MRR** | **`{self.mean_mrr:.4f}`** | Thứ hạng của đoạn văn đúng đầu tiên |",
            f"| **Hit Rate@{self.top_k}** | **`{self.mean_hit_rate:.1%}`** | Tỷ lệ tìm trúng ít nhất 1 trang cần thiết |",
            f"| **Độ trễ trung bình** | **`{self.mean_latency_ms:.1f} ms`** | Thời gian phản hồi trung bình |",
            f"| **Độ trễ P95** | **`{self.p95_latency_ms:.1f} ms`** | 95% số ca phản hồi nhanh hơn mốc này |",
        ]

        if self.mode == "e2e":
            lines.extend([
                f"| **Fact Coverage** | **`{self.mean_fact_coverage:.1%}`** | Tỷ lệ bao phủ các ý mấu chốt đối chiếu |",
                f"| **Citation Faithfulness** | **`{self.citation_faithfulness_rate:.1%}`** | Tỷ lệ trích dẫn trung thực (không ảo giác) |",
            ])

        lines.extend([
            "\n---\n",
            "## 2. BẢNG CHI TIẾT TỪNG CA KIỂM THỬ\n",
            "| STT | ID | Câu hỏi | Recall | Precision | MRR | Latency | Chunks lấy về | Trích dẫn |",
            "|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])

        for idx, c in enumerate(self.cases, 1):
            retrieved_p = ", ".join(str(p) for p in c.retrieved_pages[:5])
            cited_p = ", ".join(str(p) for p in c.citation_info.get("cited_pages", [])) or "—"
            q_escaped = c.query.replace("|", "\\|")
            lines.append(
                f"| {idx} | `{c.case_id}` | {q_escaped} | `{c.recall_at_k:.2f}` | "
                f"`{c.precision_at_k:.2f}` | `{c.mrr:.2f}` | `{c.latency_ms:.0f}ms` | "
                f"{retrieved_p} | {cited_p} |"
            )

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path


class RAGBenchmark:
    """Lớp điều phối Benchmark hệ thống RAG (hỗ trợ đo Retrieval-only và Full E2E)."""

    def __init__(
        self,
        file_id: Optional[str] = None,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        api_url: Optional[str] = None,
        embed_provider: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ):
        self.file_id = file_id or getattr(settings, "TELEGRAM_DEFAULT_FILE_ID", "8f525964-a864-43ef-b87b-34f6482d44f1")
        self.top_k = top_k
        self.use_hybrid = use_hybrid
        self.use_rerank = use_rerank
        self.api_url = api_url
        self.embed_provider = embed_provider or settings.EMBED_PROVIDER
        self.llm_provider = llm_provider or settings.LLM_PROVIDER

    @staticmethod
    def load_dataset(dataset_path: Path | str) -> tuple[str, list[BenchmarkCase]]:
        """Nạp bộ dữ liệu kiểm thử từ file JSON."""
        p = Path(dataset_path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file dataset tại: {p}")

        data = json.loads(p.read_text(encoding="utf-8"))
        dataset_name = data.get("name", p.stem)
        raw_cases = data.get("cases", [])
        cases = [BenchmarkCase.from_dict(c) for c in raw_cases]
        return dataset_name, cases

    def evaluate_retrieval_case(self, case: BenchmarkCase) -> CaseResult:
        """Kiểm thử tầng Retrieval (chạy cục bộ không gọi LLM để đo tốc độ và chi phí 0 token)."""
        from ingestion.services.embedder import get_embedder
        from retrieval.pipeline import get_search_terms, retrieve_hybrid_and_rerank
        from retrieval.router import QueryRouter

        t0 = time.time()
        try:
            router = QueryRouter()
            intent = router.classify(case.query).value

            embedder = get_embedder(self.embed_provider, task_type="RETRIEVAL_QUERY")
            retrieval_query, sparse_query, rerank_query = get_search_terms(case.query)
            query_vector = embedder.embed_query(retrieval_query)

            hits = retrieve_hybrid_and_rerank(
                query=case.query,
                query_vector=query_vector,
                sparse_query=sparse_query,
                rerank_query=rerank_query,
                file_id=self.file_id,
                top_k=self.top_k,
                use_hybrid=self.use_hybrid,
                use_rerank=self.use_rerank,
            )

            latency_ms = (time.time() - t0) * 1000

            # Tính toán các chỉ số
            r_k = recall_at_k(hits, case.expected_pages)
            p_k = precision_at_k(hits, case.expected_pages)
            rr = reciprocal_rank(hits, case.expected_pages)
            hit = hit_at_k(hits, case.expected_pages)

            retrieved_pages = set()
            for h in hits:
                retrieved_pages.update(pages_in_range(h.get("page_start", 0), h.get("page_end", 0)))

            top_score = round(float(hits[0].get("score", 0.0)), 4) if hits else 0.0

            return CaseResult(
                case_id=case.id,
                query=case.query,
                intent=intent,
                latency_ms=round(latency_ms, 1),
                recall_at_k=round(r_k, 4),
                precision_at_k=round(p_k, 4),
                mrr=round(rr, 4),
                hit_at_k=round(hit, 4),
                top_score=top_score,
                retrieved_pages=sorted(retrieved_pages),
                retrieved_chunks=hits,
                key_fact=case.key_fact,
                status="OK",
            )
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            logger.error(f"Lỗi khi đánh giá retrieval case {case.id}: {e}")
            return CaseResult(
                case_id=case.id,
                query=case.query,
                latency_ms=round(latency_ms, 1),
                status="ERROR",
                error_message=str(e),
            )

    def evaluate_e2e_case(self, case: BenchmarkCase) -> CaseResult:
        """Kiểm thử toàn trình End-to-End (bao gồm cả Retrieval + Sinh câu trả lời LLM + Đo tính trung thực)."""
        from retrieval.pipeline import search_and_generate

        t0 = time.time()
        try:
            response = search_and_generate(
                query=case.query,
                file_id=self.file_id,
                top_k=self.top_k,
                use_hybrid=self.use_hybrid,
                use_rerank=self.use_rerank,
                embed_provider=self.embed_provider,
                llm_provider=self.llm_provider,
            )
            latency_ms = (time.time() - t0) * 1000

            hits = [asdict(s) for s in response.sources]

            r_k = recall_at_k(hits, case.expected_pages) if case.expected_pages else 1.0
            p_k = precision_at_k(hits, case.expected_pages) if case.expected_pages else 1.0
            rr = reciprocal_rank(hits, case.expected_pages) if case.expected_pages else 1.0
            hit = hit_at_k(hits, case.expected_pages) if case.expected_pages else 1.0

            retrieved_pages = set()
            for h in hits:
                retrieved_pages.update(pages_in_range(h.get("page_start", 0), h.get("page_end", 0)))

            top_score = round(float(hits[0].get("score", 0.0)), 4) if hits else 0.0

            citation_info = verify_citation_faithfulness(response.answer, hits)
            fact_cov = keyword_coverage(response.answer, case.key_fact) if case.key_fact else 1.0

            return CaseResult(
                case_id=case.id,
                query=case.query,
                intent=response.intent,
                latency_ms=round(latency_ms, 1),
                recall_at_k=round(r_k, 4),
                precision_at_k=round(p_k, 4),
                mrr=round(rr, 4),
                hit_at_k=round(hit, 4),
                top_score=top_score,
                retrieved_pages=sorted(retrieved_pages),
                retrieved_chunks=hits,
                answer=response.answer,
                key_fact=case.key_fact,
                fact_coverage=fact_cov,
                citation_info=citation_info,
                status="OK",
            )
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            logger.error(f"Lỗi khi đánh giá e2e case {case.id}: {e}")
            return CaseResult(
                case_id=case.id,
                query=case.query,
                latency_ms=round(latency_ms, 1),
                status="ERROR",
                error_message=str(e),
            )

    def run(
        self,
        dataset_path: Path | str,
        mode: Literal["retrieval", "e2e"] = "retrieval",
        output_dir: Optional[Path | str] = None,
        save_csv: bool = True,
        save_md: bool = True,
        save_json: bool = True,
    ) -> BenchmarkSummary:
        """Chạy toàn bộ benchmark trên file dataset."""
        dataset_name, cases = self.load_dataset(dataset_path)
        total = len(cases)
        logger.info(f"Bắt đầu chạy Benchmark '{dataset_name}' ({total} cases, mode={mode.upper()})...")

        results: list[CaseResult] = []
        for idx, case in enumerate(cases, 1):
            logger.info(f"[{idx}/{total}] Đang đánh giá: \"{case.query}\"")
            if mode == "retrieval":
                res = self.evaluate_retrieval_case(case)
            else:
                res = self.evaluate_e2e_case(case)
            results.append(res)

        successful = [r for r in results if r.status == "OK"]
        failed = [r for r in results if r.status != "OK"]

        # Thống kê tổng hợp
        m_recall = statistics.mean([r.recall_at_k for r in successful]) if successful else 0.0
        m_precision = statistics.mean([r.precision_at_k for r in successful]) if successful else 0.0
        m_mrr = statistics.mean([r.mrr for r in successful]) if successful else 0.0
        m_hit = statistics.mean([r.hit_at_k for r in successful]) if successful else 0.0

        latencies = [r.latency_ms for r in successful]
        m_latency = statistics.mean(latencies) if latencies else 0.0
        p95_latency = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 2 else (latencies[0] if latencies else 0.0)

        m_fact = statistics.mean([r.fact_coverage for r in successful]) if (successful and mode == "e2e") else 1.0
        faithful_count = sum(1 for r in successful if r.citation_info.get("is_faithful", True))
        cit_rate = (faithful_count / len(successful)) if (successful and mode == "e2e") else 1.0

        summary = BenchmarkSummary(
            dataset_name=dataset_name,
            file_id=self.file_id,
            mode=mode,
            top_k=self.top_k,
            use_hybrid=self.use_hybrid,
            use_rerank=self.use_rerank,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_cases=total,
            successful_cases=len(successful),
            failed_cases=len(failed),
            mean_recall_at_k=round(m_recall, 4),
            mean_precision_at_k=round(m_precision, 4),
            mean_mrr=round(m_mrr, 4),
            mean_hit_rate=round(m_hit, 4),
            mean_latency_ms=round(m_latency, 1),
            p95_latency_ms=round(p95_latency, 1),
            mean_fact_coverage=round(m_fact, 4),
            citation_faithfulness_rate=round(cit_rate, 4),
            cases=results,
        )

        if output_dir:
            out_dir = Path(output_dir)
            base_filename = f"benchmark_{mode}_{Path(dataset_path).stem}_top{self.top_k}"
            if save_json:
                summary.save_json(out_dir / f"{base_filename}.json")
            if save_csv:
                summary.save_csv(out_dir / f"{base_filename}.csv")
            if save_md:
                summary.save_markdown(out_dir / f"{base_filename}.md")

        return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chạy Benchmark chuyên nghiệp cho hệ thống RAG")
    parser.add_argument("--dataset", default="evals/fifa_law11_basic.json", help="Đường dẫn file dataset JSON")
    parser.add_argument("--file-id", default=None, help="ID của file PDF đã ingest")
    parser.add_argument("--mode", choices=["retrieval", "e2e"], default="retrieval", help="Chế độ benchmark")
    parser.add_argument("--top-k", type=int, default=5, help="Số lượng Top-K chunks")
    parser.add_argument("--no-hybrid", action="store_true", help="Tắt BM25, chỉ dùng Vector")
    parser.add_argument("--no-rerank", action="store_true", help="Tắt Reranker")
    parser.add_argument("--output-dir", default="evals", help="Thư mục xuất báo cáo")
    args = parser.parse_args()

    bench = RAGBenchmark(
        file_id=args.file_id,
        top_k=args.top_k,
        use_hybrid=not args.no_hybrid,
        use_rerank=not args.no_rerank,
    )

    summary = bench.run(
        dataset_path=args.dataset,
        mode=args.mode,
        output_dir=args.output_dir,
    )

    print("=" * 70)
    print(f"KẾT QUẢ BENCHMARK ({summary.mode.upper()}) — DATASET: {summary.dataset_name}")
    print("=" * 70)
    print(f"• Mean Recall@{summary.top_k}: {summary.mean_recall_at_k:.1%}")
    print(f"• Mean Precision@{summary.top_k}: {summary.mean_precision_at_k:.1%}")
    print(f"• Mean MRR: {summary.mean_mrr:.4f}")
    print(f"• Hit Rate@{summary.top_k}: {summary.mean_hit_rate:.1%}")
    print(f"• Mean Latency: {summary.mean_latency_ms:.1f} ms")
    if summary.mode == "e2e":
        print(f"• Fact Coverage: {summary.mean_fact_coverage:.1%}")
        print(f"• Citation Faithfulness: {summary.citation_faithfulness_rate:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
