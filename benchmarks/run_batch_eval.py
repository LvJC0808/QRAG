from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_json_from_stdout(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        raise RuntimeError("No summary output returned")

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    # Some upstream components may print extra logs before final JSON.
    decoder = json.JSONDecoder()
    last_dict: dict[str, Any] | None = None
    idx = 0
    while idx < len(text):
        if text[idx] != "{":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            last_dict = obj
        idx = end

    if last_dict is None:
        raise RuntimeError("Cannot parse JSON summary from benchmark output")
    return last_dict


def find_pdf_for_sample(sample_file: Path, pdf_dir: Path) -> Path:
    stem = sample_file.stem
    direct = pdf_dir / f"{stem}.pdf"
    if direct.exists():
        return direct

    # Fallback: case-insensitive lookup.
    stem_lower = stem.lower()
    for candidate in pdf_dir.glob("*.pdf"):
        if candidate.stem.lower() == stem_lower:
            return candidate

    raise FileNotFoundError(
        f"Cannot find matching PDF for sample '{sample_file.name}' under {pdf_dir}"
    )


def run_single_doc(
    sample_file: Path,
    pdf_file: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cmd = [
        args.python_executable or sys.executable,
        "-m",
        "QRAG.benchmarks.benchmark_runner",
        "--pdf",
        str(pdf_file),
        "--samples",
        str(sample_file),
        "--output-dir",
        str(output_dir),
        "--recall-top-k",
        str(args.recall_top_k),
        "--rerank-top-n",
        str(args.rerank_top_n),
        "--evidence-top-m",
        str(args.evidence_top_m),
        "--image-max-pixels",
        str(args.image_max_pixels),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
    ]

    if args.max_samples_per_doc > 0:
        cmd.extend(["--max-samples", str(args.max_samples_per_doc)])

    if args.disable_revision:
        cmd.append("--disable-revision")

    if not args.force_table:
        cmd.append("--no-force-table")

    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed for {sample_file.name}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    summary = parse_json_from_stdout(proc.stdout)
    summary["sample_file"] = str(sample_file)
    summary["pdf_file"] = str(pdf_file)
    summary["batch_wall_time_s"] = elapsed
    return summary


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_docs": 0,
            "num_samples": 0,
            "avg_overall_score": 0.0,
            "avg_gold_page_hit_rate": None,
            "avg_citation_validity": 0.0,
            "avg_total_latency_s": 0.0,
            "doc_runs": [],
        }

    def mean_field(name: str) -> float:
        vals = [float(item.get(name, 0.0)) for item in rows]
        return sum(vals) / max(1, len(vals))

    def weighted_mean(name: str) -> float:
        weighted_sum = 0.0
        total_weight = 0
        for item in rows:
            weight = int(item.get("num_samples", 0))
            value = item.get(name)
            if weight <= 0 or value is None:
                continue
            weighted_sum += float(value) * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight else 0.0

    page_hits = [item.get("gold_page_hit_rate") for item in rows if item.get("gold_page_hit_rate") is not None]
    avg_page_hit = (sum(page_hits) / len(page_hits)) if page_hits else None
    weighted_page_hit = weighted_mean("gold_page_hit_rate") if page_hits else None

    return {
        "num_docs": len(rows),
        "num_samples": int(sum(int(item.get("num_samples", 0)) for item in rows)),
        "avg_overall_score": weighted_mean("avg_overall_score"),
        "avg_gold_page_hit_rate": avg_page_hit,
        "weighted_gold_page_hit_rate": weighted_page_hit,
        "avg_citation_validity": weighted_mean("avg_citation_validity"),
        "avg_total_latency_s": weighted_mean("avg_total_latency_s"),
        "macro_avg_overall_score": mean_field("avg_overall_score"),
        "macro_avg_citation_validity": mean_field("avg_citation_validity"),
        "macro_avg_total_latency_s": mean_field("avg_total_latency_s"),
        "doc_runs": [
            {
                "doc": Path(item["sample_file"]).stem,
                "num_samples": item.get("num_samples"),
                "avg_overall_score": item.get("avg_overall_score"),
                "gold_page_hit_rate": item.get("gold_page_hit_rate"),
                "avg_citation_validity": item.get("avg_citation_validity"),
                "output_dir": item.get("output_dir"),
                "run_id": item.get("run_id"),
            }
            for item in rows
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QRAG benchmark in batch mode across multiple sample files")
    parser.add_argument("--samples-dir", required=True, help="Directory containing per-document *.jsonl sample files")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="/root/shared-nvme/QRAG/data/benchmarks", help="Benchmark output root")
    parser.add_argument("--python-executable", default="", help="Python executable used to launch benchmark_runner")
    parser.add_argument("--glob", default="*.jsonl", help="Sample file glob pattern")
    parser.add_argument("--max-docs", type=int, default=0, help="0 means all")
    parser.add_argument("--max-samples-per-doc", type=int, default=0, help="0 means all")

    parser.add_argument("--recall-top-k", type=int, default=24)
    parser.add_argument("--rerank-top-n", type=int, default=8)
    parser.add_argument("--evidence-top-m", type=int, default=4)
    parser.add_argument("--image-max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--disable-revision", action="store_true")
    parser.add_argument("--force-table", dest="force_table", action="store_true")
    parser.add_argument("--no-force-table", dest="force_table", action="store_false")
    parser.set_defaults(force_table=True)

    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only print matched sample/pdf pairs without running")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    samples_dir = Path(args.samples_dir).expanduser().resolve()
    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    sample_files = sorted(samples_dir.glob(args.glob))
    sample_files = [p for p in sample_files if p.is_file()]
    if not sample_files:
        raise FileNotFoundError(f"No sample files found in {samples_dir} with pattern {args.glob}")

    if args.max_docs > 0:
        sample_files = sample_files[: args.max_docs]

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for sample_file in sample_files:
        try:
            pdf_file = find_pdf_for_sample(sample_file, pdf_dir)
            if args.dry_run:
                print(f"[PAIR] {sample_file.name} <- {pdf_file.name}")
                continue
            print(f"[RUN] {sample_file.name} <- {pdf_file.name}")
            summary = run_single_doc(sample_file, pdf_file, output_dir, args)
            rows.append(summary)
            print(
                f"[OK] {sample_file.stem}: overall={summary.get('avg_overall_score', 0):.2f}, "
                f"page_hit={summary.get('gold_page_hit_rate')}"
            )
        except Exception as exc:  # noqa: BLE001
            err = {"sample_file": str(sample_file), "error": str(exc)}
            errors.append(err)
            print(f"[ERR] {sample_file.name}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise

    if args.dry_run:
        print(json.dumps({"matched_docs": len(sample_files)}, ensure_ascii=False, indent=2))
        return

    aggregate = aggregate_results(rows)
    aggregate["errors"] = errors
    aggregate["args"] = vars(args)
    aggregate["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    batch_path = output_dir / f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"batch_summary": str(batch_path), **aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
