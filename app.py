from __future__ import annotations

from pathlib import Path
from typing import Any, List

import gradio as gr
import pandas as pd

from .config import DATA_DIR
from .core.tuner import JudgeTuner
from .pipeline_factory import create_pipeline
from .schemas import GenerationConfig, RetrievalConfig


pipeline = create_pipeline()
tuner = JudgeTuner(pipeline=pipeline, output_dir=Path(DATA_DIR) / "tuning")


def _format_state() -> str:
    state = pipeline.state
    if not state.ready:
        return "System not ready. Upload and index a PDF first."
    return (
        f"Ready\n"
        f"doc_id: {state.doc_id}\n"
        f"index_path: {state.index_path}\n"
        f"page_count: {state.page_count}"
    )


def _format_judge_markdown(bundle_dict: dict[str, Any]) -> str:
    judge = bundle_dict["judge_result"]
    dims = judge["dimension_scores"]
    lines = [
        "## Judge Report",
        f"- Overall: **{judge['overall_score']:.1f}**",
        f"- Verdict: **{judge['verdict']}**",
        f"- Relevance: {dims['relevance']:.1f}",
        f"- Groundedness: {dims['groundedness']:.1f}",
        f"- Completeness: {dims['completeness']:.1f}",
        f"- Numeric Consistency: {dims['numeric_consistency']:.1f}",
        f"- Citation Validity: {dims['citation_validity']:.1f}",
        "\n### Major Issues",
    ]
    if judge["major_issues"]:
        lines.extend([f"- {item}" for item in judge["major_issues"]])
    else:
        lines.append("- None")

    lines.append("\n### Actionable Feedback")
    if judge["actionable_feedback"]:
        lines.extend([f"- {item}" for item in judge["actionable_feedback"]])
    else:
        lines.append("- None")

    timings = bundle_dict["timings"]
    lines.append("\n### Stage Timings (s)")
    for key, value in timings.items():
        lines.append(f"- {key}: {value:.2f}")

    return "\n".join(lines)


def ingest_pdf(file_obj) -> str:
    if file_obj is None:
        return "Please upload a PDF file."

    try:
        pipeline.ingest_pdf(file_obj.name)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to ingest PDF: {exc}"

    return _format_state()


def ask_question(
    query: str,
    recall_top_k: int,
    rerank_top_n: int,
    evidence_top_m: int,
    image_max_pixels: int,
    max_new_tokens: int,
    temperature: float,
    force_table: bool,
    enable_revision: bool,
):
    if not query.strip():
        return "Please input a question.", "", [], {}

    try:
        retrieval_cfg = RetrievalConfig(
            recall_top_k=recall_top_k,
            rerank_top_n=rerank_top_n,
            evidence_top_m=evidence_top_m,
            image_max_pixels=image_max_pixels,
        )
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            force_table=force_table,
        )
        bundle = pipeline.run(
            query=query,
            retrieval_config=retrieval_cfg,
            generation_config=gen_cfg,
            enable_revision=enable_revision,
        )
        gallery = [
            (
                item.image_path,
                f"p{item.page_num} | retrieval={item.retrieval_score:.3f} | rerank={item.rerank_score:.3f}",
            )
            for item in bundle.evidence_items
        ]
        bundle_dict = bundle.model_dump()
        return bundle.final_answer, bundle.draft_answer, gallery, bundle_dict
    except Exception as exc:  # noqa: BLE001
        return f"Pipeline error: {exc}", "", [], {}


def run_tuning(queries_text: str):
    query_lines = [line.strip() for line in (queries_text or "").splitlines() if line.strip()]
    if not query_lines:
        return pd.DataFrame(), "Please input at least one evaluation query."

    try:
        summary = tuner.run_grid(query_lines)
        rows = []
        for run in summary.runs:
            rows.append(
                {
                    "run_id": run.run_id,
                    "objective": round(run.objective_score, 3),
                    "overall": round(run.mean_overall_score, 3),
                    "recall_top_k": run.retrieval_config.recall_top_k,
                    "rerank_top_n": run.retrieval_config.rerank_top_n,
                    "evidence_top_m": run.retrieval_config.evidence_top_m,
                    "image_max_pixels": run.retrieval_config.image_max_pixels,
                }
            )

        df = pd.DataFrame(rows).sort_values("objective", ascending=False)
        best = summary.best_config
        best_md = (
            "## Tuning Complete\n"
            f"- Best run: `{summary.best_run_id}`\n"
            f"- Best objective: **{summary.best_objective_score:.3f}**\n"
            f"- Best config: recall_top_k={best.recall_top_k}, "
            f"rerank_top_n={best.rerank_top_n}, evidence_top_m={best.evidence_top_m}, "
            f"image_max_pixels={best.image_max_pixels}"
        )
        return df, best_md
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), f"Tuning failed: {exc}"


def build_ui() -> gr.Blocks:
    default_r = pipeline.default_retrieval_config()
    default_g = pipeline.default_generation_config()

    with gr.Blocks(title="QRAG - Local Multimodal PDF QA") as demo:
        gr.Markdown("# QRAG: Local Multimodal PDF QA (Qwen3-VL)")
        gr.Markdown("Single-PDF multimodal RAG with Embedding+Reranker, LLM Judge, and tuning loop.")

        last_bundle_state = gr.State({})

        with gr.Tab("文档处理"):
            file_input = gr.File(label="上传 PDF", file_types=[".pdf"])
            ingest_btn = gr.Button("构建索引", variant="primary")
            status_box = gr.Textbox(label="系统状态", value=_format_state(), lines=6)
            ingest_btn.click(fn=ingest_pdf, inputs=[file_input], outputs=[status_box])

        with gr.Tab("问答"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_box = gr.Textbox(label="问题", lines=3, placeholder="例如：提取图4(a)的关键数值并给出证据页")
                    ask_btn = gr.Button("运行问答", variant="primary")

                    final_output = gr.Markdown(label="最终答案")
                    draft_output = gr.Textbox(label="初稿（Debug）", lines=10)

                with gr.Column(scale=1):
                    recall_slider = gr.Slider(1, 100, value=default_r.recall_top_k, step=1, label="recall_top_k")
                    rerank_slider = gr.Slider(1, 32, value=default_r.rerank_top_n, step=1, label="rerank_top_n")
                    evidence_slider = gr.Slider(1, 10, value=default_r.evidence_top_m, step=1, label="evidence_top_m")
                    pixels_slider = gr.Slider(256 * 256, 1600 * 1600, value=default_r.image_max_pixels, step=4096, label="image_max_pixels")
                    token_slider = gr.Slider(128, 4096, value=default_g.max_new_tokens, step=64, label="max_new_tokens")
                    temp_slider = gr.Slider(0.0, 1.0, value=default_g.temperature, step=0.05, label="temperature")
                    force_table_cb = gr.Checkbox(value=default_g.force_table, label="强制表格输出")
                    revision_cb = gr.Checkbox(value=True, label="启用 Judge 二次修订")

            evidence_gallery = gr.Gallery(label="证据页", columns=2, object_fit="contain", height=360)

            ask_btn.click(
                fn=ask_question,
                inputs=[
                    query_box,
                    recall_slider,
                    rerank_slider,
                    evidence_slider,
                    pixels_slider,
                    token_slider,
                    temp_slider,
                    force_table_cb,
                    revision_cb,
                ], 
                outputs=[final_output, draft_output, evidence_gallery, last_bundle_state],
            )

        with gr.Tab("Judge 报告"):
            judge_box = gr.Markdown("No run yet.")

            def refresh_judge(bundle_dict: dict):
                if not bundle_dict:
                    return "No run yet."
                return _format_judge_markdown(bundle_dict)

            refresh_btn = gr.Button("刷新最近一次报告")
            refresh_btn.click(fn=refresh_judge, inputs=[last_bundle_state], outputs=[judge_box])

        with gr.Tab("调优实验室"):
            tune_queries = gr.Textbox(
                label="评测问题（每行一个）",
                lines=8,
                placeholder="示例:\n论文提出的核心方法是什么？\n图2展示了哪些模块？\n提取表1中的关键指标。",
            )
            tune_btn = gr.Button("运行网格调优", variant="primary")
            tune_table = gr.Dataframe(label="调优结果", interactive=False)
            tune_summary = gr.Markdown(label="最佳配置")

            tune_btn.click(fn=run_tuning, inputs=[tune_queries], outputs=[tune_table, tune_summary])

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue().launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()
