#!/usr/bin/env python
"""
Batch regenerate bar-style plots for multiple experiment logs.

Example:
    python scripts/batch_bar_plots.py \\
        --logs \"logs/*_L4M.out\" \\
        --output-root analyses \\
        --topicset beir/dbpedia-entity/dev

This simply wraps plot_log_metrics so you can re-render all your existing
analyses directories with bar charts in one go.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

from plot_log_metrics import (
    build_topic_labels,
    ensure_output_dir,
    load_records,
    plot_combined_metrics,
    plot_jaccard_similarity,
    plot_metric_by_topic,
    plot_query_length_trajectories,
    plot_query_length_vs_metric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch render bar charts for logs.")
    parser.add_argument(
        "--logs",
        action="append",
        required=True,
        help="Glob pattern(s) for JSONL logs (e.g., logs/*_L4M.out).",
    )
    parser.add_argument(
        "--output-root",
        default="analyses",
        help="Root directory to place per-log plot folders (default: %(default)s).",
    )
    parser.add_argument(
        "--topicset",
        help="Optional ir_datasets topicset to label topics with titles.",
    )
    return parser.parse_args()


def expand_logs(patterns: List[str]) -> List[str]:
    paths: List[str] = []
    for pat in patterns:
        for match in glob.glob(pat):
            if Path(match).is_file():
                paths.append(str(Path(match)))
    return sorted(set(paths))


def main() -> None:
    args = parse_args()
    logs = expand_logs(args.logs)
    if not logs:
        print("[ERROR] No log files matched the provided patterns.")
        return

    for log_path in logs:
        stem = Path(log_path).stem
        out_dir = ensure_output_dir(Path(args.output_root) / f"{stem}_bar")
        records = load_records([log_path])
        ranking_records = records["ranking"]
        query_records = records["queries"]
        model_names = [rec.get("model") for rec in ranking_records if rec.get("model")]
        if not ranking_records:
            print(f"[WARN] No ranking entries in {log_path}, skipping.")
            continue
        topic_labels = build_topic_labels([rec["topic_id"] for rec in ranking_records], args.topicset)

        ndcg_path = out_dir / "ndcg_by_topic.png"
        plot_metric_by_topic(ranking_records, "metric_ndcg", ndcg_path, ylabel="nDCG@10", topic_labels=topic_labels, model_names=model_names)

        rr_path = out_dir / "rr_by_topic.png"
        plot_metric_by_topic(ranking_records, "metric_rr", rr_path, ylabel="Reciprocal Rank @10", topic_labels=topic_labels, model_names=model_names)

        recall_path = out_dir / "recall100_by_topic.png"
        plot_metric_by_topic(ranking_records, "metric_recall100", recall_path, ylabel="Recall@100", topic_labels=topic_labels, model_names=model_names)

        cum_recall_path = out_dir / "cum_recall_by_topic.png"
        plot_metric_by_topic(ranking_records, "metric_cum_recall", cum_recall_path, ylabel="Cumulative Recall (LLM-judged)", topic_labels=topic_labels, model_names=model_names)

        scatter_path = out_dir / "query_length_vs_ndcg.png"
        plot_query_length_vs_metric(ranking_records, "metric_ndcg", scatter_path, topic_labels, model_names)

        query_traj_path = out_dir / "query_length_traj.png"
        plot_query_length_trajectories(query_records, query_traj_path, topic_labels)

        combined_path = out_dir / "combined_rr_ndcg.png"
        plot_combined_metrics(ranking_records, combined_path, model_names)

        jaccard_path = out_dir / "query_jaccard_similarity.png"
        plot_jaccard_similarity(query_records, jaccard_path, model_names)

        print(f"[INFO] Wrote bar plots to {out_dir}")


if __name__ == "__main__":
    main()
