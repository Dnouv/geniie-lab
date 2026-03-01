#!/usr/bin/env python
"""
Plot average-per-iteration metrics across topics in a ranker x metric grid.

Grid layout:
  rows = metrics (default: CumRecall, CumRecall@100, Jaccard)
  cols = rankers (default: bm25, splade)
Each cell shows per-policy bars where each bar is the mean across topics
for that iteration.

Example:
  python scripts/plot_cumrecall_avg_grid.py \\
    --log logs/final/dbpedia/oss120_bm25_full_0.out --label full \\
    --log logs/final/dbpedia/oss120_bm25_fqh_0.out --label forget_queries_half \\
    --log logs/final/dbpedia/oss120_bm25_fqkr_2.out --label forget_queries_keep_reason \\
    --log logs/final/dbpedia/oss120_splade_full_0.out --label full \\
    --log logs/final/dbpedia/oss120_splade_fqh_0.out --label forget_queries_half \\
    --log logs/final/dbpedia/oss120_splade_fqkr_0.out --label forget_queries_keep_reason \\
    --output analyses/grids/dbpedia_avg_grid.png
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot average-per-iteration metrics across topics.")
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="Path to a JSONL log file (repeat for multiple policies).",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Label for each --log (e.g., memory policy name).",
    )
    parser.add_argument(
        "--metrics",
        default="CumRecall,CumRecall@100,Jaccard",
        help="Comma-separated metric keys (columns).",
    )
    parser.add_argument(
        "--ranker-order",
        default="bm25,splade",
        help="Comma-separated ranker order for columns (default: %(default)s).",
    )
    parser.add_argument(
        "--policy-label-from",
        choices=["auto", "label", "suffix", "prefix"],
        default="auto",
        help="How to derive policy name from --label (default: %(default)s).",
    )
    parser.add_argument(
        "--policy-suffix-sep",
        default="_",
        help="Separator for suffix-based policy extraction (default: %(default)s).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=0.0,
        help="Max value for y-axis (0 = auto per subplot).",
    )
    parser.add_argument(
        "--xtick-step",
        type=int,
        default=0,
        help="Show every Nth iteration tick (0 = auto).",
    )
    parser.add_argument(
        "--xtick-fontsize",
        type=int,
        default=8,
        help="Font size for x-axis tick labels (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path.",
    )
    return parser.parse_args()


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def is_cumulative_metric(metric_key: str) -> bool:
    key = metric_key.lower()
    return key.startswith("cum") or "cum" in key


def build_series(metric_key: str, pairs: List[Tuple[int, float]], max_iter: int) -> List[float | None]:
    values = {int(it): float(val) for it, val in pairs if val is not None}
    series: List[float | None] = []
    last: float | None = None
    cumulative = is_cumulative_metric(metric_key)
    for it in range(1, max_iter + 1):
        if it in values:
            last = values[it]
        if cumulative:
            series.append(last if last is not None else 0.0)
        else:
            series.append(values.get(it))
    return series


def auto_xtick_step(iterations: List[int]) -> int:
    if not iterations:
        return 1
    max_it = max(iterations)
    if max_it <= 15:
        return 1
    if max_it <= 30:
        return 2
    if max_it <= 60:
        return 4
    if max_it <= 100:
        return 5
    return 10


def derive_policy(label: str, mode: str, sep: str) -> str:
    known = [
        "forget_queries_keep_reason",
        "forget_queries_half",
        "forget_queries",
        "full",
    ]
    if mode == "auto":
        for key in known:
            if key in label:
                return key
    if mode == "label":
        return label
    parts = label.split(sep)
    if mode == "suffix":
        return parts[0] if parts else label
    return parts[-1] if parts else label


def load_log_data(
    log_path: str,
    metrics: List[str],
) -> tuple[str, str, Dict[str, Dict[str, List[Tuple[int, float]]]]]:
    per_metric_topics: Dict[str, Dict[str, List[Tuple[int, float]]]] = {m: defaultdict(list) for m in metrics}
    ranking_counter: Dict[str, int] = defaultdict(int)
    last_query_by_topic: Dict[str, str] = {}
    prev_query_by_topic: Dict[str, str] = {}
    ranker = "unknown"
    dataset = "unknown"

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            stage = entry.get("stage")
            topic = entry.get("topic_id", "<unknown>")
            if stage in ("query", "reformulation"):
                query_text = entry.get("query", "")
                prev_query_by_topic[topic] = last_query_by_topic.get(topic, "")
                last_query_by_topic[topic] = query_text
                continue
            if stage != "ranking":
                continue

            ranking_counter[topic] += 1
            ranker = entry.get("ranker") or ranker
            dataset = entry.get("dataset") or dataset
            perf = entry.get("performance") or {}
            for metric_key in metrics:
                if metric_key == "Jaccard":
                    value = jaccard_similarity(
                        last_query_by_topic.get(topic, ""),
                        prev_query_by_topic.get(topic, ""),
                    )
                else:
                    value = perf.get(metric_key)
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue
                per_metric_topics[metric_key][topic].append((ranking_counter[topic], value_f))

    return ranker, dataset, per_metric_topics


def compute_avg_series(
    metric_key: str,
    per_topic_pairs: Dict[str, List[Tuple[int, float]]],
) -> tuple[List[int], List[float | None]]:
    if not per_topic_pairs:
        return [], []
    max_iter = max((it for pairs in per_topic_pairs.values() for it, _ in pairs), default=0)
    if max_iter <= 0:
        return [], []

    per_topic_series = []
    for pairs in per_topic_pairs.values():
        series = build_series(metric_key, pairs, max_iter)
        per_topic_series.append(series)

    avg_series: List[float | None] = []
    for idx in range(max_iter):
        vals = [s[idx] for s in per_topic_series if s[idx] is not None]
        avg_series.append((sum(vals) / len(vals)) if vals else None)

    iterations = list(range(1, max_iter + 1))
    return iterations, avg_series


def main() -> None:
    args = parse_args()
    if len(args.log) != len(args.label):
        raise SystemExit("Number of --log and --label arguments must match.")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise SystemExit("No metrics specified.")

    ranker_order = [r.strip() for r in args.ranker_order.split(",") if r.strip()]
    if not ranker_order:
        raise SystemExit("No rankers specified.")

    # ranker -> policy -> metric -> (iterations, avg_series)
    series_map: Dict[str, Dict[str, Dict[str, Tuple[List[int], List[float | None]]]]] = defaultdict(lambda: defaultdict(dict))
    datasets: set[str] = set()
    policy_order: List[str] = []

    for log_path, label in zip(args.log, args.label):
        policy = derive_policy(label, args.policy_label_from, args.policy_suffix_sep)
        ranker, dataset, per_metric_topics = load_log_data(log_path, metrics)
        datasets.add(dataset)
        if policy not in policy_order:
            policy_order.append(policy)
        for metric_key, per_topic_pairs in per_metric_topics.items():
            iterations, avg_series = compute_avg_series(metric_key, per_topic_pairs)
            series_map[ranker][policy][metric_key] = (iterations, avg_series)

    fig_rows = len(metrics)
    fig_cols = len(ranker_order)
    fig_width = max(10, fig_cols * 5.2)
    fig_height = max(6, fig_rows * 3.6)
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_width, fig_height), sharex=False, sharey=False)
    if fig_rows == 1 and fig_cols == 1:
        axes_grid = [[axes]]
    elif fig_rows == 1:
        axes_grid = [axes]
    elif fig_cols == 1:
        axes_grid = [[ax] for ax in axes]
    else:
        axes_grid = axes

    color_map = plt.colormaps.get_cmap("tab10")
    legend_handles = None
    legend_labels = None

    for r_idx, metric_key in enumerate(metrics):
        for c_idx, ranker in enumerate(ranker_order):
            ax = axes_grid[r_idx][c_idx]
            policies = series_map.get(ranker, {})
            max_iter = 0
            max_val = 0.0
            plotted = False

            for policy in policy_order:
                if policy not in policies:
                    continue
                iterations, avg_series = policies[policy].get(metric_key, ([], []))
                if not iterations:
                    continue
                max_iter = max(max_iter, max(iterations))
                for val in avg_series:
                    if val is not None and math.isfinite(val):
                        max_val = max(max_val, val)

            if max_iter == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{ranker} - {metric_key}")
                continue

            base_positions = list(range(max_iter))
            bar_width = 0.8 / max(1, len(policy_order))

            for p_idx, policy in enumerate(policy_order):
                if policy not in policies:
                    continue
                iterations, avg_series = policies[policy].get(metric_key, ([], []))
                if not iterations:
                    continue
                plotted = True
                heights: List[float] = []
                for idx in range(max_iter):
                    if idx < len(avg_series) and avg_series[idx] is not None:
                        heights.append(float(avg_series[idx]))
                    else:
                        heights.append(float("nan"))
                positions = [pos - 0.4 + bar_width / 2 + p_idx * bar_width for pos in base_positions]
                color = color_map(p_idx % color_map.N)
                ax.bar(
                    positions,
                    heights,
                    width=bar_width,
                    color=color,
                    edgecolor="black",
                    linewidth=0.3,
                    label=policy,
                )

            if not plotted:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{ranker} - {metric_key}")
                continue

            step = args.xtick_step or auto_xtick_step(list(range(1, max_iter + 1)))
            tick_positions = [pos for pos in base_positions[::step]]
            tick_labels = [str(i + 1) for i in base_positions[::step]]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.tick_params(axis="x", labelsize=args.xtick_fontsize)
            ax.grid(alpha=0.3)
            ax.set_title(f"{ranker} - {metric_key}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric_key)
            if args.y_max > 0:
                ax.set_ylim(0, args.y_max)
            else:
                ax.set_ylim(0, (max_val * 1.08) if max_val > 0 else 1.0)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        legend_cols = max(1, min(4, len(legend_labels)))
        fig.legend(
            legend_handles,
            legend_labels,
            bbox_to_anchor=(0.5, -0.01),
            loc="upper center",
            ncol=legend_cols,
            fontsize="small",
            title="Policy",
        )

    dataset_title = ", ".join(sorted(d for d in datasets if d and d != "unknown"))
    if dataset_title:
        fig.suptitle(f"Average per-iteration metrics across topics\nDataset(s): {dataset_title}", y=1.02)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid to {output_path.resolve()}")


if __name__ == "__main__":
    main()
