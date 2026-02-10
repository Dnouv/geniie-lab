#!/usr/bin/env python
"""
Create a topic x metric grid directly from JSONL experiment logs.

Grid layout:
  columns = topics (left to right)
  rows = metrics (top to bottom)
Each cell shows per-policy segmented bars (iteration 1..N within each policy).

Example:
  python scripts/plot_cumrecall_grid.py \\
    --log logs/final/dbpedia/oss120_bm25_full_0.out --label full \\
    --log logs/final/dbpedia/oss120_bm25_fqh_0.out --label forget_queries_half \\
    --log logs/final/dbpedia/oss120_bm25_fqkr_2.out --label forget_queries_keep_reason \\
    --topicset beir/dbpedia-entity/test \\
    --topics-max 5 \\
    --metrics "CumRecall,CumRecall@100,Jaccard" \\
    --output analyses/grids/dbpedia_bm25_grid.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

try:
    import ir_datasets
except ImportError:  # pragma: no cover
    ir_datasets = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a topic x metric grid from experiment logs.")
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
        help="Comma-separated metric keys (rows).",
    )
    parser.add_argument(
        "--topicset",
        help="Optional ir_datasets topicset to annotate topics.",
    )
    parser.add_argument(
        "--topics",
        default="",
        help="Comma-separated topic IDs to include (left to right).",
    )
    parser.add_argument(
        "--topics-max",
        type=int,
        default=5,
        help="Max topics to include when --topics is empty (default: %(default)s).",
    )
    parser.add_argument(
        "--topics-sort",
        choices=["id", "first_seen"],
        default="id",
        help="How to order topics when --topics is empty (default: %(default)s).",
    )
    parser.add_argument(
        "--cell-width",
        type=float,
        default=5.0,
        help="Width of each grid cell in inches (default: %(default)s).",
    )
    parser.add_argument(
        "--cell-height",
        type=float,
        default=2.8,
        help="Height of each grid cell in inches (default: %(default)s).",
    )
    parser.add_argument(
        "--segment-gap",
        type=float,
        default=1.0,
        help="Extra gap (in x units) between policy segments (default: %(default)s).",
    )
    parser.add_argument(
        "--xtick-step",
        type=int,
        default=2,
        help="Show every Nth iteration tick (0 = auto).",
    )
    parser.add_argument(
        "--xtick-fontsize",
        type=int,
        default=7,
        help="Font size for x-axis ticks (default: %(default)s).",
    )
    parser.add_argument(
        "--xtick-mode",
        choices=["all", "segment", "none"],
        default="all",
        help="X tick mode: all iterations, segment endpoints, or none (default: %(default)s).",
    )
    parser.add_argument(
        "--bar-width",
        type=float,
        default=0.72,
        help="Bar width for each iteration (default: %(default)s).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=0.0,
        help="Max value for y-axis (0 = auto per metric row).",
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


def build_series(metric_key: str, pairs: List[Tuple[int, float]], max_iter: int) -> List[float]:
    values = {int(it): float(val) for it, val in pairs if val is not None}
    series: List[float] = []
    last = None
    cumulative = is_cumulative_metric(metric_key)
    for it in range(1, max_iter + 1):
        if it in values:
            last = values[it]
        if cumulative:
            series.append(last if last is not None else 0.0)
        else:
            series.append(values.get(it, 0.0))
    return series


def build_topic_labels(topic_ids: Iterable[str], topicset: str | None) -> Dict[str, str]:
    unique_ids = set(topic_ids)
    if not topicset or ir_datasets is None:
        return {tid: tid for tid in unique_ids}
    try:
        dataset = ir_datasets.load(topicset)
    except Exception:
        return {tid: tid for tid in unique_ids}

    topic_map = {}
    for topic in dataset.queries_iter():
        title = getattr(topic, "title", None) or getattr(topic, "text", None)
        topic_map[topic.query_id] = title

    labels = {}
    for tid in unique_ids:
        title = topic_map.get(tid)
        labels[tid] = f"{tid} - {title}" if title else tid
    return labels


def parse_logs(
    log_paths: List[str],
    labels: List[str],
    metrics: List[str],
) -> tuple[Dict[str, Dict[str, List[dict]]], List[str], List[str], List[str], bool]:
    data: Dict[str, Dict[str, List[dict]]] = {m: {label: [] for label in labels} for m in metrics}
    datasets: set[str] = set()
    rankers: set[str] = set()
    topics_seen: List[str] = []
    saw_query = False

    for log_path, label in zip(log_paths, labels):
        ranking_counter: Dict[str, int] = defaultdict(int)
        last_query_by_topic: Dict[str, str] = {}
        prev_query_by_topic: Dict[str, str] = {}

        with open(log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                stage = entry.get("stage")
                topic = entry.get("topic_id", "<unknown>")
                if stage in ("query", "reformulation"):
                    saw_query = True
                    query_text = entry.get("query", "")
                    prev_query_by_topic[topic] = last_query_by_topic.get(topic, "")
                    last_query_by_topic[topic] = query_text
                    continue
                if stage != "ranking":
                    continue

                ranking_counter[topic] += 1
                if topic not in topics_seen:
                    topics_seen.append(topic)

                dataset = entry.get("dataset")
                ranker = entry.get("ranker")
                if dataset:
                    datasets.add(dataset)
                if ranker:
                    rankers.add(ranker)

                perf = entry.get("performance") or {}
                for metric_key in metrics:
                    if metric_key == "Jaccard":
                        value = jaccard_similarity(
                            last_query_by_topic.get(topic, ""),
                            prev_query_by_topic.get(topic, ""),
                        )
                    else:
                        value = perf.get(metric_key)
                    data[metric_key][label].append(
                        {
                            "topic_id": topic,
                            "iteration": ranking_counter[topic],
                            "value": value,
                        }
                    )

    return data, sorted(datasets), sorted(rankers), topics_seen, saw_query


def select_topics(
    explicit: List[str],
    topics_seen: List[str],
    max_topics: int,
    sort_mode: str,
) -> List[str]:
    if explicit:
        return explicit
    if sort_mode == "first_seen":
        return topics_seen[:max_topics] if max_topics > 0 else topics_seen
    topics_sorted = sorted(topics_seen)
    return topics_sorted[:max_topics] if max_topics > 0 else topics_sorted


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


def main() -> None:
    args = parse_args()
    if len(args.log) != len(args.label):
        raise SystemExit("Number of --log and --label arguments must match.")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise SystemExit("No metrics specified.")

    data, datasets, rankers, topics_seen, saw_query = parse_logs(args.log, args.label, metrics)
    if "Jaccard" in metrics and not saw_query:
        print("[WARN] No query stages found; skipping Jaccard.", file=sys.stderr)
        metrics = [m for m in metrics if m != "Jaccard"]

    if not topics_seen:
        raise SystemExit("No ranking records found in provided logs.")

    explicit_topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    topics = select_topics(explicit_topics, topics_seen, args.topics_max, args.topics_sort)
    topic_labels = build_topic_labels(topics, args.topicset)

    num_rows = len(metrics)
    num_cols = len(topics)
    if num_rows == 0 or num_cols == 0:
        raise SystemExit("No topics or metrics available for plotting.")

    fig_width = max(8, num_cols * args.cell_width)
    fig_height = max(6, num_rows * args.cell_height)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), sharey="row")
    if num_rows == 1 and num_cols == 1:
        axes_grid = [[axes]]
    elif num_rows == 1:
        axes_grid = [axes]
    elif num_cols == 1:
        axes_grid = [[ax] for ax in axes]
    else:
        axes_grid = axes

    policy_order = list(args.label)
    color_map = plt.colormaps.get_cmap("tab10")

    row_max: Dict[str, float] = {}
    for metric_key in metrics:
        max_val = 0.0
        for label in policy_order:
            for rec in data[metric_key][label]:
                if rec["topic_id"] not in topics:
                    continue
                value = rec.get("value")
                if value is None:
                    continue
                try:
                    max_val = max(max_val, float(value))
                except (TypeError, ValueError):
                    pass
        row_max[metric_key] = max_val

    for r_idx, metric_key in enumerate(metrics):
        for c_idx, topic_id in enumerate(topics):
            ax = axes_grid[r_idx][c_idx]
            per_policy_pairs: Dict[str, List[Tuple[int, float]]] = {}
            for label in policy_order:
                pairs = [
                    (rec["iteration"], rec["value"])
                    for rec in data[metric_key][label]
                    if rec["topic_id"] == topic_id and rec["value"] is not None
                ]
                pairs.sort(key=lambda x: x[0])
                per_policy_pairs[label] = pairs

            segment_len = 0
            for pairs in per_policy_pairs.values():
                if pairs:
                    segment_len = max(segment_len, max(it for it, _ in pairs))
            if segment_len == 0:
                ax.axis("off")
                continue

            for p_idx, label in enumerate(policy_order):
                pairs = per_policy_pairs[label]
                iterations = list(range(1, segment_len + 1))
                segment_stride = segment_len + args.segment_gap
                xs = [p_idx * segment_stride + it for it in iterations]
                ys = build_series(metric_key, pairs, segment_len)
                ax.bar(
                    xs,
                    ys,
                    width=args.bar_width,
                    color=color_map(p_idx % color_map.N),
                    edgecolor="black",
                    linewidth=0.2,
                )
                if p_idx > 0:
                    ax.axvline(p_idx * segment_stride + 0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

            if args.y_max > 0:
                ax.set_ylim(0, args.y_max)
            else:
                max_val = row_max.get(metric_key, 0.0)
                ax.set_ylim(0, (max_val * 1.08) if max_val > 0 else 1.0)

            if r_idx == num_rows - 1:
                tick_positions: List[float] = []
                tick_labels: List[str] = []
                if args.xtick_mode == "none":
                    ax.set_xticks([])
                elif args.xtick_mode == "segment":
                    for p_idx in range(len(policy_order)):
                        start_pos = p_idx * segment_stride + 1
                        end_pos = p_idx * segment_stride + segment_len
                        tick_positions.append(start_pos)
                        tick_labels.append("1")
                        if segment_len > 1:
                            tick_positions.append(end_pos)
                            tick_labels.append(str(segment_len))
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels, fontsize=args.xtick_fontsize)
                else:
                    tick_step = args.xtick_step or auto_xtick_step(list(range(1, segment_len + 1)))
                    for p_idx in range(len(policy_order)):
                        for it in range(1, segment_len + 1, tick_step):
                            tick_positions.append(p_idx * segment_stride + it)
                            tick_labels.append(str(it))
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels, fontsize=args.xtick_fontsize)
            else:
                ax.set_xticks([])

            if c_idx == 0:
                ax.set_ylabel(metric_key)
            if r_idx == 0:
                ax.set_title(topic_labels.get(topic_id, topic_id), fontsize=9)

            ax.grid(axis="y", alpha=0.2)
            max_x = (len(policy_order) - 1) * segment_stride + segment_len + 0.5
            ax.set_xlim(0.5, max_x)

    handles = [
        plt.Line2D([0], [0], color=color_map(i % color_map.N), lw=6)
        for i in range(len(policy_order))
    ]
    fig.legend(handles, policy_order, loc="upper right", bbox_to_anchor=(1.01, 1.0), fontsize="small", title="Policy")

    dataset_label = ", ".join(datasets) if datasets else "unknown dataset"
    ranker_label = ", ".join(rankers) if rankers else "unknown ranker"
    fig.suptitle(f"Dataset: {dataset_label} | Ranker: {ranker_label}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.95, 0.95))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved grid to {output_path.resolve()}")


if __name__ == "__main__":
    main()
