#!/usr/bin/env python
"""
Plot cumulative recall across multiple memory policies (one subplot per policy).

Example:
  python scripts/plot_cumrecall_policies.py \\
    --log logs/run_full.out --label full \\
    --log logs/run_fq.out --label forget_queries \\
    --log logs/run_fqkr.out --label forget_queries_keep_reason \\
    --topicset beir/dbpedia-entity/test \\
    --output analyses/cumrecall_policies.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

try:
    import ir_datasets
except ImportError:  # pragma: no cover
    ir_datasets = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cumulative recall across memory policies.")
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
        "--metric-key",
        default="CumRecall",
        help="Performance key to plot (default: %(default)s).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=0.2,
        help="Max value for y-axis (default: %(default)s).",
    )
    parser.add_argument(
        "--xtick-step",
        type=int,
        default=0,
        help="Show every Nth tick on x-axis (0 = auto).",
    )
    parser.add_argument(
        "--xtick-rotate",
        type=int,
        default=45,
        help="Rotation angle for x-axis tick labels (default: %(default)s).",
    )
    parser.add_argument(
        "--xtick-fontsize",
        type=int,
        default=8,
        help="Font size for x-axis tick labels (default: %(default)s).",
    )
    parser.add_argument(
        "--topicset",
        help="Optional ir_datasets topicset to annotate topics.",
    )
    parser.add_argument(
        "--output",
        default="cumrecall_policies.png",
        help="Output image path (default: %(default)s).",
    )
    return parser.parse_args()


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


def load_rel_counts(topicset: str | None) -> Dict[str, int]:
    if not topicset or ir_datasets is None:
        return {}
    try:
        ds = ir_datasets.load(topicset)
    except Exception:
        return {}
    rels: Dict[str, int] = defaultdict(int)
    try:
        for q in ds.qrels_iter():
            if getattr(q, "relevance", 0) and q.relevance > 0:
                rels[q.query_id] += 1
    except Exception:
        return {}
    return dict(rels)


def build_topic_palette(topics: Iterable[str]) -> Dict[str, tuple]:
    cmap = plt.colormaps.get_cmap("tab20")
    palette = {}
    for idx, topic in enumerate(sorted(set(topics))):
        palette[topic] = cmap(idx % cmap.N)
    return palette


def load_ranking_records(log_path: str, metric_key: str) -> List[dict]:
    ranking_counter: Dict[str, int] = defaultdict(int)
    records: List[dict] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("stage") != "ranking":
                continue
            topic = entry.get("topic_id", "<unknown>")
            ranking_counter[topic] += 1
            perf = entry.get("performance") or {}
            value = perf.get(metric_key)
            records.append(
                {
                    "topic_id": topic,
                    "iteration": ranking_counter[topic],
                    "value": value,
                }
            )
    return records


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

    per_policy_records: Dict[str, List[dict]] = {}
    for log_path, label in zip(args.log, args.label):
        per_policy_records[label] = load_ranking_records(log_path, args.metric_key)

    all_topics = sorted({rec["topic_id"] for records in per_policy_records.values() for rec in records})
    if not all_topics:
        raise SystemExit("No ranking records found for the given logs.")

    topic_labels = build_topic_labels(all_topics, args.topicset)
    rel_counts = load_rel_counts(args.topicset)
    if rel_counts:
        topic_labels = {tid: f"{label} (rels={rel_counts.get(tid, 0)})" for tid, label in topic_labels.items()}

    palette = build_topic_palette(all_topics)

    num_policies = len(per_policy_records)
    max_iters_overall = 0
    for records in per_policy_records.values():
        iters = {rec["iteration"] for rec in records}
        if iters:
            max_iters_overall = max(max_iters_overall, max(iters))
    max_label_len = max((len(label) for label in topic_labels.values()), default=0)
    extra_legend_width = max(0.0, (max_label_len - 40) * 0.12)
    fig_width = max(10, 3.2 * num_policies, max_iters_overall * 0.28 * num_policies) + extra_legend_width
    fig, axes = plt.subplots(1, num_policies, figsize=(fig_width, 4.8), sharey=True)
    if num_policies == 1:
        axes = [axes]

    for ax, (label, records) in zip(axes, per_policy_records.items()):
        per_topic: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for rec in records:
            if rec["value"] is None:
                continue
            per_topic[rec["topic_id"]].append((rec["iteration"], rec["value"]))

        iterations = sorted({it for pairs in per_topic.values() for it, _ in pairs})
        if not iterations:
            ax.set_title(label)
            ax.set_xlabel("Ranking iteration")
            continue

        base_positions = list(range(len(iterations)))
        bar_width = 0.8 / max(1, len(all_topics))
        center_offset = (len(all_topics) - 1) * bar_width / 2

        for idx, topic in enumerate(all_topics):
            data = {it: val for it, val in per_topic.get(topic, [])}
            positions = [pos - center_offset + idx * bar_width for pos in base_positions]
            heights = [data.get(it, 0) for it in iterations]
            ax.bar(
                positions,
                heights,
                width=bar_width,
                color=palette[topic],
                edgecolor="black",
                linewidth=0.4,
                label=topic_labels.get(topic, topic),
            )

        step = args.xtick_step or auto_xtick_step(iterations)
        tick_positions = base_positions[::step]
        tick_labels = [iterations[i] for i in range(0, len(iterations), step)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=args.xtick_rotate, ha="right")
        ax.tick_params(axis="x", labelsize=args.xtick_fontsize)
        ax.set_xlabel("Ranking iteration")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, args.y_max)

    axes[0].set_ylabel(args.metric_key)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")
        fig.subplots_adjust(right=0.82)

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
