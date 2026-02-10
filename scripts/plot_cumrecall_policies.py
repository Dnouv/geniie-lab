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
import re
from datetime import datetime, timezone
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
        default="CumRecall@100",
        help="Performance key to plot (default: %(default)s).",
    )
    parser.add_argument(
        "--metrics",
        default="auto",
        help="Comma-separated performance keys to plot, or 'auto' to detect from logs (default: %(default)s).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=0.0,
        help="Max value for y-axis (0 = auto from data).",
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
        "--output-dir",
        default="cumrecall_policies",
        help="Output directory root (default: %(default)s).",
    )
    parser.add_argument(
        "--output-by-topic-dir",
        default=None,
        help="Optional directory for per-topic segmented policy plots (default: <output_dir>/<metric>/by_topic).",
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


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def load_ranking_records(log_path: str, metric_key: str) -> List[dict]:
    ranking_counter: Dict[str, int] = defaultdict(int)
    last_query_by_topic: Dict[str, str] = {}
    prev_query_by_topic: Dict[str, str] = {}
    records: List[dict] = []
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
            if metric_key == "Jaccard":
                value = jaccard_similarity(
                    last_query_by_topic.get(topic, ""),
                    prev_query_by_topic.get(topic, ""),
                )
            else:
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


def slugify_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "topic"


def plot_per_topic_segmented_policies(
    per_policy_records: Dict[str, List[dict]],
    topic_labels: Dict[str, str],
    output_dir: Path,
    y_max: float,
    metric_key: str,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_order = list(per_policy_records.keys())
    per_topic_per_policy: Dict[str, Dict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))

    for policy, records in per_policy_records.items():
        for rec in records:
            value = rec.get("value")
            if value is None:
                continue
            topic = rec.get("topic_id", "<unknown>")
            per_topic_per_policy[topic][policy].append((int(rec.get("iteration", 0)), float(value)))

    saved: List[Path] = []
    for topic in sorted(per_topic_per_policy.keys()):
        policy_data = per_topic_per_policy[topic]
        ordered_policies = [p for p in policy_order if p in policy_data]
        if not ordered_policies:
            continue
        for p in ordered_policies:
            policy_data[p].sort(key=lambda x: x[0])

        segment_len = max((max((it for it, _ in policy_data[p]), default=0) for p in ordered_policies), default=0)
        if segment_len <= 0:
            continue

        fig_width = max(10, len(ordered_policies) * 3.2 + segment_len * 0.28)
        plt.figure(figsize=(fig_width, 4.8))
        cmap = plt.colormaps.get_cmap("tab10")

        ax = plt.gca()
        for idx, policy in enumerate(ordered_policies):
            pairs = policy_data[policy]
            iterations = list(range(1, segment_len + 1))
            xs = [idx * segment_len + it for it in iterations]
            ys = build_series(metric_key, pairs, segment_len)
            color = cmap(idx % cmap.N)
            plt.bar(
                xs,
                ys,
                width=0.86,
                color=color,
                edgecolor="black",
                linewidth=0.2,
                label=policy,
            )
            if idx > 0:
                plt.axvline(idx * segment_len + 0.5, color="gray", linestyle="--", alpha=0.35, linewidth=1.0)

        tick_step = auto_xtick_step(list(range(1, segment_len + 1)))
        tick_positions: List[float] = []
        tick_labels: List[str] = []
        for idx in range(len(ordered_policies)):
            for it in range(1, segment_len + 1, tick_step):
                tick_positions.append(idx * segment_len + it)
                tick_labels.append(str(it))
        plt.xticks(tick_positions, tick_labels, rotation=0)
        plt.xlabel("Ranking iteration index (resets in each policy segment)")
        plt.ylabel(metric_key)
        if y_max > 0:
            plt.ylim(0, y_max)
        else:
            max_local = max((val for pairs in policy_data.values() for _, val in pairs), default=0.0)
            plt.ylim(0, (max_local * 1.08) if max_local > 0 else 1.0)
        plt.xlim(0.5, len(ordered_policies) * segment_len + 0.5)
        plt.grid(alpha=0.3)
        for idx in range(len(ordered_policies)):
            if idx % 2 == 1:
                start = idx * segment_len + 0.5
                end = (idx + 1) * segment_len + 0.5
                ax.axvspan(start, end, color="gray", alpha=0.05, zorder=0)
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize="small",
            title="Policy",
        )
        plt.title(topic_labels.get(topic, topic), pad=14)
        plt.tight_layout(rect=(0, 0, 0.82, 1))

        out_path = output_dir / f"cumrecall_segmented_{slugify_filename(topic)}.png"
        plt.savefig(out_path, dpi=220)
        plt.close()
        saved.append(out_path)

    if not saved:
        print(f"[WARN] No per-topic policy-segmented plots created in {output_dir}")
    return saved


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


def discover_metric_keys(log_paths: Iterable[str]) -> List[str]:
    perf_keys: set[str] = set()
    saw_query = False
    for log_path in log_paths:
        with open(log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                stage = entry.get("stage")
                if stage in ("query", "reformulation"):
                    if entry.get("query") is not None:
                        saw_query = True
                if stage != "ranking":
                    continue
                perf = entry.get("performance") or {}
                perf_keys.update(perf.keys())
    preferred = [
        "CumRecall@100",
        "CumRecall",
        "CumRelFound@100",
        "CumRelFound",
        "nDCG@10",
        "RR@10",
        "R@100",
        "Recall@100",
    ]
    ordered = [k for k in preferred if k in perf_keys]
    ordered.extend(sorted(perf_keys - set(ordered)))
    if saw_query:
        ordered.append("Jaccard")
    return ordered


def plot_metric(
    args: argparse.Namespace,
    metric_key: str,
    output_path: Path,
    output_by_topic_dir: Path,
) -> None:
    per_policy_records: Dict[str, List[dict]] = {}
    for log_path, label in zip(args.log, args.label):
        per_policy_records[label] = load_ranking_records(log_path, metric_key)

    all_topics = sorted({rec["topic_id"] for records in per_policy_records.values() for rec in records})
    if not all_topics:
        raise SystemExit("No ranking records found for the given logs.")

    topic_labels = build_topic_labels(all_topics, args.topicset)
    rel_counts = load_rel_counts(args.topicset)
    if rel_counts:
        topic_labels = {tid: f"{label} (rels={rel_counts.get(tid, 0)})" for tid, label in topic_labels.items()}

    palette = build_topic_palette(all_topics)

    num_policies = len(per_policy_records)
    max_value = 0.0
    max_iters_overall = 0
    for records in per_policy_records.values():
        iters = {rec["iteration"] for rec in records}
        if iters:
            max_iters_overall = max(max_iters_overall, max(iters))
        for rec in records:
            value = rec.get("value")
            if value is not None:
                try:
                    max_value = max(max_value, float(value))
                except (TypeError, ValueError):
                    pass
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
        max_iter = max(iterations)
        iterations = list(range(1, max_iter + 1))
        base_positions = list(range(len(iterations)))
        bar_width = 0.8 / max(1, len(all_topics))
        center_offset = (len(all_topics) - 1) * bar_width / 2

        for idx, topic in enumerate(all_topics):
            data_pairs = per_topic.get(topic, [])
            heights = build_series(metric_key, data_pairs, max_iter)
            positions = [pos - center_offset + idx * bar_width for pos in base_positions]
            ax.bar(
                positions,
                heights,
                width=bar_width,
                color=palette[topic],
                edgecolor="black",
                linewidth=0.2,
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
        if args.y_max > 0:
            ax.set_ylim(0, args.y_max)
        else:
            ax.set_ylim(0, (max_value * 1.08) if max_value > 0 else 1.0)

    axes[0].set_ylabel(metric_key)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")
        fig.subplots_adjust(right=0.82)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    output_by_topic_dir.mkdir(parents=True, exist_ok=True)
    per_topic_paths = plot_per_topic_segmented_policies(
        per_policy_records=per_policy_records,
        topic_labels=topic_labels,
        output_dir=output_by_topic_dir,
        y_max=args.y_max,
        metric_key=metric_key,
    )

    print(f"Saved {output_path.resolve()}")
    if per_topic_paths:
        print(f"Saved {len(per_topic_paths)} per-topic segmented plots to {output_by_topic_dir.resolve()}")


def main() -> None:
    args = parse_args()
    if len(args.log) != len(args.label):
        raise SystemExit("Number of --log and --label arguments must match.")

    if args.metrics.strip().lower() == "auto":
        metrics = discover_metric_keys(args.log)
    else:
        raw_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        metrics = raw_metrics or [args.metric_key]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name) for p in output_root.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    run_id = (max(existing) + 1) if existing else 0
    output_base = output_root / str(run_id)
    output_base.mkdir(parents=True, exist_ok=True)
    info_path = output_base / "run_info.txt"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with info_path.open("w", encoding="utf-8") as handle:
        handle.write(f"run_id: {run_id}\n")
        handle.write(f"timestamp: {timestamp}\n")
        handle.write(f"metrics: {', '.join(metrics)}\n")
        handle.write(f"logs:\n")
        for log_path, label in zip(args.log, args.label):
            handle.write(f"  - {label}: {log_path}\n")
    for metric_key in metrics:
        metric_slug = slugify_filename(metric_key)
        output_path = output_base / metric_slug / "overview.png"
        if args.output_by_topic_dir:
            output_by_topic_dir = Path(args.output_by_topic_dir) / metric_slug / "by_topic"
        else:
            output_by_topic_dir = output_base / metric_slug / "by_topic"
        plot_metric(args, metric_key, output_path, output_by_topic_dir)


if __name__ == "__main__":
    main()
