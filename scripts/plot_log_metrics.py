#!/usr/bin/env python
"""
Visualise experiment logs (JSONL) produced by scripts/run_*_experiment.py.

Generates:
1. nDCG@10 per ranking iteration per topic.
2. RR@10 per ranking iteration per topic.
3. Scatter of query length vs. nDCG@10.
4. Query-length trajectories per topic.

Use:
    python scripts/plot_log_metrics.py --log logs/my_session_experiment_L4M.out \
        --output-dir analyses
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import ir_datasets
except ImportError:  # pragma: no cover
    ir_datasets = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics from geniie-lab JSONL logs.")
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="Path to a JSONL log file (use multiple --log flags for more files).",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to store generated plots (default: %(default)s).",
    )
    parser.add_argument(
        "--serp-metric",
        default="nDCG@10",
        help="Ranking metric key to plot (default: %(default)s).",
    )
    parser.add_argument(
        "--topicset",
        help="Optional ir_datasets topicset name to annotate topics with human-readable titles.",
    )
    parser.add_argument(
        "--compare-metric",
        default="metric_cum_recall",
        help="Metric key for multi-run comparison plot (default: %(default)s).",
    )
    parser.add_argument(
        "--compare-output",
        default="cum_recall_across_runs.png",
        help="Filename for multi-run comparison plot (default: %(default)s).",
    )
    return parser.parse_args()


def load_records(paths: Iterable[str]) -> Dict[str, List[dict]]:
    ranking_records: List[dict] = []
    query_records: List[dict] = []
    click_records: List[dict] = []
    last_query_by_topic: Dict[str, dict] = {}
    ranking_counter: Dict[str, int] = defaultdict(int)
    query_counter: Dict[str, int] = defaultdict(int)
    click_counter: Dict[str, int] = defaultdict(int)

    for log_path in paths:
        with open(log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                stage = entry.get("stage")
                topic = entry.get("topic_id", "<unknown>")
                model_name = entry.get("model")
                if stage in ("query", "reformulation"):
                    this_query = entry.get("query", "")
                    query_counter[topic] += 1
                    query_rec = {
                        "topic_id": topic,
                        "query": this_query,
                        "query_length": len(this_query.split()),
                        "stage": stage,
                        "iteration": query_counter[topic],
                        "created_at": entry.get("created_at"),
                        "log_path": log_path,
                        "model": model_name,
                    }
                    last_query_by_topic[topic] = query_rec
                    query_records.append(query_rec)
                elif stage == "ranking":
                    ranking_counter[topic] += 1
                    perf = entry.get("performance") or {}
                    query_meta = last_query_by_topic.get(topic, {})
                    ranking_records.append(
                        {
                            "topic_id": topic,
                            "iteration": ranking_counter[topic],
                            "metric_ndcg": perf.get("nDCG@10"),
                            "metric_rr": perf.get("RR@10"),
                            "metric_recall100": perf.get("Recall@100") or perf.get("R@100"),
                            "metric_cum_recall": perf.get("CumRecall"),
                            "metric_cum_recall100": perf.get("CumRecall@100"),
                            "metric_cum_rel_found": perf.get("CumRelFound"),
                            "query_length": query_meta.get("query_length"),
                            "query_text": query_meta.get("query"),
                            "query_stage": query_meta.get("stage"),
                            "created_at": entry.get("created_at"),
                            "log_path": log_path,
                            "model": model_name,
                        }
                    )
                elif stage == "click":
                    click_counter[topic] += 1
                    click_records.append(
                        {
                            "topic_id": topic,
                            "iteration": click_counter[topic],
                            "duplicate_doc_ids": entry.get("duplicate_doc_ids") or entry.get("duplicate_doc_ids"),
                            "doc_ids": entry.get("doc_ids"),
                            "ranking_list": entry.get("rankings"),
                            "created_at": entry.get("created_at"),
                            "log_path": log_path,
                            "model": model_name,
                        }
                    )
    return {"ranking": ranking_records, "queries": query_records, "clicks": click_records}


def build_topic_labels(topic_ids: Iterable[str], topicset: str | None) -> Dict[str, str]:
    labels = {}
    unique_ids = set(topic_ids)
    if not topicset:
        return {tid: tid for tid in unique_ids}
    if ir_datasets is None:
        print("[WARN] ir_datasets not installed; falling back to topic IDs.")
        return {tid: tid for tid in unique_ids}
    try:
        dataset = ir_datasets.load(topicset)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Failed to load topicset '{topicset}': {exc}")
        return {tid: tid for tid in unique_ids}

    topic_map = {}
    for topic in dataset.queries_iter():
        title = getattr(topic, "title", None) or getattr(topic, "text", None)
        topic_map[topic.query_id] = title

    for tid in unique_ids:
        title = topic_map.get(tid)
        if title:
            labels[tid] = f"{tid} - {title}"
        else:
            labels[tid] = tid
    return labels


def ensure_output_dir(path: str) -> Path:
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def compute_bar_figsize(num_iterations: int, num_topics: int, min_width: float = 11.0, min_height: float = 6.0) -> tuple[float, float]:
    width = max(min_width, num_iterations * 0.8 + num_topics * 1.2)
    height = max(min_height, 4.5 + num_topics * 0.1)
    return (width, height)


def build_topic_palette(topics: Iterable[str]) -> Dict[str, tuple]:
    """
    Map topics to distinct colours pulled from tab20 for consistent bar styling.
    """
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


def load_rel_counts(topicset: str | None) -> Dict[str, int]:
    """
    Return a mapping of query_id -> number of relevant documents (qrels count).
    """
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


def plot_metric_by_topic(
    ranking_records: List[dict],
    metric_key: str,
    output_path: Path,
    ylabel: str,
    topic_labels: Dict[str, str],
    model_names: List[str],
) -> None:
    per_topic: Dict[str, List[tuple[int, float]]] = defaultdict(list)
    for record in ranking_records:
        value = record.get(metric_key)
        if value is None:
            continue
        per_topic[record["topic_id"]].append((record["iteration"], value))

    if not per_topic:
        print(f"[WARN] No data for {metric_key}, skipping {output_path.name}")
        return

    topics = sorted(per_topic.keys())
    all_iterations = sorted({it for pairs in per_topic.values() for it, _ in pairs})
    if not all_iterations:
        print(f"[WARN] No iterations found for {metric_key}")
        return
    fig_width, fig_height = compute_bar_figsize(len(all_iterations), len(topics))
    plt.figure(figsize=(fig_width, fig_height))

    palette = build_topic_palette(topics)
    base_positions = list(range(len(all_iterations)))
    bar_width = 0.8 / max(1, len(topics))
    center_offset = (len(topics) - 1) * bar_width / 2

    for idx, topic in enumerate(topics):
        data = {it: val for it, val in per_topic[topic]}
        positions = [pos - center_offset + idx * bar_width for pos in base_positions]
        heights = [data.get(it, 0) for it in all_iterations]
        plt.bar(
            positions,
            heights,
            width=bar_width,
            label=topic_labels.get(topic, topic),
            color=palette[topic],
            edgecolor="black",
            linewidth=0.4,
        )

    plt.xlabel("Ranking iteration")
    plt.ylabel(ylabel)
    title_model = ", ".join(sorted(set(model_names))) if model_names else "unknown model"
    plt.title(f"{ylabel} per iteration\nModel(s): {title_model}")
    plt.xticks(base_positions, all_iterations)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_across_runs_horizontal(
    per_log_records: Dict[str, List[dict]],
    metric_key: str,
    output_path: Path,
    ylabel: str,
    topicset: str | None,
) -> None:
    if not per_log_records:
        print(f"[WARN] No data for multi-run plot {output_path.name}")
        return
    log_paths = sorted(per_log_records.keys())
    num_logs = len(log_paths)
    if num_logs == 0:
        return

    # Compute global y-limit for shared axis.
    max_val = 0.0
    for log_path in log_paths:
        for rec in per_log_records[log_path]:
            val = rec.get(metric_key)
            if isinstance(val, (int, float)) and val > max_val:
                max_val = float(val)
    y_max = max(1.0, max_val * 1.05) if max_val > 0 else 1.0

    fig, axes = plt.subplots(
        1,
        num_logs,
        figsize=(max(10, 4 * num_logs), 4.8),
        sharey=True,
    )
    if num_logs == 1:
        axes = [axes]

    for idx, log_path in enumerate(log_paths):
        ax = axes[idx]
        records = per_log_records[log_path]
        if not records:
            ax.set_title(Path(log_path).stem)
            ax.set_xlabel("Ranking iteration")
            continue

        topic_labels = build_topic_labels(
            [rec["topic_id"] for rec in records],
            topicset,
        )
        rel_counts = load_rel_counts(topicset)
        if rel_counts:
            topic_labels = {tid: f"{label} (rels={rel_counts.get(tid, 0)})" for tid, label in topic_labels.items()}

        per_topic: Dict[str, List[tuple[int, float]]] = defaultdict(list)
        for record in records:
            value = record.get(metric_key)
            if value is None:
                continue
            per_topic[record["topic_id"]].append((record["iteration"], value))

        topics = sorted(per_topic.keys())
        all_iterations = sorted({it for pairs in per_topic.values() for it, _ in pairs})
        if not all_iterations:
            ax.set_title(Path(log_path).stem)
            ax.set_xlabel("Ranking iteration")
            continue

        palette = build_topic_palette(topics)
        base_positions = list(range(len(all_iterations)))
        bar_width = 0.8 / max(1, len(topics))
        center_offset = (len(topics) - 1) * bar_width / 2

        for t_idx, topic in enumerate(topics):
            data = {it: val for it, val in per_topic[topic]}
            positions = [pos - center_offset + t_idx * bar_width for pos in base_positions]
            heights = [data.get(it, 0) for it in all_iterations]
            ax.bar(
                positions,
                heights,
                width=bar_width,
                label=topic_labels.get(topic, topic),
                color=palette[topic],
                edgecolor="black",
                linewidth=0.4,
            )

        ax.set_title(Path(log_path).stem)
        ax.set_xlabel("Ranking iteration")
        ax.set_xticks(base_positions)
        ax.set_xticklabels(all_iterations)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, y_max)

    axes[0].set_ylabel(ylabel)
    # Place legend on the last subplot to reduce clutter.
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, bbox_to_anchor=(1.04, 1), loc="upper left", fontsize="small")

    fig.suptitle(f"{ylabel} across runs (shared y-axis)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_query_length_vs_metric(
    ranking_records: List[dict],
    metric_key: str,
    output_path: Path,
    topic_labels: Dict[str, str],
    model_names: List[str],
) -> None:
    xs = []
    ys = []
    colors = []
    topics = set()
    for record in ranking_records:
        qlen = record.get("query_length")
        value = record.get(metric_key)
        if qlen is None or value is None:
            continue
        xs.append(qlen)
        ys.append(value)
        topic_id = record["topic_id"]
        colors.append(topic_id)
        topics.add(topic_id)

    if not xs:
        print(f"[WARN] No query length data to plot {output_path.name}")
        return

    plt.figure(figsize=(8, 6))
    palette = build_topic_palette(topics)
    mapped_colors = [palette[c] for c in colors]
    plt.scatter(xs, ys, c=mapped_colors, alpha=0.8)
    plt.xlabel("Query length (tokens)")
    plt.ylabel(metric_key)
    title_model = ", ".join(sorted(set(model_names))) if model_names else "unknown model"
    plt.title(f"{metric_key} vs. query length\nModel(s): {title_model}")
    plt.grid(alpha=0.3)
    legend_elements = []
    for topic, color_value in palette.items():
        legend_elements.append(
            Line2D([], [], marker="o", color="w", markerfacecolor=color_value, markersize=8, label=topic_labels.get(topic, topic))
        )
    if legend_elements:
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.04, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_query_length_trajectories(query_records: List[dict], output_path: Path, topic_labels: Dict[str, str]) -> None:
    if not query_records:
        print(f"[WARN] No query records to plot {output_path.name}")
        return
    per_topic: Dict[str, List[tuple[int, int]]] = defaultdict(list)
    for rec in query_records:
        per_topic[rec["topic_id"]].append((rec["iteration"], rec["query_length"]))
    topics = sorted(per_topic.keys())
    all_iterations = sorted({it for pairs in per_topic.values() for it, _ in pairs})
    fig_width, fig_height = compute_bar_figsize(len(all_iterations), len(topics))
    plt.figure(figsize=(fig_width, fig_height))
    palette = build_topic_palette(topics)
    base_positions = list(range(len(all_iterations)))
    bar_width = 0.8 / max(1, len(topics))
    center_offset = (len(topics) - 1) * bar_width / 2

    for idx, topic in enumerate(topics):
        data = {it: val for it, val in per_topic[topic]}
        positions = [pos - center_offset + idx * bar_width for pos in base_positions]
        heights = [data.get(it, 0) for it in all_iterations]
        plt.bar(
            positions,
            heights,
            width=bar_width,
            label=topic_labels.get(topic, topic),
            color=palette[topic],
            edgecolor="black",
            linewidth=0.4,
        )

    plt.xlabel("Query iteration (query + reformulation)")
    plt.ylabel("Query length (tokens)")
    plt.title("Query length trajectory per topic")
    plt.xticks(base_positions, all_iterations)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize="small")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_duplicate_clicks(click_records: List[dict], output_path: Path, topic_labels: Dict[str, str]) -> None:
    if not click_records:
        print(f"[WARN] No click records to plot {output_path.name}")
        return
    per_topic: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for rec in click_records:
        dups = rec.get("duplicate_doc_ids") or []
        per_topic[rec["topic_id"]][rec["iteration"]] += len(dups)
    if not per_topic:
        print(f"[WARN] No duplicate clicks found, skipping {output_path.name}")
        return
    topics = sorted(per_topic.keys())
    all_iters = sorted({i for topic_data in per_topic.values() for i in topic_data.keys()})
    fig_width, fig_height = compute_bar_figsize(len(all_iters), len(topics), min_width=9.0, min_height=4.8)
    plt.figure(figsize=(fig_width, fig_height))
    bar_width = 0.8 / max(1, len(topics))
    base_positions = list(range(len(all_iters)))
    palette = build_topic_palette(topics)
    for idx, topic in enumerate(topics):
        iters_to_counts = per_topic[topic]
        positions = [p - 0.4 + bar_width / 2 + idx * bar_width for p in base_positions]
        heights = [iters_to_counts.get(it, 0) for it in all_iters]
        plt.bar(positions, heights, width=bar_width, color=palette[topic], edgecolor="black", linewidth=0.4, label=topic_labels.get(topic, topic))
    plt.xticks(base_positions, all_iters)
    plt.xlabel("Click iteration")
    plt.ylabel("Duplicate clicks (LLM re-selected)")
    plt.title("Duplicate clicks per iteration and topic")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize="small")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combined_metrics(
    ranking_records: List[dict],
    output_path: Path,
    model_names: List[str],
) -> None:
    per_iter_ndcg: Dict[int, List[float]] = defaultdict(list)
    per_iter_rr: Dict[int, List[float]] = defaultdict(list)
    per_iter_recall: Dict[int, List[float]] = defaultdict(list)
    for rec in ranking_records:
        it = rec["iteration"]
        ndcg = rec.get("metric_ndcg")
        rr = rec.get("metric_rr")
        recall = rec.get("metric_recall100")
        if ndcg is not None:
            per_iter_ndcg[it].append(ndcg)
        if rr is not None:
            per_iter_rr[it].append(rr)
        if recall is not None:
            per_iter_recall[it].append(recall)
    iterations = sorted(set(per_iter_ndcg.keys()) | set(per_iter_rr.keys()) | set(per_iter_recall.keys()))
    if not iterations:
        print(f"[WARN] No metric data to plot {output_path.name}")
        return
    avg_ndcg = [sum(per_iter_ndcg.get(i, [])) / len(per_iter_ndcg[i]) if per_iter_ndcg.get(i) else None for i in iterations]
    avg_rr = [sum(per_iter_rr.get(i, [])) / len(per_iter_rr[i]) if per_iter_rr.get(i) else None for i in iterations]
    avg_recall = [sum(per_iter_recall.get(i, [])) / len(per_iter_recall[i]) if per_iter_recall.get(i) else None for i in iterations]
    # Replace missing averages with 0 to avoid bar plotting errors when data is absent.
    avg_ndcg = [val if val is not None else 0 for val in avg_ndcg]
    avg_rr = [val if val is not None else 0 for val in avg_rr]
    avg_recall = [val if val is not None else 0 for val in avg_recall]
    fig_width = max(9, len(iterations) * 0.8)
    plt.figure(figsize=(fig_width, 5))
    bar_width = 0.25
    base_positions = list(range(len(iterations)))
    rr_positions = [p + bar_width for p in base_positions]
    recall_positions = [p + 2 * bar_width for p in base_positions]
    plt.bar(base_positions, avg_ndcg, width=bar_width, color="tab:blue", edgecolor="black", linewidth=0.4, label="nDCG@10")
    plt.bar(rr_positions, avg_rr, width=bar_width, color="tab:orange", edgecolor="black", linewidth=0.4, label="RR@10")
    plt.bar(recall_positions, avg_recall, width=bar_width, color="tab:green", edgecolor="black", linewidth=0.4, label="Recall@100")
    plt.xlabel("Ranking iteration")
    plt.ylabel("Metric value")
    title_model = ", ".join(sorted(set(model_names))) if model_names else "unknown model"
    plt.title(f"Average retrieval metrics per iteration\nModel(s): {title_model}")
    tick_positions = [p + bar_width for p in base_positions]
    plt.xticks(tick_positions, iterations)
    plt.legend(loc="upper left")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_jaccard_similarity(
    query_records: List[dict],
    output_path: Path,
    model_names: List[str],
    topic_labels: Dict[str, str] | None = None,
) -> None:
    if not query_records:
        print(f"[WARN] No query records to plot {output_path.name}")
        return
    per_topic: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for rec in query_records:
        per_topic[rec["topic_id"]].append((rec["iteration"], rec["query"]))
    per_topic_sims: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for topic, records in per_topic.items():
        records.sort()
        for i in range(1, len(records)):
            prev_q = records[i - 1][1]
            cur_it, cur_q = records[i]
            sim = jaccard_similarity(prev_q, cur_q)
            per_topic_sims[topic].append((cur_it, sim))
    if not per_topic_sims:
        print(f"[WARN] No similarity data to plot {output_path.name}")
        return

    topics = sorted(per_topic_sims.keys())
    all_iterations = sorted({it for pairs in per_topic_sims.values() for it, _ in pairs})
    fig_width = max(8, len(all_iterations) * 0.8 + len(topics) * 0.8)
    plt.figure(figsize=(fig_width, 4.8))
    palette = build_topic_palette(topics)
    for topic in topics:
        pairs = per_topic_sims.get(topic, [])
        if not pairs:
            continue
        its = [it for it, _ in pairs]
        sims = [val for _, val in pairs]
        plt.plot(
            its,
            sims,
            marker="o",
            linewidth=1.6,
            markersize=4,
            color=palette[topic],
            label=(topic_labels or {}).get(topic, topic),
        )
    plt.xlabel("Query iteration (>=2)")
    plt.ylabel("Jaccard similarity to previous query")
    title_model = ", ".join(sorted(set(model_names))) if model_names else "unknown model"
    plt.title(f"Query reformulation similarity\nModel(s): {title_model}")
    plt.xticks(all_iterations)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_jaccard_per_iteration(log_path: str) -> Tuple[str, Dict[int, List[float]], float | None]:
    per_topic: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            stage = entry.get("stage")
            if stage not in {"query", "reformulation"}:
                continue
            topic_id = entry.get("topic_id", "<unknown>")
            per_topic[topic_id].append((len(per_topic[topic_id]) + 1, entry.get("query", "")))
    per_iter: Dict[int, List[float]] = defaultdict(list)
    all_sims: List[float] = []
    for _, records in per_topic.items():
        records.sort()
        for i in range(1, len(records)):
            prev_q = records[i - 1][1]
            cur_q = records[i][1]
            sim = jaccard_similarity(prev_q, cur_q)
            iter_num = records[i][0]
            per_iter[iter_num].append(sim)
            all_sims.append(sim)
    avg_sim = sum(all_sims) / len(all_sims) if all_sims else None
    return Path(log_path).stem, per_iter, avg_sim


def plot_jaccard_across_runs_by_iteration(log_paths: List[str], output_path: Path) -> None:
    summaries = [compute_jaccard_per_iteration(lp) for lp in log_paths]
    iterations = sorted({it for _, per_iter, _ in summaries for it in per_iter.keys()})
    if not iterations:
        print(f"[WARN] No Jaccard data found across runs, skipping {output_path.name}")
        return
    num_runs = len(summaries)
    bar_width = 0.8 / max(1, num_runs)
    base_positions = list(range(len(iterations)))
    palette = plt.cm.get_cmap("tab10")

    plt.figure(figsize=(max(8, len(iterations) * 1.2), 5.5))
    for idx, (label, per_iter, avg_sim) in enumerate(summaries):
        offsets = [pos - 0.4 + bar_width / 2 + idx * bar_width for pos in base_positions]
        heights = []
        positions = []
        for pos, it in zip(offsets, iterations):
            vals = per_iter.get(it)
            if vals:
                positions.append(pos)
                heights.append(sum(vals) / len(vals))
        color = palette(idx % palette.N)
        plt.bar(positions, heights, width=bar_width, color=color, edgecolor="black", linewidth=0.4, label=label)
        if avg_sim is not None:
            plt.axhline(avg_sim, color=color, linestyle="--", linewidth=1, alpha=0.7)

    plt.xlabel("Query iteration (>=2)")
    plt.ylabel("Avg Jaccard similarity to previous query")
    plt.title("Query reformulation similarity across runs")
    plt.xticks(base_positions, iterations)
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    records = load_records(args.log)
    ranking_records = records["ranking"]
    query_records = records["queries"]
    click_records = records["clicks"]
    model_names = [rec.get("model") for rec in ranking_records if rec.get("model")]

    if not ranking_records:
        print("[ERROR] No ranking records found. Ensure the log path is correct.")
        return

    output_dir = ensure_output_dir(args.output_dir)
    topic_labels = build_topic_labels(
        [rec["topic_id"] for rec in ranking_records],
        args.topicset,
    )
    rel_counts = load_rel_counts(args.topicset)
    if rel_counts:
        topic_labels = {tid: f"{label} (rels={rel_counts.get(tid, 0)})" for tid, label in topic_labels.items()}

    ndcg_path = output_dir / "ndcg_by_topic.png"
    plot_metric_by_topic(ranking_records, "metric_ndcg", ndcg_path, ylabel="nDCG@10", topic_labels=topic_labels, model_names=model_names)

    rr_path = output_dir / "rr_by_topic.png"
    plot_metric_by_topic(ranking_records, "metric_rr", rr_path, ylabel="Reciprocal Rank @10", topic_labels=topic_labels, model_names=model_names)

    recall_path = output_dir / "recall100_by_topic.png"
    plot_metric_by_topic(ranking_records, "metric_recall100", recall_path, ylabel="Recall@100", topic_labels=topic_labels, model_names=model_names)

    cum_recall_path = output_dir / "cum_recall_by_topic.png"
    plot_metric_by_topic(ranking_records, "metric_cum_recall", cum_recall_path, ylabel="Cumulative Recall (LLM-judged)", topic_labels=topic_labels, model_names=model_names)

    cum_recall100_path = output_dir / "cum_recall100_by_topic.png"
    plot_metric_by_topic(ranking_records, "metric_cum_recall100", cum_recall100_path, ylabel="Cumulative Recall@100", topic_labels=topic_labels, model_names=model_names)

    scatter_path = output_dir / "query_length_vs_ndcg.png"
    plot_query_length_vs_metric(ranking_records, "metric_ndcg", scatter_path, topic_labels, model_names)

    query_traj_path = output_dir / "query_length_traj.png"
    plot_query_length_trajectories(query_records, query_traj_path, topic_labels)

    combined_path = output_dir / "combined_rr_ndcg.png"
    plot_combined_metrics(ranking_records, combined_path, model_names)

    jaccard_path = output_dir / "query_jaccard_similarity.png"
    plot_jaccard_similarity(query_records, jaccard_path, model_names, topic_labels)

    if len(args.log) > 1:
        jaccard_runs_path = output_dir / "query_jaccard_similarity_across_runs.png"
        plot_jaccard_across_runs_by_iteration(args.log, jaccard_runs_path)

        per_log_records: Dict[str, List[dict]] = defaultdict(list)
        for rec in ranking_records:
            per_log_records[rec["log_path"]].append(rec)
        compare_path = output_dir / args.compare_output
        metric_key = args.compare_metric
        ylabel = metric_key.replace("metric_", "").replace("_", " ").title()
        plot_metric_across_runs_horizontal(
            per_log_records,
            metric_key=metric_key,
            output_path=compare_path,
            ylabel=ylabel,
            topicset=args.topicset,
        )

    dup_clicks_path = output_dir / "duplicate_clicks_by_topic.png"
    plot_duplicate_clicks(click_records, dup_clicks_path, topic_labels)

    summary_lines = [f"Saved plots to {output_dir.resolve()}"]
    for path, desc in [
        (ndcg_path, "nDCG@10 per topic iteration"),
        (rr_path, "RR@10 per topic iteration"),
        (recall_path, "Recall@100 per topic iteration"),
        (cum_recall_path, "Cumulative session recall per topic"),
        (cum_recall100_path, "Cumulative Recall@100 per topic"),
        (scatter_path, "Scatter of query length vs nDCG@10"),
        (query_traj_path, "Query length trajectory per topic"),
        (combined_path, "Combined metrics averages per iteration"),
        (jaccard_path, "Average Jaccard similarity of successive queries"),
        (dup_clicks_path, "Total duplicate clicks per topic"),
    ]:
        if path.exists():
            summary_lines.append(f"  - {path.name}: {desc}")
    if len(args.log) > 1:
        if jaccard_runs_path.exists():
            summary_lines.append(f"  - {jaccard_runs_path.name}: Jaccard similarity comparison across runs (by iteration)")
        if compare_path.exists():
            summary_lines.append(f"  - {compare_path.name}: {ylabel} across runs (shared y-axis)")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
