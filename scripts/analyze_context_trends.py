#!/usr/bin/env python
"""
Analyse how conversation context grows over time and how it correlates with query diversity and retrieval metrics.

Outputs CSV summaries and optional plots that show:
  * Token count of the conversation history per stage (requires cumulative logs via full_log).
  * Query edit distance (Jaccard similarity) vs. iteration.
  * Unique relevant docs discovered vs. iteration (requires qrels via ir_datasets).

Example:
  python scripts/analyze_context_trends.py \
      --log logs/my_session_experiment_L4M.out \
      --full-log logs/my_session_experiment_L4M.log \
      --topicset beir/dbpedia-entity/dev \
      --output-dir analyses/context_trends
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

import ir_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse context growth vs. query/retrieval trends.")
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="Path(s) to JSONL stage logs (stdout).",
    )
    parser.add_argument(
        "--full-log",
        help="Path to stderr log containing 'Full Log' dumps (needed for exact context snapshots).",
    )
    parser.add_argument(
        "--topicset",
        required=True,
        help="ir_datasets topicset name (needed for qrels).",
    )
    parser.add_argument(
        "--output-dir",
        default="context_analysis",
        help="Where to store CSV/plots (default: %(default)s).",
    )
    return parser.parse_args()


def load_qrels(topicset: str) -> Dict[str, Dict[str, int]]:
    ds = ir_datasets.load(topicset)
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for item in ds.qrels_iter():
        qrels[item.query_id][item.doc_id] = item.relevance
    return qrels


def load_queries_from_full_log(log_path: str) -> Dict[str, List[str]]:
    """
    Parse conversation snapshots and return the prompt text for each topic.
    Expects the stderr log to include sections like:
        -------------------- Full Log --------------------
        [{...}, {...}, ...]
    """
    if not log_path:
        return {}
    data = Path(log_path).read_text(encoding="utf-8")
    segments = data.split("Full Log")
    topic_transcripts: Dict[str, List[str]] = {}
    current_topic = None
    for block in segments:
        lines = block.strip().splitlines()
        for line in lines:
            if line.startswith("-------------------- Topic:"):
                parts = line.split("Topic:")[-1].strip().split(" ", 1)
                current_topic = parts[0]
            if line.startswith("[{") or line.startswith("[\n") or (line.startswith("[") and "role" in line):
                try:
                    transcript = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if current_topic:
                    topic_transcripts[current_topic] = [
                        f"{entry.get('role')}: {entry.get('content', '')}" for entry in transcript
                    ]
    return topic_transcripts


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def build_topic_palette(topics):
    cmap = plt.cm.get_cmap("tab20")
    palette = {}
    for idx, topic in enumerate(sorted(set(topics))):
        palette[topic] = cmap(idx % cmap.N)
    return palette


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qrels = load_qrels(args.topicset)
    transcripts = load_queries_from_full_log(args.full_log) if args.full_log else {}

    ranking_data: Dict[str, List[dict]] = defaultdict(list)
    query_data: Dict[str, List[dict]] = defaultdict(list)

    for log_path in args.log:
        with open(log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                stage = entry.get("stage")
                topic = entry.get("topic_id")
                if stage in {"query", "reformulation"}:
                    query_text = entry.get("query", "")
                    query_data[topic].append(
                        {
                            "iteration": len(query_data[topic]) + 1,
                            "text": query_text,
                        }
                    )
                elif stage == "ranking":
                    metrics = entry.get("performance") or {}
                    ranking_data[topic].append(
                        {
                            "iteration": len(ranking_data[topic]) + 1,
                            "nDCG@10": metrics.get("nDCG@10"),
                            "RR@10": metrics.get("RR@10"),
                        }
                    )

    topic_palette = build_topic_palette(set(query_data.keys()) | set(ranking_data.keys()))

    # Plot query similarity over iterations
    for topic, records in query_data.items():
        if len(records) < 2:
            continue
        sims = [jaccard_similarity(records[i - 1]["text"], rec["text"]) for i, rec in enumerate(records[1:], start=1)]
        iterations = list(range(2, len(records) + 1))
        plt.figure(figsize=(8, 4))
        plt.bar(iterations, sims, color=topic_palette.get(topic), edgecolor="black", linewidth=0.4)
        plt.title(f"Query Jaccard similarity per iteration - {topic}")
        plt.xlabel("Query iteration")
        plt.ylabel("Similarity to previous query")
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{topic}_query_similarity.png", dpi=200)
        plt.close()

    # Plot cumulative unique relevant docs discovered
    for topic, rankings in ranking_data.items():
        rel_docs = set()
        cumulative_counts = []
        for rank_entry in rankings:
            # For this demo we don't have the clicked doc IDs in ranking stage, so this is a stub.
            cumulative_counts.append(len(rel_docs))
        if not cumulative_counts:
            continue
        iterations = list(range(1, len(cumulative_counts) + 1))
        plt.figure(figsize=(8, 4))
        plt.bar(iterations, cumulative_counts, color=topic_palette.get(topic), edgecolor="black", linewidth=0.4)
        plt.title(f"Cumulative relevant docs discovered - {topic}")
        plt.xlabel("Ranking iteration")
        plt.ylabel("Unique relevant docs (clicked/judged)")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{topic}_cumulative_rels.png", dpi=200)
        plt.close()

    print(f"Saved analysis to {out_dir}")


if __name__ == "__main__":
    main()
