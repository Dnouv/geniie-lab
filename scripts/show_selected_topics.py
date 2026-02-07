#!/usr/bin/env python
"""
Preview which topics would be selected by the experiment settings.

Examples:
  python scripts/show_selected_topics.py --topicset beir/dbpedia-entity/test --max-topics 3
  python scripts/show_selected_topics.py --topicset beir/dbpedia-entity/test --max-topics 3 --min-rels 100
  python scripts/show_selected_topics.py --topicset trec-covid --max-topics 5 --show-fields
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, Iterable, Tuple

import ir_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show selected topics given dataset constraints.")
    parser.add_argument("--topicset", required=True, help="ir_datasets topicset (e.g., beir/dbpedia-entity/test).")
    parser.add_argument("--max-topics", type=int, default=3, help="Number of topics to show (default: %(default)s).")
    parser.add_argument("--min-rels", type=int, default=0, help="Minimum relevant docs (relevance>0) to include.")
    parser.add_argument("--show-fields", action="store_true", help="Print available topic fields from the dataset.")
    return parser.parse_args()


def topic_text(topic) -> str:
    for field in ("title", "text", "query", "description", "narrative"):
        value = getattr(topic, field, None)
        if value:
            return str(value)
    return ""


def load_rel_counts(dataset) -> Dict[str, int]:
    rels: Dict[str, int] = defaultdict(int)
    for qrel in dataset.qrels_iter():
        if getattr(qrel, "relevance", 0) and qrel.relevance > 0:
            rels[qrel.query_id] += 1
    return rels


def main() -> None:
    args = parse_args()
    dataset = ir_datasets.load(args.topicset)

    if args.show_fields:
        try:
            sample = next(dataset.queries_iter())
            print(f"Topic fields for {args.topicset}:")
            for field in dir(sample):
                if not field.startswith("_"):
                    value = getattr(sample, field)
                    if isinstance(value, (str, int, float)):
                        print(f"  - {field}: {str(value)[:120]}")
                    else:
                        print(f"  - {field}: {type(value).__name__}")
            print()
        except StopIteration:
            print("No topics found.")

    rel_counts = load_rel_counts(dataset) if args.min_rels else {}
    shown = 0
    for topic in dataset.queries_iter():
        if args.min_rels and rel_counts.get(topic.query_id, 0) < args.min_rels:
            continue
        title = topic_text(topic)
        rels = rel_counts.get(topic.query_id, 0) if rel_counts else "n/a"
        print(f"{topic.query_id} - {title} (rels={rels})")
        shown += 1
        if shown >= args.max_topics:
            break


if __name__ == "__main__":
    main()
