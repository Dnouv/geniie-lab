#!/usr/bin/env python
"""
List relevant documents (and relevance grades) for topics in an ir_datasets collection.

Usage examples:
    python scripts/show_topic_rels.py --topicset beir/dbpedia-entity/dev --topic-id INEX_LD-20120112 --limit 10
    python scripts/show_topic_rels.py --topicset beir/dbpedia-entity/dev --limit 5
"""

import argparse
from typing import Dict, List, Tuple

import ir_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show relevant documents per topic from ir_datasets.")
    parser.add_argument(
        "--topicset",
        required=True,
        help="Dataset identifier (e.g., beir/dbpedia-entity/dev).",
    )
    parser.add_argument(
        "--topic-id",
        help="Specific topic ID to inspect; omit to list all topics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of relevance entries to display per topic (default: %(default)s).",
    )
    parser.add_argument(
        "--min-rels",
        type=int,
        default=0,
        help="Only show topics with at least this many relevant docs (relevance>0).",
    )
    return parser.parse_args()


def load_qrels(dataset) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, List[Tuple[str, int]]]]:
    """
    Return (positives_only, all_judged) maps:
      positives_only: relevance > 0
      all_judged: all qrels, including relevance == 0
    """
    positives: Dict[str, List[Tuple[str, int]]] = {}
    all_qrels: Dict[str, List[Tuple[str, int]]] = {}
    for item in dataset.qrels_iter():
        all_qrels.setdefault(item.query_id, []).append((item.doc_id, item.relevance))
        if getattr(item, "relevance", 0) and item.relevance > 0:
            positives.setdefault(item.query_id, []).append((item.doc_id, item.relevance))
    for d in (positives, all_qrels):
        for key in d:
            d[key].sort(key=lambda x: -x[1])
    return positives, all_qrels


def load_query_titles(dataset) -> Dict[str, str]:
    titles: Dict[str, str] = {}
    for topic in dataset.queries_iter():
        title = getattr(topic, "title", None) or getattr(topic, "text", "")
        titles[topic.query_id] = title
    return titles


def main() -> None:
    args = parse_args()
    dataset = ir_datasets.load(args.topicset)
    pos_qrels, all_qrels = load_qrels(dataset)
    titles = load_query_titles(dataset)

    topic_ids = [args.topic_id] if args.topic_id else sorted(titles.keys())
    for topic_id in topic_ids:
        title = titles.get(topic_id, "")
        rel_docs = pos_qrels.get(topic_id, [])
        all_docs = all_qrels.get(topic_id, [])
        non_rel_docs = [pair for pair in all_docs if pair[1] == 0]
        if args.min_rels and len(rel_docs) < args.min_rels:
            continue
        print(f"{topic_id} - {title}")
        print(f"  Total judged: {len(all_docs)} | Relevant (>0): {len(rel_docs)} | Not relevant (=0): {len(non_rel_docs)}")
        print(f"  Preview relevant (up to {args.limit}):")
        for doc_id, rel in rel_docs[: args.limit]:
            print(f"    rel={rel}: {doc_id}")
        print(f"  Preview not relevant (up to {args.limit}):")
        for doc_id, rel in non_rel_docs[: args.limit]:
            print(f"    rel={rel}: {doc_id}")
        print()


if __name__ == "__main__":
    main()
