#!/usr/bin/env python
"""Bulk index an ir_datasets corpus into OpenSearch for geniie-lab experiments."""

import argparse
import logging
import os
from typing import Iterator, Mapping, Any, Tuple, Optional

from dotenv import load_dotenv
import ir_datasets
from opensearchpy import OpenSearch, helpers

try:  # pragma: no cover
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

load_dotenv()


def parse_args() -> argparse.Namespace:
    default_user = os.environ.get("OPENSEARCH_ADMIN_USER", "admin")
    default_pass = (
        os.environ.get("OPENSEARCH_ADMIN_PASS")
        or os.environ.get("OPENSEARCH_ADMIN_PASSWORD")
        or "admin"
    )
    parser = argparse.ArgumentParser(
        description="Index an ir_datasets corpus into OpenSearch."
    )
    parser.add_argument(
        "--source",
        choices=("ir_datasets", "hf"),
        default="ir_datasets",
        help="Where to load the corpus from (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset",
        default="beir/dbpedia-entity",
        help="Dataset name as recognised by ir_datasets (default: %(default)s).",
    )
    parser.add_argument(
        "--hf-name",
        default="BeIR/dbpedia-entity",
        help="Hugging Face dataset path (default: %(default)s).",
    )
    parser.add_argument(
        "--hf-config",
        default="corpus",
        help="Hugging Face dataset config to use (default: %(default)s).",
    )
    parser.add_argument(
        "--hf-split",
        default="corpus",
        help="Hugging Face split to load (default for BEIR corpora: %(default)s).",
    )
    parser.add_argument(
        "--hf-streaming",
        action="store_true",
        help="Enable streaming mode when loading from Hugging Face (saves disk, no length info).",
    )
    parser.add_argument(
        "--index",
        default="dbpedia_entity_bm25",
        help="Name of the OpenSearch index to create/populate (default: %(default)s).",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="OpenSearch host (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9200,
        help="OpenSearch port (default: %(default)s).",
    )
    parser.add_argument(
        "--username",
        default=default_user,
        help="OpenSearch admin username (default: value from OPENSEARCH_ADMIN_USER or 'admin').",
    )
    parser.add_argument(
        "--password",
        default=default_pass,
        help="OpenSearch admin password (default: OPENSEARCH_ADMIN_PASS / OPENSEARCH_ADMIN_PASSWORD or 'admin').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of documents per bulk request (default: %(default)s).",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="Number of primary shards for the index (default: %(default)s).",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=0,
        help="Number of replica shards for the index (default: %(default)s).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the target index before indexing if it already exists.",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="Use HTTP instead of HTTPS when connecting to OpenSearch.",
    )
    return parser.parse_args()


def create_client(args: argparse.Namespace) -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": args.host, "port": args.port}],
        http_auth=(args.username, args.password),
        use_ssl=not args.no_ssl,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )


def ensure_index(client: OpenSearch, args: argparse.Namespace) -> None:
    index_body = {
        "settings": {
            "index": {
                "number_of_shards": args.shards,
                "number_of_replicas": args.replicas,
            }
        },
        "mappings": {
            "properties": {
                "docid": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
            }
        },
    }

    exists = client.indices.exists(index=args.index)
    if exists and args.recreate:
        logging.info("Deleting existing index '%s'.", args.index)
        client.indices.delete(index=args.index)
        exists = False

    if not exists:
        logging.info(
            "Creating index '%s' with %s shard(s) and %s replica(s).",
            args.index,
            args.shards,
            args.replicas,
        )
        client.indices.create(index=args.index, body=index_body)
    else:
        logging.info("Index '%s' already exists. Reusing it.", args.index)


def generate_docs_ir(dataset, index_name: str) -> Iterator[Mapping[str, Any]]:
    for doc in dataset.docs_iter():
        docid = getattr(doc, "doc_id", None)
        if docid is None:
            raise ValueError("Dataset document is missing 'doc_id'.")
        title = getattr(doc, "title", "") or ""
        text = getattr(doc, "text", "") or ""
        yield {
            "_index": index_name,
            "_id": docid,
            "_source": {
                "docid": docid,
                "title": title,
                "text": text,
            },
        }

def generate_docs_hf(dataset, index_name: str) -> Iterator[Mapping[str, Any]]:
    for doc in dataset:
        docid = (
            doc.get("doc_id")
            or doc.get("docid")
            or doc.get("_id")
            or doc.get("id")
        )
        if docid is None:
            raise ValueError("Hugging Face document is missing an id field.")
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        yield {
            "_index": index_name,
            "_id": docid,
            "_source": {
                "docid": docid,
                "title": title,
                "text": text,
            },
        }


def prepare_doc_source(args: argparse.Namespace) -> Tuple[Iterator[Mapping[str, Any]], Optional[int], str]:
    if args.source == "hf":
        if load_dataset is None:
            raise RuntimeError("datasets library is not installed but --source hf was requested.")
        dataset = load_dataset(
            args.hf_name,
            args.hf_config,
            split=args.hf_split,
            streaming=args.hf_streaming,
        )
        doc_count = None
        if not args.hf_streaming and hasattr(dataset, "num_rows"):
            doc_count = dataset.num_rows
        generator = generate_docs_hf(dataset, args.index)
        label = f"{args.hf_name}/{args.hf_config}:{args.hf_split}"
        return generator, doc_count, label

    dataset = ir_datasets.load(args.dataset)
    doc_count = dataset.docs_count()
    generator = generate_docs_ir(dataset, args.index)
    return generator, doc_count, args.dataset


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    doc_iterator, doc_count, source_label = prepare_doc_source(args)
    logging.info("Loaded dataset source '%s'.", source_label)
    if doc_count is not None:
        logging.info("Dataset contains %s documents.", doc_count)

    client = create_client(args)
    ensure_index(client, args)

    logging.info(
        "Indexing dataset '%s' into '%s' with batch size %s.",
        source_label,
        args.index,
        args.batch_size,
    )

    helpers.bulk(
        client,
        doc_iterator,
        chunk_size=args.batch_size,
        request_timeout=120,
    )
    logging.info("Finished indexing '%s'.", args.index)


if __name__ == "__main__":
    main()
