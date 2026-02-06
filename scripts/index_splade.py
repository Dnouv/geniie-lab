#!/usr/bin/env python
"""Bulk index a corpus into OpenSearch with SPLADE expansions."""

import argparse
import logging
import os
from typing import Iterator, Mapping, Any, Tuple, Optional, Iterable, List

from dotenv import load_dotenv
import ir_datasets
from opensearchpy import OpenSearch, helpers
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

try:  # pragma: no cover
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

load_dotenv()


class SpladeEncoder:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_doc_length: int,
        top_k: int,
    ):
        self.model_name = model_name
        self.device = device
        self.max_doc_length = max_doc_length
        self.top_k = top_k

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_doc_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits  # [batch, seq_len, vocab]
        sparse_weights = torch.log1p(F.relu(logits)).max(dim=1).values  # [batch, vocab]
        topk = torch.topk(sparse_weights, k=self.top_k, dim=1)

        token_ids = topk.indices.tolist()
        weights = topk.values.tolist()
        special_tokens = set(self.tokenizer.all_special_tokens)

        bow_texts = []
        for ids, ws in zip(token_ids, weights):
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            bow_tokens = []
            for token, weight in zip(tokens, ws):
                if token in special_tokens:
                    continue
                repeat_count = max(1, int(round(float(weight))))
                bow_tokens.extend([token] * repeat_count)
            bow_texts.append(" ".join(bow_tokens))
        return bow_texts


def parse_args() -> argparse.Namespace:
    default_user = os.environ.get("OPENSEARCH_ADMIN_USER", "admin")
    default_pass = (
        os.environ.get("OPENSEARCH_ADMIN_PASS")
        or os.environ.get("OPENSEARCH_ADMIN_PASSWORD")
        or "admin"
    )
    parser = argparse.ArgumentParser(
        description="Index an ir_datasets corpus into OpenSearch with SPLADE expansions."
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
        default="dbpedia_entity_splade",
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
        default=200,
        help="Number of documents per bulk request (default: %(default)s).",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=8,
        help="Number of documents per SPLADE encoding batch (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default="naver/splade-cocondenser-ensembledistil",
        help="SPLADE model name (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top SPLADE tokens to keep (default: %(default)s).",
    )
    parser.add_argument(
        "--max-doc-length",
        type=int,
        default=512,
        help="Maximum tokens per document (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for SPLADE encoding, e.g. 'cpu' or 'cuda' (default: auto).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Index only the first N documents (default: all).",
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
                "splade_text": {"type": "text"},
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


def prepare_doc_source(args: argparse.Namespace) -> Tuple[Iterable, Optional[int], str]:
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
        label = f"{args.hf_name}/{args.hf_config}:{args.hf_split}"
        return dataset, doc_count, label

    dataset = ir_datasets.load(args.dataset)
    doc_count = dataset.docs_count()
    return dataset, doc_count, args.dataset


def _get_value(obj: Any, field: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value if item is not None]
        return "\n".join(part for part in parts if part)
    return str(value)


def _extract_title_text(doc: Any) -> Tuple[str, str]:
    title_fields = ("title", "source_title", "topic")
    text_fields = ("text", "premises_texts", "conclusion", "source_text")

    title = ""
    for field in title_fields:
        value = _normalize_text(_get_value(doc, field)).strip()
        if value:
            title = value
            break

    text_parts = []
    for field in text_fields:
        value = _normalize_text(_get_value(doc, field)).strip()
        if value:
            text_parts.append(value)
    text = "\n".join(text_parts)
    return title, text


def iter_docs_ir(dataset) -> Iterator[Mapping[str, Any]]:
    for doc in dataset.docs_iter():
        docid = getattr(doc, "doc_id", None)
        if docid is None:
            raise ValueError("Dataset document is missing 'doc_id'.")
        title, text = _extract_title_text(doc)
        yield {"docid": docid, "title": title, "text": text}


def iter_docs_hf(dataset) -> Iterator[Mapping[str, Any]]:
    for doc in dataset:
        docid = (
            doc.get("doc_id")
            or doc.get("docid")
            or doc.get("_id")
            or doc.get("id")
        )
        if docid is None:
            raise ValueError("Hugging Face document is missing an id field.")
        title, text = _extract_title_text(doc)
        yield {"docid": docid, "title": title, "text": text}


def build_actions(
    args: argparse.Namespace,
    dataset,
    encoder: SpladeEncoder,
) -> Iterator[Mapping[str, Any]]:
    if args.source == "hf":
        doc_iter = iter_docs_hf(dataset)
    else:
        doc_iter = iter_docs_ir(dataset)

    buffer: List[Mapping[str, Any]] = []
    total = 0
    for doc in doc_iter:
        buffer.append(doc)
        if len(buffer) >= args.encode_batch_size:
            yield from _flush_buffer(args, encoder, buffer)
            total += len(buffer)
            buffer = []
            if total % 1000 == 0:
                logging.info("Prepared %s documents.", total)
        if args.max_docs and total + len(buffer) >= args.max_docs:
            break

    if buffer:
        yield from _flush_buffer(args, encoder, buffer)


def _flush_buffer(
    args: argparse.Namespace,
    encoder: SpladeEncoder,
    buffer: List[Mapping[str, Any]],
) -> Iterator[Mapping[str, Any]]:
    texts = []
    for doc in buffer:
        title = doc.get("title", "")
        text = doc.get("text", "")
        combined = f"{title}\n{text}".strip()
        texts.append(combined)
    splade_texts = encoder.encode_batch(texts)

    for doc, splade_text in zip(buffer, splade_texts):
        yield {
            "_index": args.index,
            "_id": doc["docid"],
            "_source": {
                "docid": doc["docid"],
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "splade_text": splade_text,
            },
        }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    dataset, doc_count, source_label = prepare_doc_source(args)
    logging.info("Loaded dataset source '%s'.", source_label)
    if doc_count is not None:
        logging.info("Dataset contains %s documents.", doc_count)

    client = create_client(args)
    ensure_index(client, args)

    encoder = SpladeEncoder(
        model_name=args.model,
        device=device,
        max_doc_length=args.max_doc_length,
        top_k=args.top_k,
    )

    logging.info(
        "Indexing into '%s' with bulk batch size %s and encode batch size %s.",
        args.index,
        args.batch_size,
        args.encode_batch_size,
    )

    helpers.bulk(
        client,
        build_actions(args, dataset, encoder),
        chunk_size=args.batch_size,
        request_timeout=120,
    )
    logging.info("Finished indexing '%s'.", args.index)


if __name__ == "__main__":
    main()
