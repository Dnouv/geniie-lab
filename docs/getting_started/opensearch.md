# How to set up an OpenSearch client

See [OpenSearch Documentation](https://docs.opensearch.org/docs/latest/about/) to learn how to index the corpus of your test collection.

Once you indexed the corpus using opensearch, then set `ToolDescription` in `ExperimentalSettings` in the runner scripts in `scripts` folder. You can set `port` and `description` too.

```
tools=[
    ToolDescription(
        name="opensearch",
        ranking_model="bm25",
        index_name="aquaint_bm25",
        port=9200,
        description="It allows you to perform searches using keywords only and employs the BM25 ranking model to order results.",
    )
```

## How to compare multiple ranking models

You can list multiple `ToolDescription` in `ExperimentalSettings`. Then the experiment will repeat the whole process across the ranking models.

The following example mixes classic and neural rankers. `bm25_rm3`, `bm25_dense`, and `bm25_cross_encoder` all use the same BM25 index for candidate generation, so you do not need a new index to try them.

```
tools=[
    ToolDescription(
        name="opensearch",
        ranking_model="bm25",
        index_name="aquaint_bm25",
        port=9200,
        description="It allows you to perform searches using keywords only and employs the BM25 ranking model to order results.",
    ),
    ToolDescription(
        name="opensearch",
        ranking_model="bm25_rm3",
        index_name="aquaint_bm25",
        port=9200,
        prf_docs=10,
        prf_terms=20,
        description="BM25 with RM3-style pseudo relevance feedback.",
    ),
    ToolDescription(
        name="opensearch",
        ranking_model="bm25_dense",
        encode_model="sentence-transformers/all-MiniLM-L6-v2",
        index_name="aquaint_bm25",
        port=9200,
        rerank_top_k=100,
        description="BM25 candidate generation with dense bi-encoder reranking.",
    ),
    ToolDescription(
        name="opensearch",
        ranking_model="bm25_cross_encoder",
        encode_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        index_name="aquaint_bm25",
        port=9200,
        rerank_top_k=100,
        description="BM25 candidate generation with cross-encoder reranking.",
    ),
    ToolDescription(
        name="opensearch",
        ranking_model="dpr",
        encode_model="sentence-transformers/msmarco-distilbert-base-tas-b",
        index_name="aquaint_dpr",
        port=9200,
        description="It allows you to perform searches using keywords only and employs the DPR ranking model to order results.",
    ),
    ToolDescription(
        name="opensearch",
        ranking_model="splade",
        encode_model="naver/splade-cocondenser-ensembledistil",
        index_name="aquaint_splade",
        port=9200,
        description="It allows you to perform searches using keywords only and employs the Splade ranking model to order results.",
    ),
```

## SPLADE indexing (separate index)

SPLADE requires a dedicated index with a `splade_text` field. Use `scripts/index_splade.py` to build it.

```
python scripts/index_splade.py \
    --dataset beir/dbpedia-entity \
    --index dbpedia_entity_splade \
    --host localhost --port 9200 --no-ssl \
    --batch-size 200 --encode-batch-size 8
```

Notes:
- Use the same `encode_model` in `ToolDescription` as the `--model` used for indexing.
- For quick sanity checks, add `--max-docs 10000`.
- SPLADE indexing is CPU/GPU intensive; it will run much faster on a GPU.
