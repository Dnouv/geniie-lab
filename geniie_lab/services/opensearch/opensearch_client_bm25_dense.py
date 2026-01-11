# Standard library
import re
from typing import List, Optional, Union

# Third-party libraries
import ir_datasets
import numpy as np
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

# Local application imports
from geniie_lab.dataclasses.serp import FullText, SearchResultItem, Serp
from geniie_lab.dataclasses.setting import Error


class OpenSearchClientBM25Dense:
    """
    BM25 retrieval followed by dense bi-encoder re-ranking.
    """

    def __init__(
        self,
        index_name: str,
        dataset_name: str,
        encode_model: str,
        host: str = "localhost",
        port: int = 9200,
        http_auth: Optional[tuple[str, str]] = None,
        use_ssl: bool = True,
        rerank_top_k: Optional[int] = None,
        max_doc_chars: Optional[int] = None,
    ):
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_compress=True,
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        self.index_name = index_name
        self.dataset = ir_datasets.load(dataset_name)
        model_name = encode_model or "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)
        self.rerank_top_k = rerank_top_k or 100
        self.max_doc_chars = max_doc_chars or 2000

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        return " ".join(text.splitlines())

    def fetch_fulltext(self, docid: str) -> Union[FullText, Error]:
        try:
            self.docstore = self.dataset.docs_store()
            text = self.docstore.get(docid).text
            return FullText(
                docid=docid,
                text=self.clean_text(text)
            )
        except Exception as e:
            return Error(error_text=str(e))

    def _search_initial(self, query: str, size: int) -> dict:
        search_body = {
            "from": 0,
            "size": size,
            "query": {"multi_match": {"query": query, "fields": ["title", "text"]}},
            "highlight": {"fields": {"text": {"type": "plain", "fragment_size": 150, "number_of_fragments": 1}}},
        }
        return self.client.search(index=self.index_name, body=search_body)

    def _prepare_doc_text(self, src: dict) -> str:
        text = f"{src.get('title', '')} {src.get('text', '')}"
        return self.clean_text(text)[: self.max_doc_chars]

    def _rerank(self, query: str, hits: List[dict]) -> List[dict]:
        if not hits:
            return []
        texts = [self._prepare_doc_text(hit.get("_source", {})) for hit in hits]
        query_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        doc_embs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        scores = np.dot(doc_embs, query_emb)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return [{"hit": hit, "score": score} for hit, score in ranked]

    def search_index_with_snippets(
        self,
        query: str,
        start: int = 0,
        size: int = 10
    ) -> Serp:
        top_k = max(self.rerank_top_k, start + size)
        response = self._search_initial(query, size=top_k)
        total_hits = response.get("hits", {}).get("total", {}).get("value", 0)
        hits = response.get("hits", {}).get("hits", [])
        if total_hits == 0 or not hits:
            return Serp(hits=0, results=[])

        ranked = self._rerank(query, hits)
        slice_ranked = ranked[start:start + size]

        items: List[SearchResultItem] = []
        for idx, item in enumerate(slice_ranked, start=1):
            hit = item["hit"]
            src = hit.get("_source", {})
            raw_snippets = hit.get("highlight", {}).get("text", [src.get("text", "")[:150]])
            snippet_text = " ... ".join(self.clean_text(s) for s in raw_snippets)
            items.append(SearchResultItem(
                ranking=start + idx,
                docid=src.get("docid"),
                title=self.clean_text(src.get("title", "No Title")),
                snippet=snippet_text
            ))
        return Serp(hits=total_hits, results=items)

    def search_docids(
        self,
        query: str,
        start: int = 0,
        size: int = 100
    ) -> list[str]:
        top_k = max(self.rerank_top_k, start + size)
        response = self._search_initial(query, size=top_k)
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            return []
        ranked = self._rerank(query, hits)
        slice_ranked = ranked[start:start + size]
        return [item["hit"].get("_source", {}).get("docid") for item in slice_ranked if item.get("hit")]
