# Standard library
import re
from collections import Counter
from typing import List, Optional, Union

# Third-party libraries
import ir_datasets
from opensearchpy import OpenSearch

# Local application imports
from geniie_lab.dataclasses.serp import FullText, SearchResultItem, Serp
from geniie_lab.dataclasses.setting import Error


class OpenSearchClientBM25RM3:
    """
    BM25 with pseudo-relevance feedback (RM3-style query expansion).
    """

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "this", "these", "those", "or",
        "not", "but", "we", "they", "you", "your", "i", "our", "their"
    }

    def __init__(
        self,
        index_name: str,
        dataset_name: str,
        host: str = "localhost",
        port: int = 9200,
        http_auth: Optional[tuple[str, str]] = None,
        use_ssl: bool = True,
        prf_docs: Optional[int] = None,
        prf_terms: Optional[int] = None,
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
        self.prf_docs = prf_docs or 10
        self.prf_terms = prf_terms or 20

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

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _collect_feedback_terms(self, query: str, hits: List[dict]) -> List[str]:
        query_terms = set(self._tokenize(query))
        counter: Counter[str] = Counter()
        for hit in hits:
            src = hit.get("_source", {})
            text = f"{src.get('title', '')} {src.get('text', '')}"
            for tok in self._tokenize(text):
                if tok in self._STOPWORDS:
                    continue
                if tok in query_terms:
                    continue
                if len(tok) < 3:
                    continue
                counter[tok] += 1
        return [term for term, _ in counter.most_common(self.prf_terms)]

    def _expand_query(self, query: str, hits: List[dict]) -> str:
        terms = self._collect_feedback_terms(query, hits)
        if not terms:
            return query
        return f"{query} " + " ".join(terms)

    def _search(
        self,
        query: str,
        start: int,
        size: int,
        with_highlight: bool = True,
    ) -> dict:
        search_body = {
            "from": start,
            "size": size,
            "query": {"multi_match": {"query": query, "fields": ["title", "text"]}},
        }
        if with_highlight:
            search_body["highlight"] = {
                "fields": {"text": {"type": "plain", "fragment_size": 150, "number_of_fragments": 1}}
            }
        return self.client.search(index=self.index_name, body=search_body)

    def search_index_with_snippets(
        self,
        query: str,
        start: int = 0,
        size: int = 10
    ) -> Serp:
        initial = self._search(query, start=0, size=self.prf_docs, with_highlight=False)
        initial_hits = initial.get("hits", {}).get("hits", [])
        expanded_query = self._expand_query(query, initial_hits)

        response = self._search(expanded_query, start=start, size=size, with_highlight=True)
        total_hits = response.get("hits", {}).get("total", {}).get("value", 0)
        if total_hits == 0:
            return Serp(hits=0, results=[])

        items: List[SearchResultItem] = []
        for idx, hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
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
        initial = self._search(query, start=0, size=self.prf_docs, with_highlight=False)
        initial_hits = initial.get("hits", {}).get("hits", [])
        expanded_query = self._expand_query(query, initial_hits)

        search_body = {
            "from": start,
            "size": size,
            "query": {"multi_match": {"query": expanded_query, "fields": ["title", "text"]}},
            "_source": ["docid"],
        }
        response = self.client.search(index=self.index_name, body=search_body)
        hits = response.get("hits", {}).get("hits", [])
        return [hit.get("_source", {}).get("docid") for hit in hits if hit.get("_source")]
