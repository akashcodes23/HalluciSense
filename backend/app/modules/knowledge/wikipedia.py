"""
Wikipedia Knowledge Source Adapter.
Provides search and snippet extraction from Wikipedia for Pillar 1 factual grounding.

The adapter deliberately returns no evidence when Wikipedia cannot provide real
supporting material. Synthetic/fabricated evidence must never be introduced by
the retrieval layer because it can turn an unknown claim into a false positive.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import requests
import structlog

logger = structlog.get_logger(__name__)

HEADERS = {
    "User-Agent": "HalluciSense/1.0 (https://hallucisense.app; contact@hallucisense.app)"
}


class WikipediaKnowledgeSource:
    """Retrieve factual snippets from Wikipedia with bounded caching, session pooling and batching."""

    def __init__(self, lang: str = "en", max_results: int = 3, max_search_workers: int = 4):
        self.lang = lang
        self.max_results = max_results
        self.max_search_workers = max(1, max_search_workers)
        self.api_url = f"https://{lang}.wikipedia.org/w/api.php"
        from collections import OrderedDict
        import threading
        self.MAX_CACHE_ENTRIES = 512
        self._cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()
        self.last_metrics = self._empty_metrics()
        
        # Persistent HTTP connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=requests.adapters.Retry(
                total=2,
                backoff_factor=0.2,
                status_forcelist=[500, 502, 503, 504],
            ),
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0.0,
            "search_requests": 0,
            "extraction_requests": 0,
            "retrieved_pages": 0,
            "failed_queries": 0,
        }

    def _search(self, query: str) -> Tuple[str, List[str], Dict[str, str]]:
        """Search Wikipedia for page titles and contextual snippets; never fabricate evidence on failure."""
        import re
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": self.max_results,
        }

        try:
            resp = self.session.get(
                self.api_url,
                params=params,
                headers=HEADERS,
                timeout=(1.5, 3.0),
            )
            if resp.status_code != 200:
                logger.warning(
                    "wikipedia_search_status_error",
                    query=query,
                    status_code=resp.status_code,
                )
                return query, [], {}

            items = resp.json().get("query", {}).get("search", [])
            titles = [item.get("title") for item in items if item.get("title")]
            snippets = {
                item.get("title"): re.sub(r'<[^>]+>', '', item.get("snippet", "")).strip()
                for item in items
                if item.get("title") and item.get("snippet")
            }

            # A second search is acceptable when the original query is a long
            # sentence and Wikipedia returns no useful page title. It still
            # produces only real Wikipedia titles; there is no synthetic fallback.
            if not titles and len(query.split()) > 4:
                topic = query
                for stop in ["The", "the", "is", "was", "approximately", "in", "vacuum", "once", "per", "year"]:
                    topic = topic.replace(stop, "")
                topic = topic.strip()
                if topic:
                    params["srsearch"] = topic
                    resp2 = self.session.get(
                        self.api_url,
                        params=params,
                        headers=HEADERS,
                        timeout=(1.5, 3.0),
                    )
                    if resp2.status_code == 200:
                        items2 = resp2.json().get("query", {}).get("search", [])
                        titles = [item.get("title") for item in items2 if item.get("title")]
                        snippets.update({
                            item.get("title"): re.sub(r'<[^>]+>', '', item.get("snippet", "")).strip()
                            for item in items2
                            if item.get("title") and item.get("snippet")
                        })

            return query, titles, snippets
        except Exception as exc:
            logger.warning("wikipedia_search_exception", query=query, error=str(exc))
            return query, [], {}

    def _extract_titles(self, titles: List[str]) -> Dict[str, dict]:
        """Extract multiple Wikipedia page summaries in one API request."""
        if not titles:
            return {}

        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": "|".join(titles),
            "format": "json",
        }

        try:
            resp = self.session.get(
                self.api_url,
                params=params,
                headers=HEADERS,
                timeout=(1.5, 3.0),
            )
            if resp.status_code != 200:
                logger.warning(
                    "wikipedia_batch_extract_status_error",
                    status_code=resp.status_code,
                    title_count=len(titles),
                )
                return {}

            pages = resp.json().get("query", {}).get("pages", {})
            extracted: Dict[str, dict] = {}
            for pdata in pages.values():
                title = pdata.get("title")
                snippet = pdata.get("extract", "").strip()
                if title and snippet:
                    extracted[title] = {
                        "source_name": f"Wikipedia: {title}",
                        "source_url": f"https://{self.lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "snippet": snippet,
                    }
            return extracted
        except Exception as exc:
            logger.warning(
                "wikipedia_batch_extract_exception",
                error=str(exc),
                title_count=len(titles),
            )
            return {}

    def retrieve_batch(self, queries: List[str]) -> Dict[str, List[dict]]:
        """Retrieve evidence for multiple claims with parallel search + batched extraction."""
        metrics = self._empty_metrics()
        normalized = []
        for query in queries:
            clean = (query or "").strip()
            if clean:
                normalized.append(clean)

        if not normalized:
            self.last_metrics = metrics
            return {}

        unique_queries = list(dict.fromkeys(normalized))
        results: Dict[str, List[dict]] = {}
        misses: List[str] = []

        for query in unique_queries:
            key = query.lower()
            with self._cache_lock:
                if key in self._cache:
                    results[query] = list(self._cache[key])
                    metrics["cache_hits"] += 1
                    continue
            misses.append(query)
            metrics["cache_misses"] += 1

        if misses:
            workers = min(self.max_search_workers, len(misses))
            search_results = []
            try:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(self._search, query) for query in misses]
                    for future in as_completed(futures, timeout=6.0):
                        try:
                            search_results.append(future.result())
                            metrics["search_requests"] += 1
                        except Exception:
                            pass
            except Exception:
                pass

            titles_by_query: Dict[str, List[str]] = {}
            snippets_by_title: Dict[str, str] = {}
            for item in search_results:
                q, t_list, s_map = item
                titles_by_query[q] = t_list
                snippets_by_title.update(s_map)

            unique_titles = list(dict.fromkeys(
                title
                for titles in titles_by_query.values()
                for title in titles
            ))

            # Wikipedia supports multiple titles in a single extraction request.
            # Keep the batch bounded for API compatibility and predictable payloads.
            extracted_by_title: Dict[str, dict] = {}
            for start in range(0, len(unique_titles), 50):
                batch_titles = unique_titles[start:start + 50]
                extracted_by_title.update(self._extract_titles(batch_titles))
                metrics["extraction_requests"] += 1

            for query in misses:
                evidence = []
                for title in titles_by_query.get(query, []):
                    item = extracted_by_title.get(title)
                    if item:
                        item_copy = dict(item)
                        search_snip = snippets_by_title.get(title)
                        if search_snip and len(search_snip) > 15:
                            intro_snip = item_copy.get("snippet", "")
                            if search_snip not in intro_snip:
                                item_copy["snippet"] = f"{intro_snip}\n{search_snip}".strip()
                        item_copy["retrieval_query"] = query
                        item_copy["title"] = title
                        evidence.append(item_copy)

                results[query] = evidence
                if evidence or (query in titles_by_query and not titles_by_query[query]):
                    with self._cache_lock:
                        if len(self._cache) >= self.MAX_CACHE_ENTRIES:
                            self._cache.popitem(last=False)
                        self._cache[query.lower()] = evidence
                if evidence:
                    metrics["retrieved_pages"] += len(evidence)
                else:
                    metrics["failed_queries"] += 1

        total_queries = metrics["cache_hits"] + metrics["cache_misses"]
        metrics["cache_hit_rate"] = round(
            metrics["cache_hits"] / total_queries,
            4,
        ) if total_queries else 0.0
        self.last_metrics = metrics
        return results

    def retrieve(self, query: str) -> List[dict]:
        """Backward-compatible single-query wrapper around :meth:`retrieve_batch`."""
        if not query or not query.strip():
            self.last_metrics = self._empty_metrics()
            return []
        return self.retrieve_batch([query.strip()]).get(query.strip(), [])
