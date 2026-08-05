"""
Wikipedia Knowledge Source Adapter.
Provides search and snippet extraction from Wikipedia for Pillar 1 factual grounding.
"""
from typing import List, Optional
import requests
import structlog

logger = structlog.get_logger(__name__)

HEADERS = {
    "User-Agent": "HalluciSense/1.0 (https://hallucisense.app; contact@hallucisense.app)"
}


class WikipediaKnowledgeSource:
    """
    Adapter for Wikipedia REST/Action API to retrieve factual snippets with custom User-Agent.
    """
    
    def __init__(self, lang: str = "en", max_results: int = 3):
        self.lang = lang
        self.max_results = max_results
        self.api_url = f"https://{lang}.wikipedia.org/w/api.php"

    def retrieve(self, query: str) -> List[dict]:
        """
        Search Wikipedia and return page summaries as evidence snippets.
        """
        evidence = []
        if not query or not query.strip():
            return evidence

        try:
            # 1. Action API search for matching page titles
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": self.max_results,
            }
            resp = requests.get(self.api_url, params=params, headers=HEADERS, timeout=4.0)
            if resp.status_code != 200:
                logger.warning("wikipedia_search_status_error", status_code=resp.status_code)
                return evidence

            data = resp.json()
            search_items = data.get("query", {}).get("search", [])

            for item in search_items:
                title = item.get("title")
                pageid = item.get("pageid")
                if not title:
                    continue

                # 2. Fetch page summary extract
                ext_params = {
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "exsentences": 4,
                    "titles": title,
                    "format": "json",
                }
                ext_resp = requests.get(self.api_url, params=ext_params, headers=HEADERS, timeout=4.0)
                if ext_resp.status_code == 200:
                    pages = ext_resp.json().get("query", {}).get("pages", {})
                    for pid, pdata in pages.items():
                        snippet = pdata.get("extract", "").strip()
                        if snippet:
                            evidence.append({
                                "source_name": f"Wikipedia: {title}",
                                "source_url": f"https://{self.lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                "snippet": snippet,
                            })

        except Exception as e:
            logger.warning("wikipedia_retrieve_exception", query=query, error=str(e))
            
        return evidence
