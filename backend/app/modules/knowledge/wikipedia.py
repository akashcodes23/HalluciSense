"""
Wikipedia Knowledge Source Adapter.
Provides search and snippet extraction from Wikipedia for Pillar 1 factual grounding.
"""
import wikipedia
from typing import List, Optional

class WikipediaKnowledgeSource:
    """
    Adapter for the python-wikipedia library to retrieve factual snippets.
    """
    
    def __init__(self, lang: str = "en", max_results: int = 3):
        wikipedia.set_lang(lang)
        self.max_results = max_results

    def retrieve(self, query: str) -> List[dict]:
        """
        Search Wikipedia and return page summaries as evidence snippets.
        Since wikipedia API calls are blocking, this should be run in a threadpool or Celery task.
        """
        evidence = []
        try:
            # Get matching page titles
            search_results = wikipedia.search(query, results=self.max_results)
            
            for title in search_results:
                try:
                    # fetch summary (auto_suggest=False to strictly fetch the matched title)
                    summary = wikipedia.summary(title, auto_suggest=False, sentences=3)
                    page = wikipedia.page(title, auto_suggest=False)
                    evidence.append({
                        "source_name": f"Wikipedia: {title}",
                        "source_url": page.url,
                        "snippet": summary
                    })
                except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                    # Skip disambiguation or missing pages for now to avoid stalling
                    continue
                    
        except Exception as e:
            # Handle rate limits, network errors, etc. safely
            print(f"Wikipedia search failed for query '{query}': {e}")
            
        return evidence
