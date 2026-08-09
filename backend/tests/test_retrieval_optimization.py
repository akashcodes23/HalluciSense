"""Regression tests for Phase 1 retrieval optimization."""

from app.modules.knowledge.wikipedia import WikipediaKnowledgeSource


def test_wikipedia_no_fabricated_fallback(monkeypatch):
    source = WikipediaKnowledgeSource()

    class Response:
        status_code = 503
        def json(self):
            return {}

    monkeypatch.setattr("app.modules.knowledge.wikipedia.requests.get", lambda *a, **k: Response())

    result = source.retrieve("a deliberately unavailable claim")
    assert result == []
    assert source.last_metrics["failed_queries"] == 1


def test_wikipedia_batch_uses_cache(monkeypatch):
    source = WikipediaKnowledgeSource(max_results=2)
    calls = []

    def fake_get(url, params=None, **kwargs):
        calls.append(params.copy())
        if params.get("action") == "query" and params.get("list") == "search":
            q = params["srsearch"]
            return type("R", (), {
                "status_code": 200,
                "json": lambda self: {"query": {"search": [{"title": q.title()}]}}
            })()
        pages = {
            str(i + 1): {"title": title, "extract": "Real Wikipedia evidence."}
            for i, title in enumerate(params["titles"].split("|"))
        }
        return type("R", (), {
            "status_code": 200,
            "json": lambda self: {"query": {"pages": pages}}
        })()

    monkeypatch.setattr("app.modules.knowledge.wikipedia.requests.get", fake_get)

    first = source.retrieve_batch(["alpha", "beta"])
    second = source.retrieve_batch(["alpha", "beta"])

    assert first["alpha"]
    assert first["beta"]
    assert second["alpha"] == first["alpha"]
    assert second["beta"] == first["beta"]
    assert source.last_metrics["cache_hits"] == 2
    assert source.last_metrics["cache_hit_rate"] == 1.0
    assert any(p.get("titles") == "Alpha|Beta" for p in calls if p.get("prop") == "extracts")
