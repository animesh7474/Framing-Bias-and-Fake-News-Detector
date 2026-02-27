"""
news_fetcher.py — Fetches related live news using DuckDuckGo (no API key).
Domain: Computer Networks (web scraping / HTTP)
"""

from duckduckgo_search import DDGS
from logger import get_logger

log = get_logger("news_fetcher")


def fetch_related_news(query: str, max_results: int = 5) -> list[dict]:
    """
    Searches DuckDuckGo News for articles related to the input query.
    Returns a list of dicts: {title, url, body, source, date}
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append({
                    "title":  r.get("title", ""),
                    "url":    r.get("url", ""),
                    "body":   r.get("body", ""),
                    "source": r.get("source", ""),
                    "date":   r.get("date", ""),
                })
        log.info(f"Fetched {len(results)} news articles for: {query[:60]}")
    except Exception as e:
        log.warning(f"News fetch failed: {e}")
    return results


def format_news_for_llm(articles: list[dict]) -> str:
    """Formats fetched news into a compact string for the LLM prompt."""
    if not articles:
        return "No related news articles found."
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(
            f"[{i}] {a['title']}\n"
            f"    Source: {a['source']} | Date: {a['date']}\n"
            f"    {a['body'][:200]}..."
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    articles = fetch_related_news("India Pakistan military tension 2025")
    for a in articles:
        print(f"• {a['title']} ({a['source']})")
