from duckduckgo_search import DDGS

def test_news():
    query = "latest news about Apple iPhone"
    print(f"Searching for: {query}")
    results = []
    try:
        with DDGS() as ddgs:
            # In 8.x it should be news()
            for r in ddgs.news(keywords=query, region="wt-wt", safesearch="off", max_results=5):
                results.append(r)
        print(f"Found {len(results)} results")
        for r in results:
            print(f"- {r.get('title')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_news()
