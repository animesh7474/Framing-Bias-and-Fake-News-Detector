import requests
import json
url = "http://localhost:5000/api/analyze"
data = {"text": "Apple released its quarterly earnings report this morning, showing a 3% increase in total revenue year-over-year. The report cited strong iPhone sales in the Asian market."}
response = requests.post(url, json=data)
r = response.json()
llm = r.get("llm_analysis", {})
print("Bias:", llm.get("framing_bias_score"))
print("Fake Likelihood:", llm.get("fake_news_likelihood"))
print("Manipulation:", llm.get("emotional_manipulation_score"))
print("Is Fallback:", llm.get("is_fallback", False))
