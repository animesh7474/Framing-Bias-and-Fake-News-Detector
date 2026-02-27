"""
llm_client.py — LLM support layer using Groq (LLaMA 3.3 70B, free tier).
Domain: NLP + AI in Cybersecurity
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
log = get_logger("llm_client")

SYSTEM_PROMPT = """You are an expert media analyst and fact-checker specializing in detecting news framing bias, fake news, emotional manipulation, and hidden agendas in media. 

You will be given:
1. A news text submitted by a user
2. ML model scores already computed (frame classification, manipulation scores)
3. Related real news articles fetched from the web for context

Your job is to produce a structured JSON analysis report. Be specific, analytical, and reference the related news articles when relevant.

IMPORTANT: Always return ONLY valid JSON. No markdown, no extra text outside the JSON."""

ANALYSIS_PROMPT = """Analyze this news text for bias, manipulation, and factual accuracy:

NEWS TEXT:
"{text}"

ML MODEL SCORES (use these as primary signals, your job is to explain and enrich them):
- Detected Frame: {frame} (confidence: {confidence:.1%})
- Manipulation Score: {manipulation_score:.1%} | Threat Level: {threat_level}
- Detected Propaganda Keywords: {flagged_words}
- Frame Confidence Breakdown: {all_confidences}

RELATED REAL NEWS ARTICLES (for factual cross-referencing):
{news_context}

Respond ONLY with this exact JSON structure:
{{
  "fake_news_likelihood": <integer 0-100>,
  "framing_bias_score": <integer 0-100>,
  "emotional_manipulation_score": <integer 0-100>,
  "agenda_invocation_score": <integer 0-100>,
  "verdict": "<one concise sentence summarizing the overall assessment>",
  "framing_bias_explanation": "<2-3 sentences explaining what frame is being pushed and how>",
  "fake_news_analysis": "<2-3 sentences on factual accuracy, citing related news if available>",
  "emotional_manipulation_analysis": "<2-3 sentences on emotional language, tone, triggers>",
  "agenda_detection": "<2-3 sentences on what narrative or agenda this text may be pushing>",
  "factual_gaps": "<what important facts, context, or perspectives are missing or distorted>",
  "related_news_comparison": "<how this text compares to the real news articles found>",
  "overall_risk_level": "<LOW | MEDIUM | HIGH | CRITICAL>"
}}"""


class LLMClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            log.warning("GROQ_API_KEY not set. Add it to .env file.")
        self.client = Groq(api_key=api_key) if api_key else None
        self.model = "llama-3.3-70b-versatile"

    def analyze(
        self,
        text: str,
        ml_results: dict,
        news_context: str,
    ) -> dict:
        """
        Sends the combined context (text + ML scores + news) to the LLM.
        Returns structured analysis dict.
        """
        if not self.client:
            log.warning("No LLM client. Returning ML-only fallback.")
            return self._fallback(ml_results)

        prompt = ANALYSIS_PROMPT.format(
            text=text[:1500],
            frame=ml_results.get("predicted_frame", "Unknown"),
            confidence=ml_results.get("confidence", 0),
            manipulation_score=ml_results.get("threat_analysis", {}).get("manipulation_score", 0),
            threat_level=ml_results.get("threat_analysis", {}).get("threat_level", "LOW"),
            flagged_words=ml_results.get("threat_analysis", {}).get("flagged_words", []),
            all_confidences=ml_results.get("all_confidences", {}),
            news_context=news_context[:2000],
        )

        try:
            log.info(f"Calling LLM for: {text[:50]}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)
            log.info(f"LLM analysis complete. Risk: {result.get('overall_risk_level','?')}")
            return result
        except Exception as e:
            log.error(f"LLM call failed: {e}")
            return self._fallback(ml_results)

    def _fallback(self, ml_results: dict) -> dict:
        """Returns a basic analysis when LLM is unavailable."""
        threat = ml_results.get("threat_analysis", {})
        ms = threat.get("manipulation_score", 0)
        frame = ml_results.get("predicted_frame", "Unknown")
        conf = ml_results.get("confidence", 0)
        level = threat.get("threat_level", "LOW")

        bias_score = int(conf * 80)
        manip_score = int(ms * 100)
        fake_score = min(int(ms * 60 + (1 - conf) * 20), 100)
        agenda_score = int(ms * 70)

        return {
            "fake_news_likelihood": fake_score,
            "framing_bias_score": bias_score,
            "emotional_manipulation_score": manip_score,
            "agenda_invocation_score": agenda_score,
            "verdict": f"Text appears to use a {frame} frame with {level.lower()} manipulation risk. Add a Groq API key for detailed LLM analysis.",
            "framing_bias_explanation": f"The ML model detected a {frame} framing with {conf:.1%} confidence. This suggests the text is using language and context typical of {frame.lower()}-focused narratives.",
            "fake_news_analysis": "LLM unavailable — add GROQ_API_KEY to .env for AI-powered fact-checking against live news.",
            "emotional_manipulation_analysis": f"Manipulation score: {ms:.1%}. {'Elevated emotional language detected.' if ms > 0.3 else 'Relatively neutral tone detected.'}",
            "agenda_detection": "LLM unavailable for agenda analysis. The manipulation score provides a proxy signal.",
            "factual_gaps": "LLM unavailable — cannot cross-reference with live news sources without API key.",
            "related_news_comparison": "LLM unavailable.",
            "overall_risk_level": level,
        }


if __name__ == "__main__":
    client = LLMClient()
    result = client._fallback({
        "predicted_frame": "Security",
        "confidence": 0.9,
        "all_confidences": {"Security": 0.9, "Political": 0.05},
        "threat_analysis": {"manipulation_score": 0.3, "threat_level": "MEDIUM", "flagged_words": []}
    })
    print(json.dumps(result, indent=2))
