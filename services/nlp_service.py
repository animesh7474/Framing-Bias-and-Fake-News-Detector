"""
nlp_service.py — Advanced AI-powered NLP extraction pipeline for text.
Domain: NLP + AI in Cybersecurity
"""

import os
import json
import spacy
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
from config import SECURITY_LOG
from logger import get_logger

log = get_logger("nlp_service")

class NLPPipeline:
    """
    Analyzes text for propaganda patterns, adversarial injection,
    manipulation risk, and true subjectivity using advanced NLP models.
    """

    def __init__(self):
        os.makedirs(os.path.dirname(SECURITY_LOG) if os.path.dirname(SECURITY_LOG) else ".", exist_ok=True)
        # Load spaCy English model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except BaseException as e:
            log.error(f"Failed to load spacy model: {e}. Attempting fallback.")
            self.nlp = None

        self.vader = SentimentIntensityAnalyzer()
        self.propaganda_keywords = {
            "destroy", "catastrophe", "evil", "corrupt", "traitor", "radical",
            "fake", "hoax", "lie", "fraud", "brainwash", "manipulate", "agenda",
            "secret", "hidden", "conspiracy", "exposed", "truth", "wake up",
            "sheeple", "puppet", "regime", "propaganda", "enemy", "danger"
        }

    def _analyze_sentiment(self, text: str) -> dict:
        scores = self.vader.polarity_scores(text)
        return {
            "compound": scores['compound'],
            "extremity": abs(scores['compound'])
        }

    def _analyze_syntax(self, doc) -> dict:
        if not doc:
            return {"subjectivity_ratio": 0.5, "adjective_density": 0.0}
        
        adjectives = sum(1 for t in doc if t.pos_ == "ADJ")
        adverbs = sum(1 for t in doc if t.pos_ == "ADV")
        nouns = sum(1 for t in doc if t.pos_ == "NOUN")
        verbs = sum(1 for t in doc if t.pos_ == "VERB")
        
        total_words = len([t for t in doc if not t.is_punct])
        if total_words == 0:
            return {"subjectivity_ratio": 0.0, "adjective_density": 0.0}

        desc_count = adjectives + adverbs
        fact_count = nouns + verbs
        subj_ratio = desc_count / max((desc_count + fact_count), 1)
        adj_density = desc_count / total_words

        return {
            "subjectivity_ratio": min(subj_ratio * 1.5, 1.0),
            "adjective_density": adj_density
        }

    def _analyze_readability(self, text: str) -> dict:
        try:
            grade_level = textstat.flesch_kincaid_grade(text)
            reading_ease = textstat.flesch_reading_ease(text)
            is_simplistic = 1.0 if grade_level < 6 else 0.0
            return {
                "grade_level": grade_level,
                "reading_ease": reading_ease,
                "simplistic_penalty": is_simplistic
            }
        except Exception:
            return {"grade_level": 8, "reading_ease": 60, "simplistic_penalty": 0.0}

    def _analyze_propaganda(self, text: str) -> float:
        lower = text.lower()
        hits = sum(1 for kw in self.propaganda_keywords if kw in lower)
        return min(hits / max(len(self.propaganda_keywords) * 0.1, 1), 1.0)

    def analyze(self, text: str) -> dict:
        doc = self.nlp(text) if self.nlp else None
        sentiment_data = self._analyze_sentiment(text)
        syntax_data = self._analyze_syntax(doc)
        readability_data = self._analyze_readability(text)
        propaganda_score = self._analyze_propaganda(text)

        # ---------------------------------------------------------------------
        # 1. Manipulation Score Calculation (0.0 to 1.0)
        # ---------------------------------------------------------------------
        manipulation_raw = (
            (sentiment_data["extremity"] * 0.30) +
            (min(syntax_data["adjective_density"] * 3, 1.0) * 0.30) + 
            (propaganda_score * 0.30) +
            (readability_data["simplistic_penalty"] * 0.10)
        )
        manipulation_score = min(manipulation_raw * 1.5, 1.0) 

        threat_level = "HIGH" if manipulation_score >= 0.6 else "MEDIUM" if manipulation_score >= 0.3 else "LOW"

        # ---------------------------------------------------------------------
        # 2. True Bias Score Calculation (0.0 to 1.0)
        # ---------------------------------------------------------------------
        bias_raw = (
            (syntax_data["subjectivity_ratio"] * 0.40) +
            (sentiment_data["extremity"] * 0.30) +
            (propaganda_score * 0.30)
        )
        bias_score = min(bias_raw * 1.5, 1.0)

        # ---------------------------------------------------------------------
        # 3. Security Metrics (Cyber Threats)
        # ---------------------------------------------------------------------
        # Heuristic for adversarial/suspicious chars
        non_ascii = sum(1 for c in text if ord(c) > 127)
        weird_punc = text.count('!') + text.count('?') + text.count('$')
        adversarial_score = min((non_ascii * 0.1) + (weird_punc * 0.05), 1.0)

        # Heuristic for repetition
        words = text.lower().split()
        unique_words = set(words)
        repetition_score = 1.0 - (len(unique_words) / max(len(words), 1)) if words else 0.0

        result = {
            "text_snippet": text[:80],
            "threat_level": threat_level,
            "manipulation_score": round(manipulation_score, 4),
            "true_bias_score": round(bias_score, 4),
            "breakdown": {
                "propaganda_keywords": round(propaganda_score, 4),
                "emotional_language": round(sentiment_data["extremity"], 4),
                "adversarial_chars": round(adversarial_score, 4),
                "repetition_patterns": round(repetition_score, 4),
                "subjectivity_ratio": round(syntax_data["subjectivity_ratio"], 4),
                "reading_grade_level": round(readability_data["grade_level"], 1)
            },
            "flagged_words": [kw for kw in self.propaganda_keywords if kw in text.lower()],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        self._log_threat(result)
        log.info(f"NLP Service: {threat_level} (Manip: {manipulation_score:.2f}, Bias: {bias_score:.2f})")
        return result

    def _log_threat(self, result: dict):
        logs = []
        if os.path.exists(SECURITY_LOG):
            try:
                with open(SECURITY_LOG, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
        logs.append(result)
        with open(SECURITY_LOG, "w", encoding="utf-8") as f:
            json.dump(logs[-500:], f, indent=2)
