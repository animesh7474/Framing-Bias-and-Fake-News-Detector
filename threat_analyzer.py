"""
threat_analyzer.py — AI-powered cybersecurity threat analysis for text.
Domain: AI in Cybersecurity
"""

import re
import json
import os
from datetime import datetime
from config import SECURITY_LOG, FRAME_LABELS
from logger import get_logger
import unicodedata

log = get_logger("threat_analyzer")


PROPAGANDA_KEYWORDS = [
    "destroy", "catastrophe", "evil", "corrupt", "traitor", "radical",
    "fake", "hoax", "lie", "fraud", "brainwash", "manipulate", "agenda",
    "secret", "hidden", "conspiracy", "exposed", "truth", "wake up",
    "sheeple", "puppet", "regime", "propaganda"
]

EMOTIONAL_INTENSIFIERS = [
    "always", "never", "everyone", "nobody", "absolutely", "completely",
    "totally", "definitely", "obviously", "clearly", "must", "will",
    "guarantee", "proven", "undeniable", "alarming", "shocking", "crisis",
    "danger", "urgent", "emergency", "terrible", "horrible", "devastating"
]


class ThreatAnalyzer:
    """
    Analyzes text for propaganda patterns, adversarial injection, and
    manipulation risk — simulating an AI-powered cybersecurity module.
    """

    def __init__(self):
        os.makedirs(os.path.dirname(SECURITY_LOG) if os.path.dirname(SECURITY_LOG) else ".", exist_ok=True)

    def _propaganda_score(self, text: str) -> float:
        lower = text.lower()
        hits = sum(1 for kw in PROPAGANDA_KEYWORDS if kw in lower)
        return min(hits / max(len(PROPAGANDA_KEYWORDS) * 0.2, 1), 1.0)

    def _emotional_score(self, text: str) -> float:
        lower = text.lower()
        words = lower.split()
        hits = sum(1 for w in words if w in EMOTIONAL_INTENSIFIERS)
        return min(hits / max(len(words) * 0.15, 1), 1.0)

    def _adversarial_score(self, text: str) -> float:
        """Detect homoglyph attacks and non-ASCII injection."""
        suspicious = 0
        for char in text:
            if ord(char) > 127:
                cat = unicodedata.category(char)
                if cat not in ("Zs",):  # Allow normal spaces
                    suspicious += 1
        return min(suspicious / max(len(text) * 0.1, 1), 1.0)

    def _repetition_score(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        unique = len(set(words))
        ratio = 1 - (unique / len(words))
        return min(ratio * 2, 1.0)

    def analyze(self, text: str) -> dict:
        prop  = self._propaganda_score(text)
        emot  = self._emotional_score(text)
        adv   = self._adversarial_score(text)
        rep   = self._repetition_score(text)

        # Weighted composite manipulation score
        manipulation_score = round(
            prop * 0.35 + emot * 0.30 + adv * 0.20 + rep * 0.15, 4
        )

        if manipulation_score >= 0.5:
            threat_level = "HIGH"
        elif manipulation_score >= 0.25:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"

        result = {
            "text_snippet": text[:80],
            "threat_level": threat_level,
            "manipulation_score": manipulation_score,
            "breakdown": {
                "propaganda_keywords": round(prop, 4),
                "emotional_language":  round(emot, 4),
                "adversarial_chars":   round(adv, 4),
                "repetition_patterns": round(rep, 4),
            },
            "flagged_words": [kw for kw in PROPAGANDA_KEYWORDS if kw in text.lower()],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        self._log_threat(result)
        log.info(f"Threat analysis: {threat_level} ({manipulation_score:.3f}) | {text[:40]}")
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
            json.dump(logs[-500:], f, indent=2)  # Keep last 500 entries


if __name__ == "__main__":
    ta = ThreatAnalyzer()
    samples = [
        "The government is absolutely destroying our economy with corrupt policies!",
        "The stock market showed moderate gains today.",
        "WAKE UP SHEEPLE! The hidden agenda will completely destroy our freedom!",
    ]
    for s in samples:
        r = ta.analyze(s)
        print(f"\n[{r['threat_level']}] Score: {r['manipulation_score']:.3f}")
        print(f"  Text: {r['text_snippet']}")
        print(f"  Flags: {r['flagged_words']}")
