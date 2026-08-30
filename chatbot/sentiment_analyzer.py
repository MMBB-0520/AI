"""
BookMate Chatbot - Supporting NLP Component: Sentiment & Frustration Analyzer
=============================================================================

Purpose:
Lightweight rule-based and lexicon-aware emotion/frustration detector.
Used for:
- Detecting guest frustration / dissatisfaction to trigger empathetic chatbot apologies.
- Real-time sentiment metrics displayed in the live assistant interface.
"""

import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception:
        pass


class SentimentAnalyzer:
    """
    Analyzes sentiment and frustration level of user utterances.
    """

    def __init__(self):
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception:
            self.sia = None

        # Domain-specific frustration indicators
        self.frustration_keywords = [
            "terrible", "awful", "horrible", "worst", "hate", "angry", "furious",
            "useless", "scam", "disaster", "bad service", "unacceptable",
            "ridiculous", "poor", "waste of money", "disappointed", "disappointing",
            "annoying", "annoyed", "frustrated", "frustrating"
        ]

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment and frustration of input text.

        Returns:
            {
                "sentiment": "positive" | "negative" | "neutral",
                "score": float (-1.0 to 1.0),
                "is_frustrated": bool,
                "details": {
                    "neg": float,
                    "neu": float,
                    "pos": float,
                    "compound": float
                }
            }
        """
        if not text or not isinstance(text, str):
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "is_frustrated": False,
                "details": {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
            }

        text_lower = text.lower().strip()

        # 1. Lexicon-based VADER score
        if self.sia:
            scores = self.sia.polarity_scores(text)
            compound = scores.get("compound", 0.0)
        else:
            scores = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
            compound = 0.0

        # 2. Determine sentiment label
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        # 3. Frustration / Escalation detection
        is_frustrated = False
        if compound <= -0.35:
            is_frustrated = True
        elif any(re.search(rf"\b{re.escape(kw)}\b", text_lower) for kw in self.frustration_keywords):
            is_frustrated = True
            if label == "neutral":
                label = "negative"
                compound = -0.5

        return {
            "sentiment": label,
            "score": round(compound, 2),
            "is_frustrated": is_frustrated,
            "details": scores
        }


# Global singleton instance
_default_analyzer = SentimentAnalyzer()


def analyze_sentiment(text: str) -> dict:
    """
    Convenience function to analyze user text sentiment.
    """
    return _default_analyzer.analyze(text)
