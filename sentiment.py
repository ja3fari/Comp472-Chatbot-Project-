"""Sentiment analysis for the student support assistant.

Wraps a Hugging Face transformers pipeline to classify a message as
POSITIVE, NEUTRAL or NEGATIVE along with a confidence score.
"""

from dataclasses import dataclass


@dataclass
class SentimentResult:
    """Result of analysing one message: a label and its confidence."""
    label: str
    score: float

    def __str__(self):
        return f"{self.label} ({self.score:.2f})"


class SentimentAnalyzer:
    """Classifies text sentiment using a 3-class transformers model.

    The default distilbert model only knows POSITIVE/NEGATIVE, so we use a
    model that also has a NEUTRAL class so that plain questions are not
    flagged as negative.
    """

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(self):
        # Load the model once and reuse it for every message.
        self.model = self._load_model()

    @classmethod
    def _load_model(cls):
        try:
            from transformers import pipeline
            return pipeline("sentiment-analysis", model=cls.MODEL_NAME)
        except Exception as error:
            print(f"Could not load sentiment model ({error}). "
                  f"Using a basic keyword check instead.")
            return None

    def analyze(self, text):
        """Return a SentimentResult for the given text."""
        if not text or not text.strip():
            return SentimentResult("NEUTRAL", 1.0)

        if self.model is None:
            return self._keyword_check(text)

        try:
            prediction = self.model(text)[0]
            return SentimentResult(prediction["label"].upper(),
                                   float(prediction["score"]))
        except Exception as error:
            print(f"Sentiment analysis failed ({error}).")
            return self._keyword_check(text)

    def _keyword_check(self, text):
        """Fallback used only when the model cannot be loaded."""
        negative = {"terrible", "awful", "hate", "horrible", "frustrated",
                    "angry", "worst", "useless", "bad", "broken", "failed",
                    "annoyed", "disappointed", "upset"}
        positive = {"great", "thanks", "thank", "good", "excellent", "love",
                    "awesome", "happy", "perfect", "helpful", "nice"}

        words = [w.strip(".,!?").lower() for w in text.split()]
        neg = sum(w in negative for w in words)
        pos = sum(w in positive for w in words)

        if neg > pos:
            return SentimentResult("NEGATIVE", 0.95 if neg > 1 else 0.75)
        if pos > neg:
            return SentimentResult("POSITIVE", 0.95 if pos > 1 else 0.75)
        return SentimentResult("NEUTRAL", 0.60)


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    for text in ["I cannot access my account and this is terrible",
                 "Where is the registrar office?",
                 "Thank you so much, this was really helpful!"]:
        print(text, "->", analyzer.analyze(text))
