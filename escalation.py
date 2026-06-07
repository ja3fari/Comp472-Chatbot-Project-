"""Escalation logic for the student support assistant.

If a student's message is strongly negative, the assistant recommends
contacting a human advisor instead of relying on the bot.
"""

from sentiment import SentimentResult

ESCALATION_MESSAGE = "We recommend contacting a human advisor"
CONFIDENCE_THRESHOLD = 0.9


def should_escalate(result, threshold=CONFIDENCE_THRESHOLD):
    """Return True if the sentiment is negative and confident enough to escalate."""
    if result is None:
        return False
    return result.label == "NEGATIVE" and result.score > threshold


def get_escalation_message(result, threshold=CONFIDENCE_THRESHOLD):
    """Return the escalation message, or None if escalation is not needed."""
    if should_escalate(result, threshold):
        return ESCALATION_MESSAGE
    return None


if __name__ == "__main__":
    cases = [SentimentResult("NEGATIVE", 0.99),
             SentimentResult("NEGATIVE", 0.70),
             SentimentResult("POSITIVE", 0.99),
             SentimentResult("NEUTRAL", 0.98)]
    for case in cases:
        print(case, "-> escalate:", should_escalate(case))
