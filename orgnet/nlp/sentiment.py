"""Sentiment, emotion, and tone analysis (from Enron project)."""

import re
import numpy as np
from typing import Dict
from dataclasses import dataclass

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""

    sentiment: str  # 'positive', 'negative', 'neutral'
    score: float  # Sentiment score (-1 to 1)
    confidence: float  # Confidence score (0 to 1)


@dataclass
class EmotionResult:
    """Container for emotion detection results."""

    emotions: Dict[str, float]  # emotion -> score (0-1)
    dominant_emotion: str
    emotion_score: float


@dataclass
class ToneResult:
    """Container for tone analysis results."""

    formality: str  # 'formal', 'informal', 'neutral'
    urgency: str  # 'urgent', 'casual', 'neutral'
    politeness: str  # 'polite', 'neutral', 'rude'
    tone_scores: Dict[str, float]


# Sentiment lexicons
POSITIVE_WORDS = {
    "great",
    "excellent",
    "good",
    "perfect",
    "amazing",
    "wonderful",
    "fantastic",
    "thanks",
    "thank",
    "appreciate",
    "pleased",
    "happy",
    "glad",
    "excited",
    "success",
    "successful",
    "achievement",
    "progress",
    "improvement",
    "better",
    "agree",
    "agreement",
    "support",
    "helpful",
    "assist",
    "collaborate",
    "congratulations",
    "congrats",
    "celebrate",
    "achievement",
    "milestone",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "horrible",
    "worst",
    "disappointed",
    "frustrated",
    "angry",
    "upset",
    "concerned",
    "worried",
    "problem",
    "issue",
    "error",
    "fail",
    "failure",
    "failed",
    "mistake",
    "wrong",
    "incorrect",
    "broken",
    "delay",
    "late",
    "missed",
    "overdue",
    "urgent",
    "critical",
    "emergency",
    "disagree",
    "disagreement",
    "conflict",
    "unable",
    "cannot",
    "won't",
    "unfortunately",
    "sorry",
    "apologize",
    "regret",
    "concern",
}

INTENSIFIERS = {
    "very",
    "extremely",
    "really",
    "quite",
    "highly",
    "totally",
    "completely",
    "absolutely",
    "definitely",
    "certainly",
    "surely",
}

NEGATORS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "none",
    "nothing",
    "nobody",
    "nowhere",
    "hardly",
    "scarcely",
    "barely",
    "few",
    "little",
}

# Emotion keyword patterns
EMOTION_PATTERNS = {
    "anger": [
        r"\b(angry|furious|rage|annoyed|irritated|frustrated|mad|upset)\b",
        r"\b(hate|despise|loathe|disgusted)\b",
    ],
    "frustration": [
        r"\b(frustrated|frustrating|frustration|annoyed|annoying)\b",
        r"\b(fed up|sick of|tired of|had enough)\b",
        r"\b(problem|issue|difficulty|struggle)\b",
    ],
    "satisfaction": [
        r"\b(satisfied|pleased|happy|glad|content|delighted)\b",
        r"\b(great|excellent|perfect|wonderful|fantastic)\b",
        r"\b(appreciate|grateful|thankful|thanks)\b",
    ],
    "concern": [
        r"\b(concerned|worried|anxious|nervous|uneasy)\b",
        r"\b(issue|problem|trouble|difficulty)\b",
    ],
    "excitement": [
        r"\b(excited|excitement|thrilled|pumped|eager)\b",
        r"\b(amazing|wow|incredible|unbelievable)\b",
    ],
}

# Tone indicators
FORMAL_INDICATORS = [
    r"\b(regards|sincerely|respectfully|dear|mr\.|mrs\.|ms\.|dr\.)\b",
    r"\b(please|kindly|request|require|necessary|appropriate)\b",
    r"\b(best regards|yours sincerely|cordially)\b",
]

INFORMAL_INDICATORS = [
    r"\b(hey|hi|hey there|what\'s up|yo)\b",
    r"\b(thanks|thx|tks|cheers|later|talk soon)\b",
    r"\b(lol|haha|omg|wtf|btw|fyi)\b",
    r"\b(can\'t|don\'t|won\'t|it\'s|that\'s)\b",  # Contractions
]

URGENT_INDICATORS = [
    r"\b(urgent|asap|immediately|critical|emergency|deadline)\b",
    r"\b(time sensitive|right away|without delay)\b",
    r"\b(need|must|require|essential|crucial)\b",
]

CASUAL_INDICATORS = [
    r"\b(whenever|at your convenience|no rush|when you get a chance)\b",
    r"\b(take your time|no hurry|no pressure)\b",
]

POLITE_INDICATORS = [
    r"\b(please|kindly|thank you|appreciate|grateful)\b",
    r"\b(would you|could you|would it be possible)\b",
    r"\b(sorry|apologize|excuse me|pardon)\b",
]

RUDE_INDICATORS = [
    r"\b(demand|insist|you must|you have to|required immediately)\b",
    r"\b(disappointed|unacceptable|failure|blame)\b",
]


class SentimentAnalyzer:
    """Analyzes sentiment, emotion, and tone in text."""

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Classify sentiment using rule-based approach.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult
        """
        if not text or not isinstance(text, str):
            return SentimentResult(sentiment="neutral", score=0.0, confidence=0.0)

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        if not words:
            return SentimentResult(sentiment="neutral", score=0.0, confidence=0.0)

        # Count positive and negative words
        positive_count = 0
        negative_count = 0

        for i, word in enumerate(words):
            # Check for negators
            is_negated = False
            for j in range(max(0, i - 3), i):
                if words[j] in NEGATORS:
                    is_negated = True
                    break

            # Check for intensifiers
            is_intensified = False
            for j in range(max(0, i - 2), i):
                if words[j] in INTENSIFIERS:
                    is_intensified = True
                    break

            multiplier = 2.0 if is_intensified else 1.0

            if word in POSITIVE_WORDS:
                if is_negated:
                    negative_count += multiplier
                else:
                    positive_count += multiplier
            elif word in NEGATIVE_WORDS:
                if is_negated:
                    positive_count += multiplier
                else:
                    negative_count += multiplier

        total_words = len(words)
        if total_words == 0:
            score = 0.0
        else:
            score = (positive_count - negative_count) / total_words
            score = np.clip(score, -1.0, 1.0)

        if score > 0.1:
            sentiment = "positive"
            confidence = min(abs(score), 1.0)
        elif score < -0.1:
            sentiment = "negative"
            confidence = min(abs(score), 1.0)
        else:
            sentiment = "neutral"
            confidence = 1.0 - abs(score)

        return SentimentResult(sentiment=sentiment, score=score, confidence=confidence)

    def detect_emotions(self, text: str) -> EmotionResult:
        """
        Detect emotions in text.

        Args:
            text: Text to analyze

        Returns:
            EmotionResult with detected emotions
        """
        if not text or not isinstance(text, str):
            return EmotionResult(
                emotions={"neutral": 1.0}, dominant_emotion="neutral", emotion_score=0.0
            )

        text_lower = text.lower()
        emotion_scores = {}

        for emotion, patterns in EMOTION_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches

            # Normalize by text length
            word_count = len(re.findall(r"\b\w+\b", text))
            if word_count > 0:
                normalized_score = min(score / max(word_count / 10, 1), 1.0)
            else:
                normalized_score = 0.0

            emotion_scores[emotion] = normalized_score

        # If no emotions detected, set neutral
        if not emotion_scores or max(emotion_scores.values()) == 0:
            emotion_scores["neutral"] = 1.0
            dominant_emotion = "neutral"
            emotion_score = 0.0
        else:
            # Get dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            emotion_score = emotion_scores[dominant_emotion]

            # Add neutral score (inverse of max emotion)
            emotion_scores["neutral"] = max(0.0, 1.0 - emotion_score)

        return EmotionResult(
            emotions=emotion_scores, dominant_emotion=dominant_emotion, emotion_score=emotion_score
        )

    def analyze_tone(self, text: str) -> ToneResult:
        """
        Analyze tone of text.

        Args:
            text: Text to analyze

        Returns:
            ToneResult with tone analysis
        """
        if not text or not isinstance(text, str):
            return ToneResult(
                formality="neutral", urgency="neutral", politeness="neutral", tone_scores={}
            )

        text_lower = text.lower()

        pattern_counts = {}
        for pattern_list, name in [
            (FORMAL_INDICATORS, "formal"),
            (INFORMAL_INDICATORS, "informal"),
            (URGENT_INDICATORS, "urgent"),
            (CASUAL_INDICATORS, "casual"),
            (POLITE_INDICATORS, "polite"),
            (RUDE_INDICATORS, "rude"),
        ]:
            pattern_counts[name] = sum(
                len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in pattern_list
            )

        formal_count = pattern_counts["formal"]
        informal_count = pattern_counts["informal"]
        urgent_count = pattern_counts["urgent"]
        casual_count = pattern_counts["casual"]
        polite_count = pattern_counts["polite"]
        rude_count = pattern_counts["rude"]

        # Normalize by text length
        word_count = len(re.findall(r"\b\w+\b", text))
        normalization = max(word_count / 20, 1)

        formal_score = min(formal_count / normalization, 1.0)
        informal_score = min(informal_count / normalization, 1.0)
        urgent_score = min(urgent_count / normalization, 1.0)
        casual_score = min(casual_count / normalization, 1.0)
        polite_score = min(polite_count / normalization, 1.0)
        rude_score = min(rude_count / normalization, 1.0)

        # Determine formality
        if formal_score > informal_score and formal_score > 0.1:
            formality = "formal"
        elif informal_score > formal_score and informal_score > 0.1:
            formality = "informal"
        else:
            formality = "neutral"

        # Determine urgency
        if urgent_score > casual_score and urgent_score > 0.1:
            urgency = "urgent"
        elif casual_score > urgent_score and casual_score > 0.1:
            urgency = "casual"
        else:
            urgency = "neutral"

        # Determine politeness
        if polite_score > rude_score and polite_score > 0.1:
            politeness = "polite"
        elif rude_score > polite_score and rude_score > 0.1:
            politeness = "rude"
        else:
            politeness = "neutral"

        tone_scores = {
            "formal": formal_score,
            "informal": informal_score,
            "urgent": urgent_score,
            "casual": casual_score,
            "polite": polite_score,
            "rude": rude_score,
        }

        return ToneResult(
            formality=formality, urgency=urgency, politeness=politeness, tone_scores=tone_scores
        )

    def analyze_all(self, text: str) -> Dict:
        """
        Perform complete sentiment, emotion, and tone analysis.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with all analysis results
        """
        sentiment = self.analyze_sentiment(text)
        emotions = self.detect_emotions(text)
        tone = self.analyze_tone(text)

        return {
            "sentiment": {
                "label": sentiment.sentiment,
                "score": sentiment.score,
                "confidence": sentiment.confidence,
            },
            "emotions": {
                "dominant": emotions.dominant_emotion,
                "score": emotions.emotion_score,
                "all_emotions": emotions.emotions,
            },
            "tone": {
                "formality": tone.formality,
                "urgency": tone.urgency,
                "politeness": tone.politeness,
                "scores": tone.tone_scores,
            },
        }
