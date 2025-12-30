"""
Sentiment analysis and emotional intelligence module.

Includes sentiment classification, emotion detection, tone analysis,
and sentiment trend tracking.
"""

from .sentiment import (
    classify_sentiment,
    train_sentiment_classifier,
    SentimentClassifier,
    SentimentResult
)
from .emotions import (
    detect_emotions,
    EmotionResult,
    EmotionClassifier
)
from .tone import (
    analyze_tone,
    ToneResult,
    ToneClassifier
)
from .trends import (
    analyze_sentiment_trends,
    SentimentTrendAnalysis
)

__all__ = [
    'classify_sentiment',
    'train_sentiment_classifier',
    'SentimentClassifier',
    'SentimentResult',
    'detect_emotions',
    'EmotionResult',
    'EmotionClassifier',
    'analyze_tone',
    'ToneResult',
    'ToneClassifier',
    'analyze_sentiment_trends',
    'SentimentTrendAnalysis',
]

