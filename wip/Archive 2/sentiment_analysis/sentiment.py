"""
Sentiment classification (positive/negative/neutral).

Uses rule-based and ML approaches for sentiment analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    sentiment: str  # 'positive', 'negative', 'neutral'
    score: float  # Sentiment score (-1 to 1)
    confidence: float  # Confidence score (0 to 1)


@dataclass
class SentimentClassifier:
    """Container for sentiment classifier."""
    method: str = 'rule_based'  # 'rule_based' or 'ml'
    model: Optional[any] = None


# Sentiment lexicons
POSITIVE_WORDS = {
    'great', 'excellent', 'good', 'perfect', 'amazing', 'wonderful', 'fantastic',
    'thanks', 'thank', 'appreciate', 'pleased', 'happy', 'glad', 'excited',
    'success', 'successful', 'achievement', 'progress', 'improvement', 'better',
    'agree', 'agreement', 'support', 'helpful', 'assist', 'collaborate',
    'congratulations', 'congrats', 'celebrate', 'achievement', 'milestone'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointed', 'frustrated',
    'angry', 'upset', 'concerned', 'worried', 'problem', 'issue', 'error',
    'fail', 'failure', 'failed', 'mistake', 'wrong', 'incorrect', 'broken',
    'delay', 'late', 'missed', 'overdue', 'urgent', 'critical', 'emergency',
    'disagree', 'disagreement', 'conflict', 'unable', 'cannot', 'won\'t',
    'unfortunately', 'sorry', 'apologize', 'regret', 'concern'
}

INTENSIFIERS = {
    'very', 'extremely', 'really', 'quite', 'highly', 'totally', 'completely',
    'absolutely', 'definitely', 'certainly', 'surely'
}

NEGATORS = {
    'not', 'no', 'never', 'neither', 'nor', 'none', 'nothing', 'nobody',
    'nowhere', 'hardly', 'scarcely', 'barely', 'few', 'little'
}


def classify_sentiment_rule_based(text: str) -> SentimentResult:
    """
    Classify sentiment using rule-based approach.
    
    Uses sentiment lexicons and pattern matching.
    
    Args:
        text: Text to analyze
        
    Returns:
        SentimentResult
    """
    if not text or not isinstance(text, str):
        return SentimentResult(sentiment='neutral', score=0.0, confidence=0.0)
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    if not words:
        return SentimentResult(sentiment='neutral', score=0.0, confidence=0.0)
    
    # Count positive and negative words
    positive_count = 0
    negative_count = 0
    
    for i, word in enumerate(words):
        # Check for negators
        is_negated = False
        for j in range(max(0, i-3), i):
            if words[j] in NEGATORS:
                is_negated = True
                break
        
        # Check for intensifiers
        is_intensified = False
        for j in range(max(0, i-2), i):
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
    
    # Calculate sentiment score (-1 to 1)
    total_words = len(words)
    if total_words == 0:
        score = 0.0
    else:
        score = (positive_count - negative_count) / max(total_words, 1)
        score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
    
    # Determine sentiment
    if score > 0.1:
        sentiment = 'positive'
        confidence = min(abs(score), 1.0)
    elif score < -0.1:
        sentiment = 'negative'
        confidence = min(abs(score), 1.0)
    else:
        sentiment = 'neutral'
        confidence = 1.0 - abs(score)
    
    return SentimentResult(
        sentiment=sentiment,
        score=score,
        confidence=confidence
    )


def classify_sentiment(
    texts: List[str],
    classifier: Optional[SentimentClassifier] = None
) -> List[SentimentResult]:
    """
    Classify sentiment for multiple texts.
    
    Args:
        texts: List of texts to analyze
        classifier: Optional SentimentClassifier (uses rule-based if None)
        
    Returns:
        List of SentimentResult objects
    """
    if classifier is None or classifier.method == 'rule_based':
        results = [classify_sentiment_rule_based(text) for text in texts]
    elif classifier.method == 'ml' and classifier.model:
        # ML-based classification (would use trained model)
        logger.warning("ML-based sentiment classification not yet implemented. Using rule-based.")
        results = [classify_sentiment_rule_based(text) for text in texts]
    else:
        results = [classify_sentiment_rule_based(text) for text in texts]
    
    return results


def train_sentiment_classifier(
    train_df: pd.DataFrame,
    text_col: str = 'text',
    label_col: str = 'sentiment',
    method: str = 'naive_bayes'
) -> SentimentClassifier:
    """
    Train ML-based sentiment classifier.
    
    Args:
        train_df: Training DataFrame with text and sentiment labels
        text_col: Column name for text
        label_col: Column name for sentiment labels ('positive', 'negative', 'neutral')
        method: Classification method
        
    Returns:
        Trained SentimentClassifier
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        logger.warning("scikit-learn not available. Cannot train ML classifier.")
        return SentimentClassifier(method='rule_based')
    
    logger.info(f"Training {method} sentiment classifier...")
    
    # Prepare data
    X_train = train_df[text_col].fillna('').astype(str).values
    y_train = train_df[label_col].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train model
    if method == 'naive_bayes':
        model = MultinomialNB()
    elif method == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    model.fit(X_train_vec, y_encoded)
    model.label_encoder_ = label_encoder
    model.vectorizer_ = vectorizer
    
    logger.info("Sentiment classifier trained successfully")
    
    return SentimentClassifier(method='ml', model=model)


if __name__ == "__main__":
    # Test sentiment analysis
    test_texts = [
        "Thank you so much for your help! This is great news.",
        "I'm very disappointed with the results. This is a serious problem.",
        "The meeting is scheduled for next week.",
        "I'm extremely frustrated that this wasn't completed on time."
    ]
    
    results = classify_sentiment(test_texts)
    
    print("Sentiment Analysis Results:")
    for text, result in zip(test_texts, results):
        print(f"\nText: {text[:50]}...")
        print(f"  Sentiment: {result.sentiment}")
        print(f"  Score: {result.score:.2f}")
        print(f"  Confidence: {result.confidence:.2f}")

