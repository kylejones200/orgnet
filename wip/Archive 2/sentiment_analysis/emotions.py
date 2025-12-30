"""
Emotion detection (anger, frustration, satisfaction, etc.).

Detects specific emotions in email text.
"""

import re
from typing import List, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """Container for emotion detection results."""
    emotions: Dict[str, float]  # emotion -> score (0-1)
    dominant_emotion: str
    emotion_score: float


# Emotion keyword patterns
EMOTION_PATTERNS = {
    'anger': [
        r'\b(angry|furious|rage|annoyed|irritated|frustrated|mad|upset)\b',
        r'\b(hate|despise|loathe|disgusted)\b',
    ],
    'frustration': [
        r'\b(frustrated|frustrating|frustration|annoyed|annoying)\b',
        r'\b(fed up|sick of|tired of|had enough)\b',
        r'\b(problem|issue|difficulty|struggle)\b',
    ],
    'satisfaction': [
        r'\b(satisfied|pleased|happy|glad|content|delighted)\b',
        r'\b(great|excellent|perfect|wonderful|fantastic)\b',
        r'\b(appreciate|grateful|thankful|thanks)\b',
    ],
    'concern': [
        r'\b(concerned|worried|anxious|nervous|uneasy)\b',
        r'\b(issue|problem|trouble|difficulty)\b',
    ],
    'excitement': [
        r'\b(excited|excitement|thrilled|pumped|eager)\b',
        r'\b(amazing|wow|incredible|unbelievable)\b',
    ],
    'neutral': []  # Default
}


def detect_emotions(text: str) -> EmotionResult:
    """
    Detect emotions in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        EmotionResult with detected emotions
    """
    if not text or not isinstance(text, str):
        return EmotionResult(
            emotions={'neutral': 1.0},
            dominant_emotion='neutral',
            emotion_score=0.0
        )
    
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, patterns in EMOTION_PATTERNS.items():
        if emotion == 'neutral':
            continue
        
        score = 0.0
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            score += matches
        
        # Normalize by text length
        word_count = len(re.findall(r'\b\w+\b', text))
        if word_count > 0:
            normalized_score = min(score / max(word_count / 10, 1), 1.0)
        else:
            normalized_score = 0.0
        
        emotion_scores[emotion] = normalized_score
    
    # If no emotions detected, set neutral
    if not emotion_scores or max(emotion_scores.values()) == 0:
        emotion_scores['neutral'] = 1.0
        dominant_emotion = 'neutral'
        emotion_score = 0.0
    else:
        # Get dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        emotion_score = emotion_scores[dominant_emotion]
        
        # Add neutral score (inverse of max emotion)
        emotion_scores['neutral'] = 1.0 - emotion_score
    
    return EmotionResult(
        emotions=emotion_scores,
        dominant_emotion=dominant_emotion,
        emotion_score=emotion_score
    )


@dataclass
class EmotionClassifier:
    """Container for emotion classifier (for future ML implementation)."""
    method: str = 'rule_based'


if __name__ == "__main__":
    # Test emotion detection
    test_texts = [
        "I'm extremely frustrated with this situation. This is very annoying.",
        "Thank you so much! I'm really satisfied with the results.",
        "I'm concerned about the deadline. This could be a problem.",
        "I'm so excited about the new project! This is amazing!"
    ]
    
    for text in test_texts:
        result = detect_emotions(text)
        print(f"\nText: {text[:50]}...")
        print(f"  Dominant emotion: {result.dominant_emotion}")
        print(f"  Emotion score: {result.emotion_score:.2f}")
        print(f"  Top emotions: {sorted(result.emotions.items(), key=lambda x: x[1], reverse=True)[:3]}")

