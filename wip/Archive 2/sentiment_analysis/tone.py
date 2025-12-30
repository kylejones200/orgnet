"""
Tone analysis (formal/informal, urgent/casual, etc.).

Analyzes the tone and style of email communication.
"""

import re
from typing import Dict, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToneResult:
    """Container for tone analysis results."""
    formality: str  # 'formal', 'informal', 'neutral'
    urgency: str  # 'urgent', 'casual', 'neutral'
    politeness: str  # 'polite', 'neutral', 'rude'
    tone_scores: Dict[str, float]


# Tone indicators
FORMAL_INDICATORS = [
    r'\b(regards|sincerely|respectfully|dear|mr\.|mrs\.|ms\.|dr\.)\b',
    r'\b(please|kindly|request|require|necessary|appropriate)\b',
    r'\b(best regards|yours sincerely|cordially)\b',
]

INFORMAL_INDICATORS = [
    r'\b(hey|hi|hey there|what\'s up|yo)\b',
    r'\b(thanks|thx|tks|cheers|later|talk soon)\b',
    r'\b(lol|haha|omg|wtf|btw|fyi)\b',
    r'\b(can\'t|don\'t|won\'t|it\'s|that\'s)\b',  # Contractions
]

URGENT_INDICATORS = [
    r'\b(urgent|asap|immediately|critical|emergency|deadline)\b',
    r'\b(time sensitive|right away|without delay)\b',
    r'\b(need|must|require|essential|crucial)\b',
]

CASUAL_INDICATORS = [
    r'\b(whenever|at your convenience|no rush|when you get a chance)\b',
    r'\b(take your time|no hurry|no pressure)\b',
]

POLITE_INDICATORS = [
    r'\b(please|kindly|thank you|appreciate|grateful)\b',
    r'\b(would you|could you|would it be possible)\b',
    r'\b(sorry|apologize|excuse me|pardon)\b',
]

RUDE_INDICATORS = [
    r'\b(demand|insist|you must|you have to|required immediately)\b',
    r'\b(disappointed|unacceptable|failure|blame)\b',
]


def analyze_tone(text: str) -> ToneResult:
    """
    Analyze tone of text.
    
    Args:
        text: Text to analyze
        
    Returns:
        ToneResult with tone analysis
    """
    if not text or not isinstance(text, str):
        return ToneResult(
            formality='neutral',
            urgency='neutral',
            politeness='neutral',
            tone_scores={}
        )
    
    text_lower = text.lower()
    
    # Count indicators
    formal_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in FORMAL_INDICATORS)
    informal_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in INFORMAL_INDICATORS)
    urgent_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in URGENT_INDICATORS)
    casual_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in CASUAL_INDICATORS)
    polite_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in POLITE_INDICATORS)
    rude_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in RUDE_INDICATORS)
    
    # Normalize by text length
    word_count = len(re.findall(r'\b\w+\b', text))
    normalization = max(word_count / 20, 1)
    
    formal_score = min(formal_count / normalization, 1.0)
    informal_score = min(informal_count / normalization, 1.0)
    urgent_score = min(urgent_count / normalization, 1.0)
    casual_score = min(casual_count / normalization, 1.0)
    polite_score = min(polite_count / normalization, 1.0)
    rude_score = min(rude_count / normalization, 1.0)
    
    # Determine formality
    if formal_score > informal_score and formal_score > 0.1:
        formality = 'formal'
    elif informal_score > formal_score and informal_score > 0.1:
        formality = 'informal'
    else:
        formality = 'neutral'
    
    # Determine urgency
    if urgent_score > casual_score and urgent_score > 0.1:
        urgency = 'urgent'
    elif casual_score > urgent_score and casual_score > 0.1:
        urgency = 'casual'
    else:
        urgency = 'neutral'
    
    # Determine politeness
    if polite_score > rude_score and polite_score > 0.1:
        politeness = 'polite'
    elif rude_score > polite_score and rude_score > 0.1:
        politeness = 'rude'
    else:
        politeness = 'neutral'
    
    tone_scores = {
        'formal': formal_score,
        'informal': informal_score,
        'urgent': urgent_score,
        'casual': casual_score,
        'polite': polite_score,
        'rude': rude_score
    }
    
    return ToneResult(
        formality=formality,
        urgency=urgency,
        politeness=politeness,
        tone_scores=tone_scores
    )


@dataclass
class ToneClassifier:
    """Container for tone classifier (for future ML implementation)."""
    method: str = 'rule_based'


if __name__ == "__main__":
    # Test tone analysis
    test_texts = [
        "Dear Mr. Smith, I would be grateful if you could please review the attached document at your earliest convenience. Best regards, John",
        "Hey! Can you get this done ASAP? It's urgent!",
        "Hi there, whenever you get a chance, could you take a look? Thanks!",
        "You must complete this immediately. This is unacceptable."
    ]
    
    for text in test_texts:
        result = analyze_tone(text)
        print(f"\nText: {text[:60]}...")
        print(f"  Formality: {result.formality}")
        print(f"  Urgency: {result.urgency}")
        print(f"  Politeness: {result.politeness}")

