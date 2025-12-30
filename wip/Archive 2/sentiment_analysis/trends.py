"""
Sentiment trends over time.

Analyzes how sentiment, emotions, and tone change over time.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from .sentiment import classify_sentiment, SentimentResult
from .emotions import detect_emotions, EmotionResult
from .tone import analyze_tone, ToneResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentTrendAnalysis:
    """Container for sentiment trend analysis results."""
    time_periods: List[str]
    sentiment_distribution: Dict[str, List[float]]  # sentiment -> [counts over time]
    emotion_distribution: Dict[str, List[float]]  # emotion -> [counts over time]
    average_sentiment_score: List[float]  # Average sentiment score over time
    tone_distribution: Dict[str, List[float]]  # tone -> [counts over time]


def analyze_sentiment_trends(
    df: pd.DataFrame,
    text_col: str = 'text',
    date_col: str = 'date_parsed',
    time_period: str = 'month'
) -> SentimentTrendAnalysis:
    """
    Analyze sentiment, emotion, and tone trends over time.
    
    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        date_col: Column name for dates
        time_period: Time period ('day', 'week', 'month', 'quarter', 'year')
        
    Returns:
        SentimentTrendAnalysis with trend data
    """
    logger.info(f"Analyzing sentiment trends (period: {time_period})")
    
    # Prepare data
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col, text_col])
    
    if len(df_copy) == 0:
        logger.warning("No valid data for sentiment trend analysis")
        return SentimentTrendAnalysis(
            time_periods=[],
            sentiment_distribution={},
            emotion_distribution={},
            average_sentiment_score=[],
            tone_distribution={}
        )
    
    # Group by time period
    if time_period == 'day':
        df_copy['period'] = df_copy[date_col].dt.date.astype(str)
    elif time_period == 'week':
        df_copy['period'] = df_copy[date_col].dt.to_period('W').astype(str)
    elif time_period == 'month':
        df_copy['period'] = df_copy[date_col].dt.to_period('M').astype(str)
    elif time_period == 'quarter':
        df_copy['period'] = df_copy[date_col].dt.to_period('Q').astype(str)
    elif time_period == 'year':
        df_copy['period'] = df_copy[date_col].dt.to_period('Y').astype(str)
    else:
        raise ValueError(f"Unknown time period: {time_period}")
    
    time_periods = sorted(df_copy['period'].unique())
    
    # Initialize distributions
    sentiment_dist = {'positive': [], 'negative': [], 'neutral': []}
    emotion_dist = {}
    avg_sentiment_scores = []
    tone_formality_dist = {'formal': [], 'informal': [], 'neutral': []}
    tone_urgency_dist = {'urgent': [], 'casual': [], 'neutral': []}
    
    # Analyze each period
    for period in time_periods:
        period_df = df_copy[df_copy['period'] == period]
        texts = period_df[text_col].fillna('').astype(str).tolist()
        
        # Sentiment analysis
        sentiment_results = classify_sentiment(texts)
        positive_count = sum(1 for r in sentiment_results if r.sentiment == 'positive')
        negative_count = sum(1 for r in sentiment_results if r.sentiment == 'negative')
        neutral_count = sum(1 for r in sentiment_results if r.sentiment == 'neutral')
        total = len(sentiment_results)
        
        if total > 0:
            sentiment_dist['positive'].append(positive_count / total)
            sentiment_dist['negative'].append(negative_count / total)
            sentiment_dist['neutral'].append(neutral_count / total)
            
            avg_score = np.mean([r.score for r in sentiment_results])
            avg_sentiment_scores.append(avg_score)
        else:
            sentiment_dist['positive'].append(0.0)
            sentiment_dist['negative'].append(0.0)
            sentiment_dist['neutral'].append(0.0)
            avg_sentiment_scores.append(0.0)
        
        # Emotion analysis
        emotion_results = [detect_emotions(text) for text in texts]
        all_emotions = set()
        for r in emotion_results:
            all_emotions.update(r.emotions.keys())
        
        for emotion in all_emotions:
            if emotion not in emotion_dist:
                emotion_dist[emotion] = []
            
            count = sum(1 for r in emotion_results if r.dominant_emotion == emotion)
            emotion_dist[emotion].append(count / len(emotion_results) if emotion_results else 0.0)
        
        # Tone analysis
        tone_results = [analyze_tone(text) for text in texts]
        formality_counts = {'formal': 0, 'informal': 0, 'neutral': 0}
        urgency_counts = {'urgent': 0, 'casual': 0, 'neutral': 0}
        
        for r in tone_results:
            formality_counts[r.formality] += 1
            urgency_counts[r.urgency] += 1
        
        total_tones = len(tone_results)
        if total_tones > 0:
            for key in formality_counts:
                tone_formality_dist[key].append(formality_counts[key] / total_tones)
            for key in urgency_counts:
                tone_urgency_dist[key].append(urgency_counts[key] / total_tones)
        else:
            for key in formality_counts:
                tone_formality_dist[key].append(0.0)
            for key in urgency_counts:
                tone_urgency_dist[key].append(0.0)
    
    # Combine tone distributions
    tone_distribution = {**tone_formality_dist, **tone_urgency_dist}
    
    logger.info(f"Analyzed sentiment trends across {len(time_periods)} periods")
    
    return SentimentTrendAnalysis(
        time_periods=time_periods,
        sentiment_distribution=sentiment_dist,
        emotion_distribution=emotion_dist,
        average_sentiment_score=avg_sentiment_scores,
        tone_distribution=tone_distribution
    )


def detect_sentiment_shifts(
    trend_analysis: SentimentTrendAnalysis,
    threshold: float = 0.2
) -> List[Dict]:
    """
    Detect significant shifts in sentiment over time.
    
    Args:
        trend_analysis: SentimentTrendAnalysis object
        threshold: Threshold for significant change
        
    Returns:
        List of shift events
    """
    shifts = []
    
    if len(trend_analysis.average_sentiment_score) < 2:
        return shifts
    
    for i in range(1, len(trend_analysis.average_sentiment_score)):
        prev_score = trend_analysis.average_sentiment_score[i-1]
        curr_score = trend_analysis.average_sentiment_score[i]
        
        change = abs(curr_score - prev_score)
        if change > threshold:
            direction = 'positive' if curr_score > prev_score else 'negative'
            shifts.append({
                'period': trend_analysis.time_periods[i],
                'type': 'sentiment_shift',
                'direction': direction,
                'change': change,
                'prev_score': prev_score,
                'curr_score': curr_score
            })
    
    return shifts


if __name__ == "__main__":
    # Test sentiment trends
    from ingest.email_parser import load_emails
    from temporal_features.extractors import extract_temporal_features
    
    print("Testing sentiment trend analysis...")
    df = load_emails('maildir', data_format='maildir', max_rows=500)
    df = extract_temporal_features(df)
    
    trends = analyze_sentiment_trends(df, time_period='month')
    
    print(f"\nSentiment trends across {len(trends.time_periods)} periods")
    print(f"Average sentiment scores: {trends.average_sentiment_score[:5]}...")
    
    shifts = detect_sentiment_shifts(trends)
    print(f"Significant sentiment shifts: {len(shifts)}")

