"""
Sentiment analysis example.

Demonstrates sentiment classification, emotion detection, and tone analysis.
"""

from pipeline import build_knowledge_model
from sentiment_analysis.sentiment import classify_sentiment
from sentiment_analysis.emotions import detect_emotions
from sentiment_analysis.tone import analyze_tone
from sentiment_analysis.trends import analyze_sentiment_trends

def main():
    """Example of sentiment analysis capabilities."""
    
    print("Building model for sentiment analysis...")
    model = build_knowledge_model(
        data_path='maildir',
        data_format='maildir',
        sample_size=2000
    )
    
    # Analyze sentiment for sample emails
    print("\nAnalyzing sentiment...")
    sample_texts = model.df['text'].head(10).tolist()
    sentiment_results = classify_sentiment(sample_texts)
    
    print("\nSample Sentiment Analysis:")
    for text, result in zip(sample_texts[:5], sentiment_results[:5]):
        print(f"\nText: {text[:60]}...")
        print(f"  Sentiment: {result.sentiment} (score: {result.score:.2f})")
    
    # Emotion detection
    print("\n\nEmotion Detection Examples:")
    for text in sample_texts[:3]:
        emotions = detect_emotions(text)
        print(f"\nText: {text[:50]}...")
        print(f"  Dominant emotion: {emotions.dominant_emotion}")
        print(f"  Top emotions: {sorted(emotions.emotions.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Tone analysis
    print("\n\nTone Analysis Examples:")
    for text in sample_texts[:3]:
        tone = analyze_tone(text)
        print(f"\nText: {text[:50]}...")
        print(f"  Formality: {tone.formality}")
        print(f"  Urgency: {tone.urgency}")
        print(f"  Politeness: {tone.politeness}")
    
    # Sentiment trends
    print("\n\nAnalyzing sentiment trends...")
    trends = analyze_sentiment_trends(model.df, time_period='month')
    print(f"Analyzed {len(trends.time_periods)} time periods")
    if trends.average_sentiment_score:
        print(f"Average sentiment score: {sum(trends.average_sentiment_score) / len(trends.average_sentiment_score):.2f}")

if __name__ == "__main__":
    main()

