"""Response time prediction (from Enron project)."""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ResponseTimePredictor:
    """Container for response time prediction model."""

    model: Optional[any] = None
    method: str = "heuristic"  # 'heuristic' or 'ml'
    avg_response_time: float = 24.0  # Average response time in hours


def extract_response_time_features(
    df: pd.DataFrame,
    text_col: str = "text",
    sender_col: str = "sender",
    date_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Extract features for response time prediction.

    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        sender_col: Column name for sender
        date_col: Column name for dates

    Returns:
        DataFrame with feature columns
    """
    df = df.copy()

    urgency_keywords = ["urgent", "asap", "immediately", "critical", "deadline"]
    text_lower = df[text_col].fillna("").astype(str).str.lower()
    df["urgency_count"] = sum(text_lower.str.contains(kw, regex=False) for kw in urgency_keywords)

    # Question marks
    df["question_count"] = df[text_col].fillna("").astype(str).str.count(r"\?")

    # Email length
    df["text_length"] = df[text_col].fillna("").astype(str).str.len()

    # Temporal features
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["hour"] = df[date_col].dt.hour
        df["day_of_week"] = df[date_col].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_business_hours"] = (
            (df["hour"] >= 9) & (df["hour"] < 17) & (df["is_weekend"] == 0)
        ).astype(int)
    else:
        df["hour"] = 12
        df["day_of_week"] = 2
        df["is_weekend"] = 0
        df["is_business_hours"] = 1

    # Sender importance (if available)
    if "sender_degree" in df.columns:
        df["sender_importance"] = df["sender_degree"].fillna(0)
    else:
        df["sender_importance"] = 0

    return df


def predict_response_time_heuristic(df: pd.DataFrame) -> np.ndarray:
    """
    Predict response time using heuristic rules.

    Args:
        df: DataFrame with email data (must have features from extract_response_time_features)

    Returns:
        Array of predicted response times (hours)
    """
    df = df.copy()

    base_time = np.where(df["urgency_count"].fillna(0) > 0, 2.0, 12.0)

    is_weekend = df["is_weekend"].fillna(0).astype(bool)
    is_business_hours = df["is_business_hours"].fillna(1).astype(bool)

    base_time = np.where(is_weekend, base_time * 2.0, base_time)
    base_time = np.where(~is_weekend & ~is_business_hours, base_time * 1.5, base_time)

    sender_importance = df["sender_importance"].fillna(0)
    base_time = np.where(sender_importance > 100, base_time * 0.7, base_time)
    base_time = np.where(
        (sender_importance > 50) & (sender_importance <= 100), base_time * 0.85, base_time
    )

    question_count = df["question_count"].fillna(0)
    question_multiplier = np.where(
        question_count > 0, np.maximum(0.8, 1.0 - question_count * 0.1), 1.0
    )
    base_time = base_time * question_multiplier

    return np.maximum(0.5, base_time)


def predict_response_time(
    df: pd.DataFrame, predictor: Optional[ResponseTimePredictor] = None, text_col: str = "text"
) -> np.ndarray:
    """
    Predict response times for emails.

    Args:
        df: DataFrame with email data
        predictor: Optional ResponseTimePredictor
        text_col: Column name for email text

    Returns:
        Array of predicted response times (hours)
    """
    if predictor is None:
        predictor = ResponseTimePredictor()

    # Extract features
    df_features = extract_response_time_features(df, text_col=text_col)

    if predictor.method == "heuristic":
        predictions = predict_response_time_heuristic(df_features)
    elif predictor.method == "ml" and predictor.model:
        # ML-based prediction
        feature_cols = [
            "urgency_count",
            "question_count",
            "text_length",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_business_hours",
            "sender_importance",
        ]
        X = df_features[[c for c in feature_cols if c in df_features.columns]].fillna(0)
        predictions = predictor.model.predict(X)
    else:
        predictions = predict_response_time_heuristic(df_features)

    return predictions


def train_response_time_model(
    df: pd.DataFrame, actual_response_times: Optional[pd.Series] = None, text_col: str = "text"
) -> ResponseTimePredictor:
    """
    Train ML model for response time prediction.

    Args:
        df: DataFrame with email data
        actual_response_times: Optional Series with actual response times
        text_col: Column name for email text

    Returns:
        Trained ResponseTimePredictor
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available. Using heuristic predictor.")
        return ResponseTimePredictor()

    if actual_response_times is None:
        logger.warning("No actual response times provided. Using heuristic predictor.")
        return ResponseTimePredictor()

    logger.info("Training response time prediction model...")

    # Extract features
    df_features = extract_response_time_features(df, text_col=text_col)

    # Prepare data
    feature_cols = [
        "urgency_count",
        "question_count",
        "text_length",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_business_hours",
        "sender_importance",
    ]
    X = df_features[[c for c in feature_cols if c in df_features.columns]].fillna(0)
    y = actual_response_times.values

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    logger.info(f"Response time model R² score: {score:.3f}")

    return ResponseTimePredictor(model=model, method="ml", avg_response_time=np.mean(y))
