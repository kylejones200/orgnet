"""Escalation risk prediction (from Enron project)."""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class EscalationPredictor:
    """Container for escalation prediction model."""

    model: Optional[any] = None
    method: str = "heuristic"  # 'heuristic' or 'ml'
    escalation_threshold: float = 0.5


def extract_escalation_features(
    df: pd.DataFrame, text_col: str = "text", subject_col: str = "subject"
) -> pd.DataFrame:
    """
    Extract features for escalation prediction.

    Args:
        df: DataFrame with email data
        text_col: Column name for email text
        subject_col: Column name for subject

    Returns:
        DataFrame with feature columns
    """
    df = df.copy()

    complaint_keywords = [
        "complaint",
        "dissatisfied",
        "unhappy",
        "unacceptable",
        "poor service",
        "bad experience",
        "disappointed",
        "frustrated",
    ]
    text_lower = df[text_col].fillna("").astype(str).str.lower()
    df["complaint_count"] = sum(
        text_lower.str.contains(kw, regex=False) for kw in complaint_keywords
    )

    urgency_keywords = ["urgent", "asap", "immediately", "critical", "emergency"]
    df["urgency_count"] = sum(text_lower.str.contains(kw, regex=False) for kw in urgency_keywords)

    negative_words = [
        "problem",
        "issue",
        "error",
        "wrong",
        "failed",
        "broken",
        "unable",
        "cannot",
        "won't",
        "didn't",
    ]
    df["negative_sentiment_count"] = sum(
        text_lower.str.contains(word, regex=False) for word in negative_words
    )

    # Email length
    df["text_length"] = df[text_col].fillna("").astype(str).str.len()

    # Question count
    df["question_count"] = df[text_col].fillna("").astype(str).str.count(r"\?")

    return df


def predict_escalation_risk_heuristic(df: pd.DataFrame) -> np.ndarray:
    """
    Predict escalation risk using heuristic rules.

    Args:
        df: DataFrame with email data (must have features from extract_escalation_features)

    Returns:
        Array of escalation risk scores (0-1)
    """
    df = df.copy()

    risk = np.full(len(df), 0.3)

    complaint_count = df["complaint_count"].fillna(0)
    risk += np.minimum(complaint_count * 0.2, 0.4)

    negative_count = df["negative_sentiment_count"].fillna(0)
    risk += np.minimum(negative_count * 0.1, 0.3)

    urgency_count = df["urgency_count"].fillna(0)
    risk += np.minimum(urgency_count * 0.15, 0.2)

    text_length = df["text_length"].fillna(0)
    risk = np.where(text_length > 1000, risk + 0.1, risk)

    question_count = df["question_count"].fillna(0)
    risk = np.where(question_count > 3, risk + 0.1, risk)

    return np.clip(risk, 0.0, 1.0)


def predict_escalation_risk(
    df: pd.DataFrame, predictor: Optional[EscalationPredictor] = None, text_col: str = "text"
) -> np.ndarray:
    """
    Predict escalation risk for emails.

    Args:
        df: DataFrame with email data
        predictor: Optional EscalationPredictor
        text_col: Column name for email text

    Returns:
        Array of escalation risk scores (0-1)
    """
    if predictor is None:
        predictor = EscalationPredictor()

    # Extract features
    df_features = extract_escalation_features(df, text_col=text_col)

    if predictor.method == "heuristic":
        risks = predict_escalation_risk_heuristic(df_features)
    elif predictor.method == "ml" and predictor.model:
        # ML-based prediction
        feature_cols = [
            "complaint_count",
            "urgency_count",
            "negative_sentiment_count",
            "text_length",
            "question_count",
        ]
        X = df_features[[c for c in feature_cols if c in df_features.columns]].fillna(0)
        risks = predictor.model.predict_proba(X)[:, 1]  # Probability of escalation
    else:
        risks = predict_escalation_risk_heuristic(df_features)

    return risks


def train_escalation_model(
    df: pd.DataFrame, actual_escalations: Optional[pd.Series] = None, text_col: str = "text"
) -> EscalationPredictor:
    """
    Train ML model for escalation prediction.

    Args:
        df: DataFrame with email data
        actual_escalations: Optional Series with actual escalation labels (1=escalated, 0=not)
        text_col: Column name for email text

    Returns:
        Trained EscalationPredictor
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available. Using heuristic predictor.")
        return EscalationPredictor()

    if actual_escalations is None:
        logger.warning("No actual escalation data provided. Using heuristic predictor.")
        return EscalationPredictor()

    logger.info("Training escalation prediction model...")

    # Extract features
    df_features = extract_escalation_features(df, text_col=text_col)

    # Prepare data
    feature_cols = [
        "complaint_count",
        "urgency_count",
        "negative_sentiment_count",
        "text_length",
        "question_count",
    ]
    X = df_features[[c for c in feature_cols if c in df_features.columns]].fillna(0)
    y = actual_escalations.values

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    logger.info(f"Escalation model accuracy: {score:.3f}")

    return EscalationPredictor(model=model, method="ml")
