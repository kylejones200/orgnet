"""Email classification (priority, category, action detection) from Enron project."""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ClassificationResult:
    """Container for email classification results."""

    priority: List[str]  # 'high', 'medium', 'low'
    category: List[str]  # Category labels
    action_required: List[bool]  # Binary
    urgency_score: List[float]  # 0-1
    is_spam: List[bool]  # Binary


# Priority keywords
URGENCY_KEYWORDS = {
    "high": ["urgent", "asap", "immediately", "critical", "emergency", "time sensitive"],
    "medium": ["important", "please", "soon", "deadline", "need", "required"],
    "low": ["whenever", "if possible", "at your convenience"],
}

# Category keyword patterns
CATEGORY_PATTERNS = {
    "sales": [
        r"\b(sale|purchase|buy|order|quote|price|cost|invoice|contract|deal|proposal|client|customer)\b",
        r"\b(revenue|commission|discount|pricing|negotiat)\b",
    ],
    "support": [
        r"\b(support|help|issue|problem|bug|error|fix|resolve|troubleshoot|ticket)\b",
        r"\b(customer service|technical support|assistance|resolve|broken)\b",
    ],
    "hr": [
        r"\b(hire|hiring|job|position|resume|cv|interview|employee|staff|recruit)\b",
        r"\b(vacation|pto|leave|benefit|salary|payroll|performance review)\b",
    ],
    "legal": [
        r"\b(legal|law|attorney|lawyer|contract|agreement|compliance|regulation|litigation)\b",
        r"\b(nda|terms|conditions|liability|disclaimer|copyright)\b",
    ],
    "finance": [
        r"\b(budget|expense|cost|payment|invoice|accounting|financial|revenue|profit)\b",
        r"\b(tax|audit|account|balance|transaction|refund|payment|billing)\b",
    ],
    "operations": [
        r"\b(meeting|schedule|calendar|project|task|deadline|deliverable|status)\b",
        r"\b(process|procedure|workflow|operations|production|supply|logistics)\b",
    ],
    "general": [],  # Default category
}

# Spam keywords
SPAM_KEYWORDS = [
    "free",
    "click here",
    "limited time",
    "act now",
    "winner",
    "prize",
    "guarantee",
    "risk free",
    "opt out",
    "unsubscribe",
    "viagra",
    "casino",
    "lottery",
    "congratulations",
    "urgent action required",
]

# Action required indicators
ACTION_INDICATORS = [
    r"\b(please|kindly|request|need|require|must|should|would you|could you)\b",
    r"\b(action|respond|reply|answer|confirm|approve|review|sign)\b",
    r"\?",  # Question marks
]


class EmailClassifier:
    """Classifies emails by priority, category, and action requirements."""

    def __init__(self):
        """Initialize email classifier."""
        self.priority_model = None
        self.category_model = None
        self.action_model = None

    def classify_priority(self, text: str, subject: Optional[str] = None) -> Tuple[str, float]:
        """
        Classify email priority (high/medium/low).

        Args:
            text: Email body text
            subject: Email subject (optional)

        Returns:
            Tuple of (priority_label, urgency_score)
        """
        if not text:
            return ("low", 0.0)

        full_text = ((subject or "") + " " + text).lower()

        # Calculate urgency score
        urgency_score = 0.0
        max_score = 0.0

        high_count = sum(kw in full_text for kw in URGENCY_KEYWORDS["high"])
        urgency_score += high_count * 0.3
        max_score += 0.3

        medium_count = sum(kw in full_text for kw in URGENCY_KEYWORDS["medium"])
        urgency_score += medium_count * 0.15
        max_score += 0.15

        low_count = sum(kw in full_text for kw in URGENCY_KEYWORDS["low"])
        urgency_score -= low_count * 0.1

        # Question marks (action needed)
        question_count = full_text.count("?")
        urgency_score += min(question_count * 0.05, 0.2)
        max_score += 0.2

        # Exclamation marks (importance)
        exclamation_count = full_text.count("!")
        urgency_score += min(exclamation_count * 0.03, 0.15)
        max_score += 0.15

        # Normalize to 0-1
        if max_score > 0:
            normalized_score = min(urgency_score / max_score, 1.0)
        else:
            normalized_score = 0.0

        normalized_score = max(0.0, normalized_score)

        # Determine priority
        if normalized_score > 0.6:
            priority = "high"
        elif normalized_score > 0.3:
            priority = "medium"
        else:
            priority = "low"

        return (priority, normalized_score)

    def classify_category(self, text: str, subject: Optional[str] = None) -> str:
        """
        Classify email category.

        Args:
            text: Email body text
            subject: Email subject (optional)

        Returns:
            Category label
        """
        if not text:
            return "general"

        full_text = ((subject or "") + " " + text).lower()

        # Calculate scores for each category
        scores = {}
        for category, patterns in CATEGORY_PATTERNS.items():
            if category == "general":
                continue
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, full_text, re.IGNORECASE))
                score += matches
            scores[category] = score

        # Assign category with highest score
        if scores and max(scores.values()) > 0:
            category = max(scores.items(), key=lambda x: x[1])[0]
        else:
            category = "general"

        return category

    def detect_action_required(
        self, text: str, subject: Optional[str] = None
    ) -> Tuple[bool, float]:
        """
        Detect if email requires action.

        Args:
            text: Email body text
            subject: Email subject (optional)

        Returns:
            Tuple of (action_required, confidence)
        """
        if not text:
            return (False, 0.0)

        full_text = ((subject or "") + " " + text).lower()

        action_score = 0.0

        # Check for action indicators
        for pattern in ACTION_INDICATORS:
            matches = len(re.findall(pattern, full_text, re.IGNORECASE))
            action_score += matches * 0.1

        # Question marks strongly indicate action needed
        question_count = full_text.count("?")
        action_score += question_count * 0.2

        # Normalize to 0-1
        action_score = min(action_score, 1.0)

        # Threshold for action required
        action_required = action_score > 0.3
        confidence = min(action_score, 1.0)

        return (action_required, confidence)

    def detect_spam(
        self, text: str, subject: Optional[str] = None, sender: Optional[str] = None
    ) -> Tuple[bool, float]:
        """
        Detect if email is spam.

        Args:
            text: Email body text
            subject: Email subject (optional)
            sender: Email sender (optional)

        Returns:
            Tuple of (is_spam, probability)
        """
        if not text:
            return (False, 0.0)

        full_text = ((subject or "") + " " + text).lower()

        spam_score = 0.0

        keyword_count = sum(kw in full_text for kw in SPAM_KEYWORDS)
        spam_score += keyword_count * 0.15

        # All caps in subject
        if subject:
            caps_ratio = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
            if caps_ratio > 0.5:
                spam_score += 0.2

        # Excessive exclamation/question marks
        exclamation_count = full_text.count("!")
        question_count = full_text.count("?")
        if exclamation_count + question_count > 5:
            spam_score += 0.2

        # Suspicious links
        if "http://" in full_text or "https://" in full_text:
            link_count = full_text.count("http")
            if link_count > 3:
                spam_score += 0.15

        if sender:
            sender_lower = sender.lower()
            suspicious_patterns = {"noreply", "no-reply", "mailer", "notification"}
            if any(pattern in sender_lower for pattern in suspicious_patterns):
                spam_score += 0.1

        # Normalize to probability
        spam_probability = min(spam_score, 1.0)
        is_spam = spam_probability > 0.5

        return (is_spam, spam_probability)

    def classify(
        self,
        texts: List[str],
        subjects: Optional[List[str]] = None,
        senders: Optional[List[str]] = None,
    ) -> ClassificationResult:
        """
        Classify multiple emails.

        Args:
            texts: List of email body texts
            subjects: Optional list of email subjects
            senders: Optional list of email senders

        Returns:
            ClassificationResult with all classifications
        """
        if subjects is None:
            subjects = [None] * len(texts)
        if senders is None:
            senders = [None] * len(texts)

        priorities = []
        urgency_scores = []
        categories = []
        action_required = []
        action_confidences = []
        is_spam = []
        spam_probabilities = []

        for text, subject, sender in zip(texts, subjects, senders):
            # Priority
            priority, urgency = self.classify_priority(text, subject)
            priorities.append(priority)
            urgency_scores.append(urgency)

            # Category
            category = self.classify_category(text, subject)
            categories.append(category)

            # Action required
            action, confidence = self.detect_action_required(text, subject)
            action_required.append(action)
            action_confidences.append(confidence)

            # Spam
            spam, spam_prob = self.detect_spam(text, subject, sender)
            is_spam.append(spam)
            spam_probabilities.append(spam_prob)

        return ClassificationResult(
            priority=priorities,
            category=categories,
            action_required=action_required,
            urgency_score=urgency_scores,
            is_spam=is_spam,
        )
