"""Entity extraction with NER and pattern matching (from Enron project)."""

import re
from typing import List, Dict
from dataclasses import dataclass, field

from orgnet.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class EntityResult:
    """Container for extracted entities."""

    persons: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    email_addresses: List[str] = field(default_factory=list)
    financial_amounts: List[Dict] = field(default_factory=list)
    dates_times: List[str] = field(default_factory=list)
    other: List[Dict] = field(default_factory=list)  # Other entity types with labels


# Email address pattern
EMAIL_PATTERN = r"[\w\.-]+@[\w\.-]+\.\w+"

# Financial amount patterns
MONEY_PATTERNS = [
    r"\$[\d,]+\.?\d*",  # $1,000 or $100.50
    r"[\d,]+\.?\d*\s*dollars?",  # 1,000 dollars
    r"[\d,]+\.?\d*\s*USD",  # 1,000 USD
    r"[\d,]+\.?\d*\s*million",  # 5 million
    r"[\d,]+\.?\d*\s*billion",  # 2 billion
]

# Date/time patterns
DATE_PATTERNS = [
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",  # MM/DD/YYYY
    r"\w+\s+\d{1,2},?\s+\d{4}",  # January 1, 2024
    r"\d{1,2}\s+\w+\s+\d{4}",  # 1 January 2024
    r"\d{4}-\d{2}-\d{2}",  # 2024-01-15
]

TIME_PATTERNS = [
    r"\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)",  # 3:30 PM
    r"\d{1,2}:\d{2}",  # 15:30
]

# Global model cache
_spacy_model = None
_transformers_ner = None


class EntityExtractor:
    """Extracts entities from text using NER and pattern matching."""

    def __init__(self, use_spacy: bool = True, use_transformers: bool = False):
        """
        Initialize entity extractor.

        Args:
            use_spacy: Use spaCy for NER if available
            use_transformers: Use transformers for NER if available
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self._spacy_model = None
        self._transformers_ner = None

    def _load_spacy_model(self, model_name: str = "en_core_web_sm"):
        """Load spaCy NER model."""
        global _spacy_model

        if not SPACY_AVAILABLE:
            return None

        if _spacy_model is None:
            try:
                logger.info(f"Loading spaCy model: {model_name}")
                _spacy_model = spacy.load(model_name)
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning(
                    f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}"
                )
                return None

        return _spacy_model

    def _load_transformers_ner(self, model_name: str = "dslim/bert-base-NER"):
        """Load transformers NER pipeline."""
        global _transformers_ner

        if not TRANSFORMERS_AVAILABLE:
            return None

        if _transformers_ner is None:
            try:
                logger.info(f"Loading transformers NER model: {model_name}")
                _transformers_ner = pipeline("ner", model=model_name, aggregation_strategy="simple")
                logger.info("Transformers NER model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load transformers NER: {e}")
                return None

        return _transformers_ner

    def extract_entities(self, text: str) -> EntityResult:
        """
        Extract all entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            EntityResult with extracted entities
        """
        if not text or not isinstance(text, str):
            return EntityResult()

        result = EntityResult()

        # Extract using pattern matching (always available)
        result.email_addresses = self._extract_emails(text)
        result.financial_amounts = self._extract_financial_amounts(text)
        result.dates_times = self._extract_dates_times(text)

        # Extract using NER if available
        if self.use_spacy:
            spacy_result = self._extract_with_spacy(text)
            result.persons.extend(spacy_result.persons)
            result.organizations.extend(spacy_result.organizations)
            result.locations.extend(spacy_result.locations)
            result.other.extend(spacy_result.other)
        elif self.use_transformers:
            transformers_result = self._extract_with_transformers(text)
            result.persons.extend(transformers_result.persons)
            result.organizations.extend(transformers_result.organizations)
            result.locations.extend(transformers_result.locations)
            result.other.extend(transformers_result.other)
        else:
            # Fallback to regex patterns
            regex_result = self._extract_with_regex(text)
            result.persons.extend(regex_result.persons)
            result.organizations.extend(regex_result.organizations)
            result.locations.extend(regex_result.locations)

        # Remove duplicates
        result.persons = list(set(result.persons))
        result.organizations = list(set(result.organizations))
        result.locations = list(set(result.locations))
        result.email_addresses = list(set(result.email_addresses))

        return result

    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses."""
        emails = re.findall(EMAIL_PATTERN, text, re.IGNORECASE)
        return list({e.lower().strip() for e in emails})

    def _extract_financial_amounts(self, text: str) -> List[Dict]:
        """Extract financial amounts."""
        amounts = []

        for pattern in MONEY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_text = match.group(0)
                start, end = match.span()

                # Extract context
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)
                context = text[context_start:context_end].strip()

                # Extract numeric value
                numeric_value = None
                numeric_match = re.search(r"[\d,]+\.?\d*", amount_text)
                if numeric_match:
                    numeric_str = numeric_match.group(0).replace(",", "")
                    try:
                        numeric_value = float(numeric_str)
                        if "million" in amount_text.lower():
                            numeric_value *= 1_000_000
                        elif "billion" in amount_text.lower():
                            numeric_value *= 1_000_000_000
                    except ValueError:
                        pass

                amounts.append(
                    {
                        "amount_text": amount_text,
                        "numeric_value": numeric_value,
                        "context": context,
                        "position": (start, end),
                    }
                )

        seen = set()
        unique_amounts = []
        for amount in amounts:
            key = amount["amount_text"]
            if key not in seen:
                seen.add(key)
                unique_amounts.append(amount)

        return unique_amounts

    def _extract_dates_times(self, text: str) -> List[str]:
        """Extract dates and times."""
        all_patterns = DATE_PATTERNS + TIME_PATTERNS
        dates_times = [
            match for pattern in all_patterns for match in re.findall(pattern, text, re.IGNORECASE)
        ]
        return list(set(dates_times))

    def _extract_with_spacy(self, text: str) -> EntityResult:
        """Extract entities using spaCy."""
        result = EntityResult()

        nlp = self._load_spacy_model()
        if nlp is None:
            return result

        doc = nlp(text)

        for ent in doc.ents:
            entity_text = ent.text.strip()

            if ent.label_ in ["PERSON"]:
                result.persons.append(entity_text)
            elif ent.label_ in ["ORG", "ORGANIZATION"]:
                result.organizations.append(entity_text)
            elif ent.label_ in ["GPE", "LOC", "LOCATION"]:
                result.locations.append(entity_text)
            else:
                result.other.append(
                    {
                        "text": entity_text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )

        return result

    def _extract_with_transformers(self, text: str) -> EntityResult:
        """Extract entities using transformers."""
        result = EntityResult()

        ner_pipeline = self._load_transformers_ner()
        if ner_pipeline is None:
            return result

        try:
            entities = ner_pipeline(text)

            for entity in entities:
                entity_text = entity["word"].strip()
                label = entity["entity_group"]

                if label in ["PER", "PERSON"]:
                    result.persons.append(entity_text)
                elif label in ["ORG", "ORGANIZATION"]:
                    result.organizations.append(entity_text)
                elif label in ["LOC", "LOCATION"]:
                    result.locations.append(entity_text)
                else:
                    result.other.append(
                        {"text": entity_text, "label": label, "score": entity.get("score", 0.0)}
                    )
        except Exception as e:
            logger.warning(f"Transformers NER failed: {e}")

        return result

    def _extract_with_regex(self, text: str) -> EntityResult:
        """Extract entities using regex patterns (fallback)."""
        result = EntityResult()

        # Simple patterns for person names (capitalized words, 2+ words)
        # This is a basic fallback - NER is much better
        person_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        persons = re.findall(person_pattern, text)
        result.persons = [p.strip() for p in persons if len(p.split()) >= 2]

        # Organization patterns
        org_pattern = (
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc\.?|Corp\.?|LLC|Ltd\.?|Company|Co\.?))\b"
        )
        orgs = re.findall(org_pattern, text)
        result.organizations = [o.strip() for o in orgs]

        return result
