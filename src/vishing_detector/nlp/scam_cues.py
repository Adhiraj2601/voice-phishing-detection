"""Rule-based and machine-learning scam cue text analysis module."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class CueMatch:
    """Individual matched scam cue within analyzed text.

    Attributes:
        category: Lexicon category identifier (e.g., 'credential_harvesting').
        category_name: Human-readable category label.
        matched_text: Exact string snippet matched from the input.
        pattern: Regex pattern that produced the match.
        weight: Assigned category severity weight.
        start_char: Character start position in analyzed text.
        end_char: Character end position in analyzed text.
    """

    category: str
    category_name: str
    matched_text: str
    pattern: str
    weight: float
    start_char: int
    end_char: int


@dataclass
class ScamAnalysisResult:
    """Consolidated results of scam cue analysis on a transcript.

    Attributes:
        text: Input transcript text analyzed.
        cue_matches: List of detected CueMatch objects.
        category_scores: Dict mapping category to raw triggered score.
        rule_score: Normalized rule-based threat score in [0.0, 1.0].
        ml_score: Probability score from secondary TF-IDF classifier (if available).
        combined_text_score: Final fused textual threat score in [0.0, 1.0].
        summary_tags: Top alert labels for quick dashboard or CLI inspection.
    """

    text: str
    cue_matches: List[CueMatch] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    rule_score: float = 0.0
    ml_score: Optional[float] = None
    combined_text_score: float = 0.0
    summary_tags: List[str] = field(default_factory=list)

    @property
    def has_critical_indicators(self) -> bool:
        """Indicates if high-risk categories (credentials or financial) were triggered."""
        critical_cats = {"credential_harvesting", "financial_demand", "remote_access"}
        return any(m.category in critical_cats for m in self.cue_matches)


class ScamCueAnalyzer:
    """Rule-based scam cue extractor with configurable YAML lexicon and optional ML classifier."""

    def __init__(
        self,
        lexicon_path: Optional[Union[str, Path]] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the scam cue analyzer.

        Args:
            lexicon_path: Path to YAML lexicon file. If None, uses default configuration.
            custom_weights: Optional dict overriding category weights.
        """
        self.lexicon_path = Path(lexicon_path) if lexicon_path else None
        self.categories: Dict[str, Dict[str, Any]] = {}
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        self.custom_weights = custom_weights or {}

        # Optional ML model components
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None

        self._load_lexicon()

    def _load_lexicon(self) -> None:
        """Load and compile regex patterns from YAML file or built-in defaults."""
        data = None
        if self.lexicon_path and self.lexicon_path.exists():
            try:
                with open(self.lexicon_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                logger.info("Loaded scam lexicon from: %s", self.lexicon_path)
            except Exception as err:
                logger.warning("Failed to load lexicon YAML (%s); using defaults", err)

        if not data or "categories" not in data:
            data = self._get_fallback_lexicon()

        self.categories = data["categories"]
        self._compiled_patterns.clear()

        for cat_id, cat_info in self.categories.items():
            # Apply weight override if provided
            if cat_id in self.custom_weights:
                cat_info["weight"] = self.custom_weights[cat_id]

            patterns = cat_info.get("patterns", [])
            compiled_list = []
            for pat_str in patterns:
                try:
                    compiled = re.compile(pat_str, re.IGNORECASE)
                    compiled_list.append((compiled, pat_str))
                except re.error as err:
                    logger.error("Invalid regex in lexicon [%s]: %s (%s)", cat_id, pat_str, err)

            self._compiled_patterns[cat_id] = compiled_list

    @staticmethod
    def _get_fallback_lexicon() -> Dict[str, Any]:
        """Provides internal fallback lexicon if YAML file is unavailable."""
        return {
            "categories": {
                "urgency_threat": {
                    "name": "Urgency and Coercion",
                    "weight": 1.5,
                    "patterns": [
                        r"(?:immediately|urgent|within (?:the next )?\d+ (?:minutes|hours)|right now)",
                        r"(?:warrant|arrest|law enforcement|fbi|irs|police|custody|legal action)",
                        r"(?:suspend(?:ed)?|freeze|frozen|terminate|cancel(?:led)?|penalty|fine)",
                    ],
                },
                "credential_harvesting": {
                    "name": "Credential and Authentication Requests",
                    "weight": 2.2,
                    "patterns": [
                        r"(?:one-time password|otp|verification code|security code)",
                        r"(?:cvv|cvc|pin number|pin|password|passcode)",
                        r"(?:social security|ssn|aadhaar|id number)",
                        r"(?:read back the code|share the code|tell me the numbers?)",
                    ],
                },
                "impersonation": {
                    "name": "Entity Impersonation",
                    "weight": 1.6,
                    "patterns": [
                        r"(?:internal revenue service|irs|federal trade commission|ftc)",
                        r"(?:microsoft support|apple support|amazon fraud department)",
                        r"(?:bank of america|wells fargo|chase bank|fraud prevention)",
                    ],
                },
                "financial_demand": {
                    "name": "Unusual Payment and Fund Transfer",
                    "weight": 2.0,
                    "patterns": [
                        r"(?:gift card|apple gift card|target gift card|steam card)",
                        r"(?:bitcoin|crypto|cryptocurrency|bitcoin atm|coinbase)",
                        r"(?:wire transfer|western union|zelle|cash app)",
                        r"(?:safe account|safekeeping account|government safe account)",
                    ],
                },
                "remote_access": {
                    "name": "Remote Computer Access",
                    "weight": 2.0,
                    "patterns": [
                        r"(?:anydesk|teamviewer|ultraviewer|quick assist|screen share)",
                        r"(?:download software|install the application|allow remote access)",
                    ],
                },
                "secrecy_isolation": {
                    "name": "Secrecy and Isolation Tactics",
                    "weight": 1.8,
                    "patterns": [
                        r"(?:don'?t tell (?:anyone|your bank|your family))",
                        r"(?:keep this confidential|stay on the line|do not hang up)",
                    ],
                },
            }
        }

    def analyze_text(self, text: str) -> ScamAnalysisResult:
        """Analyze a transcript text block for scam cues and calculate threat scores.

        Args:
            text: Input transcript string.

        Returns:
            ScamAnalysisResult containing detected matches and normalized scores.
        """
        if not text or not text.strip():
            return ScamAnalysisResult(text=text or "")

        cleaned_text = text.strip()
        matches: List[CueMatch] = []
        category_scores: Dict[str, float] = {cat: 0.0 for cat in self.categories}

        for cat_id, cat_info in self.categories.items():
            cat_name = cat_info.get("name", cat_id)
            weight = float(cat_info.get("weight", 1.0))
            compiled_list = self._compiled_patterns.get(cat_id, [])

            for pattern_obj, raw_pattern in compiled_list:
                for match in pattern_obj.finditer(cleaned_text):
                    start, end = match.span()
                    matched_substr = match.group(0)

                    cue_match = CueMatch(
                        category=cat_id,
                        category_name=cat_name,
                        matched_text=matched_substr,
                        pattern=raw_pattern,
                        weight=weight,
                        start_char=start,
                        end_char=end,
                    )
                    matches.append(cue_match)
                    category_scores[cat_id] += weight

        # Compute raw weighted score with non-linear saturation curve
        total_raw_score = sum(category_scores.values())
        # Saturation formula: score = 1.0 - exp(-raw / 3.5), mapping [0, inf) -> [0.0, 1.0]
        rule_score = float(1.0 - np.exp(-total_raw_score / 3.5)) if total_raw_score > 0 else 0.0

        # Optional ML classifier evaluation
        ml_score = None
        if self.classifier is not None and self.tfidf_vectorizer is not None:
            try:
                feats = self.tfidf_vectorizer.transform([cleaned_text])
                probs = self.classifier.predict_proba(feats)[0]
                ml_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
            except Exception as err:
                logger.warning("ML text scoring error: %s", err)

        # Fused textual score: combine rule and ML (favoring rules for transparent recall)
        if ml_score is not None:
            combined_score = 0.65 * rule_score + 0.35 * ml_score
        else:
            combined_score = rule_score

        # Identify top alert tags
        tags: List[str] = []
        for cat_id, cat_score in category_scores.items():
            if cat_score > 0:
                tags.append(self.categories[cat_id].get("name", cat_id))

        return ScamAnalysisResult(
            text=cleaned_text,
            cue_matches=matches,
            category_scores=category_scores,
            rule_score=rule_score,
            ml_score=ml_score,
            combined_text_score=float(np.clip(combined_score, 0.0, 1.0)),
            summary_tags=tags,
        )

    def train_ml_classifier(
        self,
        texts: List[str],
        labels: List[int],
    ) -> None:
        """Train optional TF-IDF + Logistic Regression secondary text classifier.

        Args:
            texts: List of training transcript texts.
            labels: List of binary labels (0 for benign, 1 for vishing).
        """
        logger.info("Training secondary TF-IDF text classifier on %d samples", len(texts))
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            stop_words="english",
        )
        x_tfidf = self.tfidf_vectorizer.fit_transform(texts)

        self.classifier = LogisticRegression(class_weight="balanced", random_state=42)
        self.classifier.fit(x_tfidf, labels)
        logger.info("TF-IDF text classifier training completed.")
