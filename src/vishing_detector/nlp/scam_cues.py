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
    inquiry_detected: bool = False

    @property
    def has_critical_indicators(self) -> bool:
        """Indicates if high-risk categories (credentials or financial) were triggered."""
        critical_cats = {"credential_harvesting", "financial_demand", "remote_access"}
        return any(m.category in critical_cats for m in self.cue_matches)


class ScamCueAnalyzer:
    """Rule-based scam cue extractor with configurable YAML lexicon, intent awareness, and ML classifier."""

    DEFAULT_INQUIRY_PATTERNS = [
        r"\b(?:is that normal|is it normal|is this normal)\b",
        r"\b(?:did (?:you|our bank|someone) (?:send|initiate|call|email|text))\b",
        r"\b(?:should i (?:change|reset|pay|give|confirm|worry))\b",
        r"\b(?:can you (?:help|verify|show me|explain|check))\b",
        r"\b(?:could you (?:verify|check|help|tell me))\b",
        r"\b(?:why (?:was|did) (?:my|our))\b",
        r"\b(?:asking (?:why|for|if)|inquire about|wondering if)\b",
        r"\b(?:i (?:just )?(?:received|got|saw) (?:an? )?(?:sms|text|letter|call|voicemail|notification|charge))\b",
        r"\b(?:reporting a suspicious|suspicious voicemail|unfamiliar vendor)\b",
        r"\b(?:claim(?:ing|ed) to be|fraudulent scam)\b",
        r"\b(?:told them no|hung up)\b",
        r"\b(?:dispute the (?:charge|transaction))\b",
        r"\b(?:want to verify|help me verify)\b",
        r"\b(?:calling .* support to ask why)\b",
        r"\b(?:trying to reset my)\b",
    ]

    def __init__(
        self,
        lexicon_path: Optional[Union[str, Path]] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        inquiry_dampener: float = 0.20,
    ) -> None:
        """Initialize the scam cue analyzer.

        Args:
            lexicon_path: Path to YAML lexicon file. If None, uses default configuration.
            custom_weights: Optional dict overriding category weights.
            inquiry_dampener: Multiplier (0.0 - 1.0) applied when text indicates benign inquiry/reporting.
        """
        self.lexicon_path = Path(lexicon_path) if lexicon_path else None
        self.categories: Dict[str, Dict[str, Any]] = {}
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        self._compiled_inquiry_patterns: List[re.Pattern] = []
        self.custom_weights = custom_weights or {}
        self.inquiry_dampener = inquiry_dampener

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
        self._compiled_inquiry_patterns = [
            re.compile(pat, re.IGNORECASE) for pat in self.DEFAULT_INQUIRY_PATTERNS
        ]

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
                    "weight": 1.6,
                    "patterns": [
                        r"(?:immediately|urgent|within (?:the next )?\d+ (?:minutes|hours)|right now|act fast|today)",
                        r"(?:arrest warrant|taken into custody|indictment|criminal investigation unit)",
                        r"(?:account has been compromised|critical warning|final notice|freeze order|liabilities|asset seizure)",
                        r"(?:malicious ransomware|trojan virus|breached|terminated due to fraudulent)",
                    ],
                },
                "credential_harvesting": {
                    "name": "Credential and Authentication Demands",
                    "weight": 2.8,
                    "patterns": [
                        r"(?:\b(?:read\s+(?:back|out|me)?|give\s+(?:me|us)?|tell\s+(?:me|us)?|provide|disclose|share\s+(?:with\s+me)?|send\s+(?:me|us)?|enter)\b(?:\s+\w+){0,8}\s+\b(?:otp|passcode|passcodes?|pin|password|cvv|cvc|ssn|card\s+number|security\s+code)\b)",
                        r"(?:\b(?:what\s+is\s+your|need\s+your|require\s+your|confirm\s+your|verify\s+your)\b(?:\s+\w+){0,4}\s+\b(?:otp|pin|password|passcode|cvv|cvc|ssn)\b)",
                        r"(?:\bread\s+back\s+the\s+code\b)",
                    ],
                },
                "impersonation": {
                    "name": "Entity Impersonation",
                    "weight": 1.6,
                    "patterns": [
                        r"(?:\b(?:internal revenue service|irs|federal trade commission|ftc|social security administration|treasury department|justice|police department|fbi|customs agency|windows defender|apple technical|microsoft (?:support|customer)|chase security|target fraud|wells fargo)\b)",
                        r"(?:\b(?:officer|special agent|investigator|security officer)\b)",
                    ],
                },
                "financial_demand": {
                    "name": "Unusual Payment Demands",
                    "weight": 2.2,
                    "patterns": [
                        r"(?:\b(?:buy|pay|send|purchase|load)\b(?:\s+\w+){0,6}\s+\b(?:gift\s+cards?|apple\s+gift|target\s+gift|google\s+play|steam\s+card)\b)",
                        r"(?:\b(?:transfer|deposit|send|pay|wire)\b(?:\s+\w+){0,6}\s+\b(?:bitcoin|crypto|cryptocurrency|bitcoin\s+atm|western\s+union|wire\s+transfer)\b)",
                        r"(?:\b(?:safe\s+account|custody\s+account|safekeeping\s+account|government\s+safe\s+account)\b)",
                    ],
                },
                "remote_access": {
                    "name": "Remote Computer Access",
                    "weight": 2.5,
                    "patterns": [
                        r"(?:\b(?:download|install|grant|give|allow)\b(?:\s+\w+){0,6}\s+\b(?:anydesk|teamviewer|ultraviewer|remote\s+access|remote\s+control)\b)",
                    ],
                },
                "secrecy_isolation": {
                    "name": "Secrecy and Isolation Tactics",
                    "weight": 1.8,
                    "patterns": [
                        r"(?:don'?t tell (?:anyone|your bank|your family))",
                        r"(?:keep this confidential|stay on the line|do not hang up)",
                        r"(?:do not inform your family or bank teller)",
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

        # Check for benign inquiry or reporting context
        is_inquiry = any(p.search(cleaned_text) for p in self._compiled_inquiry_patterns)

        # Compute raw weighted score with non-linear saturation curve
        total_raw_score = sum(category_scores.values())
        if is_inquiry:
            total_raw_score *= self.inquiry_dampener

        # Saturation formula: score = 1.0 - exp(-raw / 3.0), mapping [0, inf) -> [0.0, 1.0]
        rule_score = float(1.0 - np.exp(-total_raw_score / 3.0)) if total_raw_score > 0 else 0.0

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
            inquiry_detected=is_inquiry,
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
