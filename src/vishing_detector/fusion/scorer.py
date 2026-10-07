"""Multi-modal threat score fusion and explainability engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vishing_detector.anomaly.detector import AnomalyInferenceResult
from vishing_detector.nlp.scam_cues import CueMatch, ScamAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ChunkRiskAssessment:
    """Risk evaluation for an individual audio chunk within the stream.

    Attributes:
        chunk_index: Index of the evaluated chunk.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        acoustic_score: Normalized acoustic anomaly score in [0.0, 1.0].
        text_score: Normalized textual scam cue score in [0.0, 1.0].
        instantaneous_risk: Combined risk score in [0.0, 100.0].
        smoothed_risk: Exponentially smoothed running risk score in [0.0, 100.0].
        risk_level: Human-readable threat tier ('SAFE', 'LOW', 'ELEVATED', 'HIGH', 'CRITICAL').
        triggered_cues: Cues detected in this chunk.
        acoustic_deviations: Notable deviating acoustic features in this chunk.
    """

    chunk_index: int
    start_time: float
    end_time: float
    acoustic_score: float
    text_score: float
    instantaneous_risk: float
    smoothed_risk: float
    risk_level: str
    triggered_cues: List[CueMatch] = field(default_factory=list)
    acoustic_deviations: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class CallExplanation:
    """Comprehensive, human-readable forensic explanation of call assessment.

    Attributes:
        overall_risk_score: Final aggregate risk score in [0.0, 100.0].
        peak_risk_score: Peak risk reached during the call.
        risk_level: Final threat tier ('SAFE', 'LOW', 'ELEVATED', 'HIGH', 'CRITICAL').
        is_threat: Whether the call breached the threat alert threshold.
        first_alert_time: Timestamp in seconds of first threat detection (detection latency).
        primary_reasons: Key human-readable bullet points explaining the decision.
        top_triggered_cues: Top matched scam cues throughout the call.
        anomalous_moments: Highlighted timestamps where vocal acoustics were anomalous.
        recommended_action: Guidance for the recipient or security operations analyst.
    """

    overall_risk_score: float
    peak_risk_score: float
    risk_level: str
    is_threat: bool
    first_alert_time: Optional[float]
    primary_reasons: List[str]
    top_triggered_cues: List[Dict[str, Any]]
    anomalous_moments: List[Dict[str, Any]]
    recommended_action: str


class RiskScorer:
    """Fuses acoustic anomalies and textual scam indicators into a calibrated 0-100 risk score."""

    def __init__(
        self,
        weight_acoustic: float = 0.35,
        weight_text: float = 0.65,
        smoothing_alpha: float = 0.4,
        threshold_low: float = 25.0,
        threshold_elevated: float = 50.0,
        threshold_high: float = 70.0,
        threshold_critical: float = 85.0,
    ) -> None:
        """Initialize the risk scorer.

        Args:
            weight_acoustic: Relative weight given to acoustic feature anomalies (0.0 - 1.0).
            weight_text: Relative weight given to NLP scam cue matches (0.0 - 1.0).
            smoothing_alpha: EMA decay factor (0.0 = total lag, 1.0 = instantaneous).
            threshold_low: Lower bound for 'LOW' risk tier.
            threshold_elevated: Threshold for 'ELEVATED' risk.
            threshold_high: Threshold for 'HIGH' risk (threat alert).
            threshold_critical: Threshold for 'CRITICAL' risk.
        """
        # Normalize weights so they sum to 1.0
        total_w = weight_acoustic + weight_text
        self.w_acoustic = weight_acoustic / total_w
        self.w_text = weight_text / total_w

        self.smoothing_alpha = smoothing_alpha
        self.th_low = threshold_low
        self.th_elevated = threshold_elevated
        self.th_high = threshold_high
        self.th_critical = threshold_critical

        self.running_smoothed_score: float = 0.0

    def reset(self) -> None:
        """Reset running exponential moving average for a new call."""
        self.running_smoothed_score = 0.0

    def compute_chunk_risk(
        self,
        chunk_index: int,
        start_time: float,
        end_time: float,
        anomaly_result: AnomalyInferenceResult,
        nlp_result: ScamAnalysisResult,
    ) -> ChunkRiskAssessment:
        """Compute instantaneous and smoothed risk for a single streaming window.

        Args:
            chunk_index: Index of current chunk.
            start_time: Chunk start time in seconds.
            end_time: Chunk end time in seconds.
            anomaly_result: Outcome from AcousticAnomalyDetector.
            nlp_result: Outcome from ScamCueAnalyzer.

        Returns:
            ChunkRiskAssessment with calculated metrics.
        """
        ac_score = float(anomaly_result.anomaly_score)
        tx_score = float(nlp_result.combined_text_score)

        # Base weighted fusion [0.0, 100.0]
        base_risk = 100.0 * (self.w_acoustic * ac_score + self.w_text * tx_score)

        # Synergy boost: when strong scam language is present alongside high acoustic deviation
        if nlp_result.has_critical_indicators and ac_score > 0.5:
            synergy_boost = 15.0 * ac_score
            base_risk = min(100.0, base_risk + synergy_boost)

        instantaneous = float(np.clip(base_risk, 0.0, 100.0))

        # Exponential moving average update
        if chunk_index == 0 and self.running_smoothed_score == 0.0:
            self.running_smoothed_score = instantaneous
        else:
            self.running_smoothed_score = (
                self.smoothing_alpha * instantaneous
                + (1.0 - self.smoothing_alpha) * self.running_smoothed_score
            )

        smoothed = float(np.clip(self.running_smoothed_score, 0.0, 100.0))
        tier = self._classify_tier(smoothed)

        return ChunkRiskAssessment(
            chunk_index=chunk_index,
            start_time=start_time,
            end_time=end_time,
            acoustic_score=ac_score,
            text_score=tx_score,
            instantaneous_risk=instantaneous,
            smoothed_risk=smoothed,
            risk_level=tier,
            triggered_cues=nlp_result.cue_matches,
            acoustic_deviations=anomaly_result.top_deviating_features,
        )

    def _classify_tier(self, score: float) -> str:
        """Map numerical score to semantic threat tier."""
        if score >= self.th_critical:
            return "CRITICAL"
        elif score >= self.th_high:
            return "HIGH"
        elif score >= self.th_elevated:
            return "ELEVATED"
        elif score >= self.th_low:
            return "LOW"
        return "SAFE"

    def generate_explanation(
        self,
        assessments: List[ChunkRiskAssessment],
        full_transcript: str = "",
    ) -> CallExplanation:
        """Synthesize timeline assessments into an actionable human-readable forensic report.

        Args:
            assessments: Chronological list of ChunkRiskAssessment for the call.
            full_transcript: Optional complete call transcript string.

        Returns:
            CallExplanation dataclass with full diagnostic reasoning.
        """
        if not assessments:
            return CallExplanation(
                overall_risk_score=0.0,
                peak_risk_score=0.0,
                risk_level="SAFE",
                is_threat=False,
                first_alert_time=None,
                primary_reasons=["No audio stream data provided."],
                top_triggered_cues=[],
                anomalous_moments=[],
                recommended_action="Normal call - no intervention required.",
            )

        peak_score = max(a.smoothed_risk for a in assessments)
        final_score = assessments[-1].smoothed_risk
        # Weighted overall summary favoring peak threat occurrences
        overall_score = float(np.clip(0.4 * final_score + 0.6 * peak_score, 0.0, 100.0))
        risk_level = self._classify_tier(overall_score)
        is_threat = overall_score >= self.th_elevated

        # Determine time-to-first-alert (latency)
        first_alert_time = None
        for a in assessments:
            if a.smoothed_risk >= self.th_elevated:
                first_alert_time = a.start_time
                break

        # Aggregate detected cues
        all_cues: List[Dict[str, Any]] = []
        seen_patterns = set()
        for a in assessments:
            for c in a.triggered_cues:
                key = (c.category, c.matched_text.lower())
                if key not in seen_patterns:
                    seen_patterns.add(key)
                    all_cues.append(
                        {
                            "category": c.category_name,
                            "matched_text": c.matched_text,
                            "weight": c.weight,
                            "timestamp": f"{a.start_time:.1f}s - {a.end_time:.1f}s",
                        }
                    )

        # Aggregate peak acoustic deviations
        anomalous_moments: List[Dict[str, Any]] = []
        for a in assessments:
            if a.acoustic_score > 0.55:
                anomalous_moments.append(
                    {
                        "start_time": a.start_time,
                        "end_time": a.end_time,
                        "acoustic_score": a.acoustic_score,
                        "deviations": [f"{name} (z={z:.1f})" for name, z in a.acoustic_deviations[:2]],
                    }
                )

        # Primary bullet explanations
        reasons: List[str] = []
        if any(c["category"] == "Credential and Authentication Requests" for c in all_cues):
            reasons.append("[CREDENTIALS] Detected explicit requests for sensitive credentials (OTP / Passwords / PINs).")
        if any(c["category"] == "Unusual Payment and Fund Transfer" for c in all_cues):
            reasons.append("[PAYMENT] Identified coercive demands for unconventional payment methods (Gift Cards / Crypto / Wire).")
        if any(c["category"] == "Remote Computer Access" for c in all_cues):
            reasons.append("[REMOTE ACCESS] Detected requests to install remote access control software (AnyDesk / TeamViewer).")
        if any(c["category"] == "Urgency and Coercion" for c in all_cues):
            reasons.append("[URGENCY] Identified high-pressure coercive tactics and artificial deadlines.")
        if any(c["category"] == "Entity Impersonation" for c in all_cues):
            reasons.append("[IMPERSONATION] Detected impersonation of government agencies, banks, or tech support departments.")
        if any(c["category"] == "Secrecy and Isolation Tactics" for c in all_cues):
            reasons.append("[SECRECY] Identified instructions demanding secrecy and discouraging third-party consultation.")
        if anomalous_moments:
            reasons.append(
                f"[ACOUSTICS] Vocal acoustic patterns deviated significantly from benign baseline across {len(anomalous_moments)} segments (atypical vocal tension, speech tempo, or spectral distribution)."
            )

        if not reasons:
            reasons.append("No suspicious conversational or acoustic anomalies detected. Speech conforms to normal benign patterns.")

        # Actionable recommendations
        if risk_level in {"CRITICAL", "HIGH"}:
            action = (
                "TERMINATE CALL IMMEDIATELY. Do not provide any verification codes, passwords, or payment. "
                "Contact the institution directly using verified official contact channels."
            )
        elif risk_level == "ELEVATED":
            action = (
                "EXERCISE CAUTION. Caller has exhibited manipulative patterns or atypical acoustics. "
                "Refuse any requests to install software or disclose banking details."
            )
        else:
            action = "Normal conversation. Continue standard vigilance against unsolicited callers."

        return CallExplanation(
            overall_risk_score=round(overall_score, 1),
            peak_risk_score=round(peak_score, 1),
            risk_level=risk_level,
            is_threat=is_threat,
            first_alert_time=first_alert_time,
            primary_reasons=reasons,
            top_triggered_cues=all_cues,
            anomalous_moments=anomalous_moments,
            recommended_action=action,
        )
