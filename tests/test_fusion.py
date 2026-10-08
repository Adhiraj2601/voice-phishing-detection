"""Unit tests for multi-modal risk score fusion and forensic explainability."""

from vishing_detector.anomaly.detector import AnomalyInferenceResult
from vishing_detector.fusion.scorer import RiskScorer
from vishing_detector.nlp.scam_cues import CueMatch, ScamAnalysisResult


def test_scorer_benign_assessment():
    scorer = RiskScorer()
    anom = AnomalyInferenceResult(
        is_anomaly=False,
        anomaly_score=0.1,
        raw_score=0.2,
        top_deviating_features=[],
    )
    nlp = ScamAnalysisResult(text="Just calling to see how you are doing", combined_text_score=0.0)

    assess = scorer.compute_chunk_risk(0, 0.0, 3.0, anom, nlp)
    assert assess.risk_level in ["SAFE", "LOW"]
    assert assess.smoothed_risk < 25.0


def test_scorer_scam_assessment_with_synergy():
    scorer = RiskScorer()
    anom = AnomalyInferenceResult(
        is_anomaly=True,
        anomaly_score=0.85,
        raw_score=-0.4,
        top_deviating_features=[("pitch_f0_std", 3.2)],
    )
    cue = CueMatch(
        category="credential_harvesting",
        category_name="Credential and Authentication Requests",
        matched_text="OTP code",
        pattern="otp",
        weight=2.2,
        start_char=0,
        end_char=8,
    )
    nlp = ScamAnalysisResult(
        text="Give me your OTP code right now",
        cue_matches=[cue],
        combined_text_score=0.9,
    )

    assess = scorer.compute_chunk_risk(0, 0.0, 3.0, anom, nlp)
    assert assess.risk_level in ["HIGH", "CRITICAL"]
    assert assess.smoothed_risk >= 70.0


def test_scorer_explanation_generation():
    scorer = RiskScorer()
    anom = AnomalyInferenceResult(
        is_anomaly=True,
        anomaly_score=0.8,
        raw_score=-0.3,
        top_deviating_features=[("spectral_centroid_mean", 2.5)],
    )
    cue = CueMatch(
        category="financial_demand",
        category_name="Unusual Payment and Fund Transfer",
        matched_text="gift card",
        pattern="gift card",
        weight=2.0,
        start_char=0,
        end_char=9,
    )
    nlp = ScamAnalysisResult(text="pay with gift card", cue_matches=[cue], combined_text_score=0.85)

    assess = scorer.compute_chunk_risk(0, 0.0, 3.0, anom, nlp)
    expl = scorer.generate_explanation([assess])

    assert expl.is_threat
    assert expl.overall_risk_score > 50.0
    assert any("[PAYMENT]" in r for r in expl.primary_reasons)
    assert "TERMINATE CALL" in expl.recommended_action or "CAUTION" in expl.recommended_action


def test_stacker_fit_predict_and_persistence(tmp_path):
    import numpy as np

    from vishing_detector.fusion.stacker import LogisticRiskStacker

    stacker = LogisticRiskStacker(random_state=42, cv_folds=2)

    # Synthetic training data: 20 samples, 10 features
    X = np.random.RandomState(42).randn(20, 10).astype(np.float32)
    # Give scam samples higher feature values
    X[10:] += 2.0
    y = np.array([0] * 10 + [1] * 10, dtype=int)

    stacker.fit(X, y)
    assert stacker.is_fitted
    assert 0.0 <= stacker.calibrated_threshold <= 1.0

    prob = stacker.predict_proba(X[0])
    assert 0.0 <= prob <= 1.0

    risk = stacker.predict_risk_score(X[0])
    assert 0.0 <= risk <= 100.0

    # Persistence
    save_file = tmp_path / "test_stacker.joblib"
    stacker.save(save_file)
    assert save_file.exists()

    loaded = LogisticRiskStacker.load(save_file)
    assert loaded.is_fitted
    assert abs(loaded.predict_proba(X[0]) - prob) < 1e-4


def test_scorer_with_stacker_mode():
    import numpy as np

    from vishing_detector.fusion.stacker import LogisticRiskStacker

    stacker = LogisticRiskStacker(random_state=42, cv_folds=2)
    X = np.random.RandomState(42).randn(20, 10).astype(np.float32)
    X[10:] += 2.0
    y = np.array([0] * 10 + [1] * 10, dtype=int)
    stacker.fit(X, y)

    scorer = RiskScorer(fusion_mode="stacker", stacker=stacker)
    anom = AnomalyInferenceResult(is_anomaly=False, anomaly_score=0.1, raw_score=0.2, top_deviating_features=[])
    nlp = ScamAnalysisResult(text="Hello friend", combined_text_score=0.0)

    assess = scorer.compute_chunk_risk(0, 0.0, 3.0, anom, nlp)
    assert 0.0 <= assess.smoothed_risk <= 100.0
    assert assess.risk_level in ["SAFE", "LOW", "ELEVATED", "HIGH", "CRITICAL"]
