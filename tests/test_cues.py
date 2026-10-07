"""Unit tests for NLP scam cue extraction and rule scoring."""

from vishing_detector.nlp.scam_cues import ScamCueAnalyzer


def test_benign_text_scoring():
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")
    benign_text = "Hello David, let us review the meeting agenda for tomorrow afternoon."
    res = analyzer.analyze_text(benign_text)

    assert res.combined_text_score == 0.0
    assert len(res.cue_matches) == 0
    assert not res.has_critical_indicators


def test_scam_credential_cues():
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")
    scam_text = "Please read back the verification OTP code and your bank PIN immediately."
    res = analyzer.analyze_text(scam_text)

    assert res.combined_text_score > 0.5
    assert res.has_critical_indicators
    matched_cats = [m.category for m in res.cue_matches]
    assert "credential_harvesting" in matched_cats
    assert "urgency_threat" in matched_cats


def test_scam_financial_cues():
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")
    scam_text = "Buy an Apple gift card or transfer via Bitcoin ATM to our safe account."
    res = analyzer.analyze_text(scam_text)

    assert res.combined_text_score > 0.6
    assert res.has_critical_indicators
    matched_cats = [m.category for m in res.cue_matches]
    assert "financial_demand" in matched_cats


def test_scam_remote_access():
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")
    scam_text = "This is Microsoft support. Install AnyDesk and grant remote access to your computer."
    res = analyzer.analyze_text(scam_text)

    assert res.combined_text_score > 0.6
    matched_cats = [m.category for m in res.cue_matches]
    assert "remote_access" in matched_cats
    assert "impersonation" in matched_cats


def test_ml_classifier_training():
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")
    train_texts = [
        "hi how are you today",
        "let's schedule lunch tomorrow",
        "give me your credit card and otp code",
        "pay fine with gift cards immediately",
    ]
    train_labels = [0, 0, 1, 1]

    analyzer.train_ml_classifier(train_texts, train_labels)
    res = analyzer.analyze_text("please read the otp code")

    assert res.ml_score is not None
    assert 0.0 <= res.ml_score <= 1.0
