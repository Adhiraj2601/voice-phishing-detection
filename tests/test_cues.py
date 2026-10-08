"""Unit tests for NLP scam cue extraction, intent-awareness, and contrastive demand vs inquiry pairs."""

from vishing_detector.nlp.scam_cues import ScamCueAnalyzer


def test_benign_text_scoring():
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")
    benign_text = "Hello David, let us review the meeting agenda for tomorrow afternoon."
    res = analyzer.analyze_text(benign_text)

    assert res.combined_text_score == 0.0
    assert len(res.cue_matches) == 0
    assert not res.has_critical_indicators
    assert not res.inquiry_detected


def test_scam_credential_cues():
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")
    scam_text = "Please read back the verification OTP code and your bank PIN immediately."
    res = analyzer.analyze_text(scam_text)

    assert res.combined_text_score > 0.5
    assert res.has_critical_indicators
    matched_cats = [m.category for m in res.cue_matches]
    assert "credential_harvesting" in matched_cats
    assert "urgency_threat" in matched_cats


def test_contrastive_credential_demand_vs_victim_inquiry():
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")

    # Attacker demand
    attacker_text = "Please read me the 6 digit OTP passcode sent to your phone right now."
    res_attacker = analyzer.analyze_text(attacker_text)
    assert res_attacker.combined_text_score > 0.5
    assert res_attacker.has_critical_indicators
    assert not res_attacker.inquiry_detected

    # Victim inquiry / question about the same terms
    victim_text = "Hi, I just received an SMS with an OTP code from my bank. Is that normal or should I change my password?"
    res_victim = analyzer.analyze_text(victim_text)
    assert res_victim.combined_text_score < 0.25
    assert res_victim.inquiry_detected
    assert not res_victim.has_critical_indicators
    assert res_attacker.combined_text_score > res_victim.combined_text_score * 2.0


def test_contrastive_delivery_courier_reporting():
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")

    # Attacker directive
    demand_text = "Disclose your four digit verification PIN right away to prevent account termination."
    res_demand = analyzer.analyze_text(demand_text)
    assert res_demand.combined_text_score > 0.5
    assert res_demand.has_critical_indicators

    # Third-party benign courier mention
    mention_text = "Hey honey, the grocery delivery courier is outside asking for the four digit verification PIN on the app."
    res_mention = analyzer.analyze_text(mention_text)
    assert res_mention.combined_text_score < 0.25
    assert res_mention.inquiry_detected


def test_contrastive_scam_reporting_and_refusal():
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")

    scam_report_text = "Mom, someone called claiming to be from the bank asking for my card numbers, but I told them no and hung up."
    res = analyzer.analyze_text(scam_report_text)
    assert res.combined_text_score < 0.25
    assert res.inquiry_detected
    assert not res.has_critical_indicators


def test_intent_aware_vs_legacy_mode_ablation():
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")
    hard_neg_text = "Hi, I just received an SMS with an OTP code from my bank. Is that normal?"

    # Intent-aware mode suppresses false positive
    res_intent = analyzer.analyze_text(hard_neg_text, intent_aware=True)
    assert res_intent.combined_text_score < 0.25

    # Legacy unconstrained mode triggers false positive on bare keywords
    res_legacy = analyzer.analyze_text(hard_neg_text, intent_aware=False)
    assert res_legacy.combined_text_score > 0.40
    assert res_legacy.combined_text_score > res_intent.combined_text_score


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
