# Plan v0.2: Evaluation Integrity, Component Hardening & Honest Reporting

## Objective
Execute a rigorous v0.2.0 engineering pass on `voice-phishing-detection` focused on:
1. Eliminating train/test channel mismatch via uniform telephony simulation across all splits.
2. Establishing strictly disjoint splits (split by script and voice variant) with expanded sample sizes (>= 150/class test, >= 60 hard negatives test, >= 100/class val).
3. Introducing held-out test scripts with distinct vocabulary to evaluate generalization beyond author-lexicon overlap.
4. Retraining and calibrating unsupervised acoustic anomaly models (`IsolationForest` vs `OneClassSVM`) strictly on telephony validation audio.
5. Upgrading text cue analysis to intent-aware proximity matching with inquiry dampening to cut hard-negative false alarms.
6. Training a calibrated learned logistic stacker fusion on validation data alongside the hand-tuned baseline.
7. Expanding evaluation in `evaluate.py` with 5 ablation variants, 95% bootstrap confidence intervals, per-voice breakdown, and granular error analysis.
8. Aligning documentation and README claims with empirical evidence, removing hype, adding "What I Learned", and ensuring CI passes green.

---

## Ground Rules
- **No Test Tuning**: All thresholds, hyper-parameters, stacker weights, and lexicon rules are selected strictly on the `val` split, then frozen. The `test` split is evaluated once without retroactive tuning.
- **Honest Reporting**: Whatever the numbers turn out to be, report them transparently with confidence intervals and sample sizes ($N$). If acoustic signal remains weak or hard-negative FPR remains non-trivial, state it directly.
- **Before/After Tracking**: Maintain comparative before (v0.1) vs after (v0.2) metrics in the documentation.

---

## Step-by-Step Execution Plan

### Step 1: Data Pipeline Overhaul (`scripts/generate_synthetic_data.py`)
- **Telephony Simulation Pipeline**:
  - 8 kHz downsampling
  - ITU-T G.711 $\mu$-law compression and companding
  - Bandpass filtering (300 Hz – 3400 Hz)
  - Randomized line noise with SNR 15–30 dB + mains hum
- **Disjoint Train / Val / Test Splits**:
  - Disjoint Voice Variants: Systematically modulate the single Windows SAPI5 engine (Microsoft Zira) across rate, pitch, and formants, assigning non-overlapping variants to train, val, and test.
  - Disjoint Scripts:
    - Train: Benign conversational scripts only (120 clips).
    - Val: Separate benign, hard-negative, and scam scripts (100 benign [40 hard neg] + 100 scam = 200 clips).
    - Test: Completely held-out benign, hard-negative, and scam scripts authored with novel phrasing/vocabulary (150 benign [65 hard neg] + 150 scam = 300 clips).
- **Manifest**: Record `split`, `label`, `is_hard_negative`, `voice_id`, `script_id`, `tts_engine`, and augmentation parameters in `manifest.json`.
- **Commit**: `feat(data): apply telephony simulation to all splits with disjoint train/val/test`

### Step 2: Retrain & Calibrate Acoustic Anomaly Models (`src/vishing_detector/anomaly/detector.py`)
- Retrain `IsolationForest` and `OneClassSVM` strictly on telephony `train` benign features.
- Evaluate both models on `val` telephony audio; compare ROC-AUC and F1.
- Calibrate the alert threshold empirically on `val` to hit target FPR / maximize F1.
- Update `config/default.yaml` and model artifacts with the selected model and calibrated threshold.
- **Commit**: `feat(anomaly): retrain on telephony audio and calibrate threshold on validation`

### Step 3: Intent-Aware Scam Cue Scoring (`src/vishing_detector/nlp/scam_cues.py`, `config/scam_lexicon.yaml`)
- Implement token-window proximity matching (e.g. within 8 tokens) between imperative demand verbs ("read me", "give me", "tell me your", "provide", "disclose", etc.) and credential/payment keywords.
- Add benign context patterns that down-weight or cancel cues on inquiries or reporting ("did you send", "is it normal", "I received a code", "should I share", "the courier is asking", etc.).
- Maintain granular explainability: record matched pattern, span, category, and inquiry dampener status.
- Add unit tests with contrastive pairs (attacker demand vs victim inquiry). Freeze rules before testing.
- **Commit**: `feat(nlp): add intent-aware demand and benign-context cue scoring`

### Step 4: Learned Stacker Fusion (`src/vishing_detector/fusion/`)
- Implement a learned logistic regression stacker trained on `val` features:
  - Max and mean acoustic anomaly score
  - Text cue score
  - Cue category counts
  - Acoustic $\times$ Text interaction
- Apply probability calibration (`CalibratedClassifierCV` / Platt scaling).
- Select threat decision threshold on `val`.
- Configurable EMA smoothing factor $\alpha$ for streaming risk assessment.
- Keep the legacy hand-tuned fusion as a comparative baseline.
- **Commit**: `feat(fusion): add learned logistic stacker with calibrated threshold`

### Step 5: Comprehensive Evaluation Upgrade (`evaluate.py`)
- Evaluate 5 ablation variants on `test` at their frozen val-calibrated thresholds:
  1. Acoustic-only
  2. Text-only (baseline keyword matching)
  3. Text-only (intent-aware)
  4. Fused (legacy hand-tuned)
  5. Fused (learned stacker)
- Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, Hard-Negative FPR ($N=65$), Plain Benign FPR ($N=85$), Latency (mean/median seconds from chunk timestamps).
- 95% Bootstrap Confidence Intervals (1,000 resamples) for F1, ROC-AUC, and Hard-Neg FPR.
- Per-voice-variant breakdown table.
- Error Analysis: 5 worst false positives and 5 worst false negatives with transcripts and cue details.
- Save `results/results_v0.2.json` and regenerate all 6 visualization figures in `docs/figures/`.
- **Commit**: `feat(eval): add ablation variants, bootstrap CIs, per-voice breakdown, error analysis`

### Step 6: Test Suite & Quality Gates (`tests/`)
- Add unit tests covering:
  - Telephony simulation function (sample rate, bandwidth, SNR)
  - Threshold calibration on validation data
  - Intent-aware vs benign inquiry contrastive text pairs
  - Learned stacker fit, predict, and explain
- Verify `pytest` passes with 100% success and `ruff check .` passes cleanly.
- **Commit**: `test: cover telephony simulation, calibration, intent cues, and stacker`

### Step 7: Documentation & Honesty Pass (`README.md`, `notebooks/`, `app/`)
- Rewrite README:
  - Remove all unsupported claims ("vocal tension", "synthesized speech", "superiority", "rigorous", etc.).
  - Cite real corpora (Switchboard via LDC).
  - Clear Before (v0.1) vs After (v0.2) ablation table with sample counts and 95% CIs.
  - Transparently disclose synthetic data limitations, single SAPI5 voice engine, and developer author overlap.
  - Add "What I Learned" section covering real-world lessons.
  - Ensure all figures and Mermaid diagrams render cleanly via relative repository links.
  - Update `notebooks/demo.ipynb` and `streamlit_app.py` for new models and stacker.
- **Commit**: `docs: rewrite README claims, results tables, and figures` and `chore: update notebook and Streamlit app for new model`

### Step 8: Push, CI Verification & Release v0.2.0
- Push to `main`.
- Monitor GitHub Actions workflow until green across all 4 OS/Python matrix jobs.
- Tag and release `v0.2.0` with honest release notes.
- Verify fresh clone in a clean temporary directory.
