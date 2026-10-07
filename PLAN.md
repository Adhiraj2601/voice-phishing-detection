# Implementation Plan: AI-Powered Voice Phishing Detection

## 1. Project Overview & Setup
- **Objective**: Build an end-to-end, modular, portfolio-grade system detecting voice phishing (vishing) calls by combining acoustic anomaly detection, offline-capable speech recognition, transparent scam cue lexical analysis, and streaming risk score fusion.
- **Milestone 1**: Initialize repository (`git init`, `.gitignore`, `LICENSE` (MIT), initial `README.md`, `pyproject.toml`, `requirements.txt`).
- **Milestone 2**: Setup Python virtual environment (Python 3.11 with `uv`) and install dependencies: `pydub`, `librosa`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `soundfile`, `vosk`/`faster-whisper`, `pyyaml`, `pytest`, `streamlit`, `rich`, `typer`/`argparse`, `joblib`.
- **Milestone 3**: Connect GitHub repository using `gh` (`voice-phishing-detection`, public repo under `Adhiraj2601`).

## 2. Core Modules Implementation (`src/vishing_detector/`)
- **Audio Ingestion & Preprocessing** (`audio/loader.py`):
  - PyDub + soundfile loading for WAV/MP3/M4A/OGG.
  - Conversion to 16 kHz mono WAV, loudness normalization (LUFS / peak), silence trimming.
  - Windowing into overlapping chunks (e.g. 3.0s window, 1.0s hop) for streaming simulation.
- **Acoustic Feature Extraction** (`features/acoustic.py`):
  - Librosa extraction: MFCCs (13 coefficients + $\Delta$ + $\Delta\Delta$), chroma, spectral centroid, spectral bandwidth, spectral rolloff, spectral contrast, zero-crossing rate (ZCR), RMS energy, pitch tracking (PYIN / YIN), pause ratio and speech rate estimation.
  - Formats: fixed-length feature vectors with column names and chunk timestamps.
- **Speech Recognition (ASR)** (`asr/transcriber.py`):
  - Swappable interface: `Transcriber` base class with concrete implementations (Vosk and/or Faster-Whisper, plus a lightweight Mock/Fallback transcriber for testing and quick offline evaluation).
  - Time-aligned transcription segments.
- **Scam Cue NLP Analyzer** (`nlp/scam_cues.py`):
  - Configurable regex/lexical matching from YAML (`config/scam_lexicon.yaml`).
  - Cue categories: Urgency & Threats, Credential harvesting (OTP, PIN, CVV, passwords), Impersonation (IRS/police/bank/tech support), Financial transfer (gift cards, wire, crypto), Remote Access (AnyDesk, TeamViewer), Secrecy ("don't tell anyone").
  - Weighted scoring per chunk and cumulative trajectory.
  - Optional TF-IDF + Logistic Regression text classifier for scam likelihood.
- **Acoustic Anomaly Detection** (`anomaly/detector.py`):
  - Unsupervised models trained on benign speech: `IsolationForest` and `OneClassSVM`.
  - Feature standardizer (`StandardScaler`), model serialization (`joblib`), threshold calibration, anomaly scoring normalized to [0, 1].
- **Multi-Modal Risk Scorer & Explanations** (`fusion/scorer.py`):
  - Weighted fusion of acoustic anomaly score and scam text score into a unified 0-100 Threat Index.
  - Generates transparent, human-readable explanations: trigger cues with timestamps, peak anomalous acoustic chunks, risk category (Low, Elevated, Critical).
- **Streaming Pipeline** (`pipeline.py`):
  - Simulates real-time chunk-by-chunk call processing.
  - Emits real-time timeline events (chunk start/end, risk score, alert trigger, cue tags).

## 3. Data & Evaluation
- **Synthetic Data Generation** (`scripts/generate_synthetic_data.py`):
  - Ethically generate synthetic benign conversational samples and scripted scam samples using programmatic speech synthesis / TTS with acoustic variation (pitch, tempo, background noise).
  - Documentation in `data/README.md` with guidelines for LibriSpeech/Mozilla Common Voice.
- **Evaluation Engine** (`evaluate.py`):
  - Precision, Recall, F1, ROC-AUC, Confusion Matrix, and Detection Latency (Time to First Alert).
  - Generation of portfolio figures (`docs/figures/roc_curve.png`, `risk_timeline.png`, `anomaly_distribution.png`).

## 4. Interfaces & Demonstrations
- **CLI** (`cli.py`): Commands for `train`, `detect`, `stream`, `evaluate`.
- **Streamlit Demo** (`app/streamlit_app.py`): Interactive UI for uploading call audio, visualizing waveform, streaming transcript, highlighted scam cues, and risk score timeline.
- **Jupyter Notebook** (`notebooks/demo.ipynb`): End-to-end tutorial walkthrough.

## 5. Quality, Testing & CI/CD
- Unit tests (`tests/test_loader.py`, `tests/test_acoustic.py`, `tests/test_cues.py`, `tests/test_detector.py`, `tests/test_fusion.py`, `tests/test_pipeline.py`).
- GitHub Actions CI (`.github/workflows/ci.yml`).
- Comprehensive `README.md` with Mermaid architecture diagram, benchmarks, installation guide, ethical considerations, and MIT license.
- Automated Conventional Commits and GitHub Release tag `v0.1.0`.
