# AI-Powered Voice Phishing (Vishing) Detection

[![CI](https://github.com/Adhiraj2601/voice-phishing-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Adhiraj2601/voice-phishing-detection/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)

An end-to-end, multi-modal machine learning system engineered to detect voice phishing (**vishing**) calls in real-time. By fusing **acoustic feature anomaly detection** with **offline-capable speech-to-text** and **rule-based/lexical scam cue analysis**, this system generates an interpretable, streaming risk score (0–100) and pinpoints suspicious conversational triggers with timestamped explanations.

---

## Architecture Overview

```mermaid
flowchart TD
    A["Incoming Call Audio (WAV / MP3 / M4A)"] --> B["PyDub Audio Ingestion & Preprocessing"]
    B -->|"16 kHz Mono & Silence Trim"| C["Streaming Windowing (3s window, 1s hop)"]
    
    subgraph Multi-Modal Feature Extraction
        C --> D["Acoustic Feature Extractor (Librosa)"]
        C --> E["Offline Speech Recognition (ASR)"]
        
        D -->|"MFCCs, Chroma, Pitch, Spectral, Pause Ratio"| F["Acoustic Anomaly Detector (IsolationForest / OCSVM)"]
        E -->|"Time-aligned Transcripts"| G["Scam Cue NLP Analyzer (YAML Lexicon + Regex)"]
    end
    
    F -->|"Acoustic Anomaly Score (0-1)"| H["Multi-Modal Risk Scorer & Explainer"]
    G -->|"Lexical Threat Score (0-1)"| H
    
    H --> I["Real-Time Risk Score (0-100)"]
    H --> J["Transparent Alert Explanations & Trigger Timestamps"]
    
    I --> K["CLI Monitor / Streamlit Dashboard"]
    J --> K
```

---

## Key Features

- **Robust Audio Ingestion**: Normalizes loudness, strips dead silence, and chunks incoming calls with overlapping rolling windows to simulate low-latency streaming telephony.
- **Rich Acoustic Profiling**: Extracts 40+ acoustic dimensions with Librosa, including MFCCs ($\Delta, \Delta\Delta$), chroma, spectral rolloff/contrast/centroid, zero-crossing rate, RMS energy, fundamental frequency ($F_0$), and speech-to-pause ratios.
- **Unsupervised Anomaly Modeling**: Uses `IsolationForest` and `One-Class SVM` trained exclusively on normal conversations to flag high-stress, rapid, robotic, or unnatural vocal acoustics without requiring labeled attack audio.
- **Explainable Scam Cue Analysis**: Identifies coercion, urgency, credential harvesting (OTP/PIN/CVV/SSN), authority impersonation (IRS/police/bank), financial demands (crypto/gift cards), remote desktop requests, and secrecy tactics.
- **Streaming Threat Scoring**: Computes a continuous, smoothed Threat Index (0–100) with configurable alerts: *Low*, *Elevated*, and *Critical*.
- **Interactive Visualizations**: Includes a full Streamlit dashboard with real-time waveform inspection, risk progression graphs, and highlighted transcripts.

---

## Quickstart & Installation

### Prerequisites
- Python 3.10 or higher
- [FFmpeg](https://ffmpeg.org/) (for PyDub audio decoding)

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/Adhiraj2601/voice-phishing-detection.git
cd voice-phishing-detection
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest -v
```

### 3. Launch Demo UI
```bash
streamlit run src/vishing_detector/app/streamlit_app.py
```
