# Dataset Documentation & Guidelines

This project analyzes voice phishing (vishing) and acoustic conversation patterns in telephony audio.

## Ethical Standards & Privacy Policy
**Strict Policy**: We do **NOT** scrape, record, or distribute recordings of real victims or non-consenting individuals. 
Real scam recordings frequently contain sensitive personal identifying information (PII), including victim names, phone numbers, addresses, account credentials, and distressed vocal patterns. Distributing such audio without strict consent is unethical and may violate wiretapping laws and GDPR/CCPA regulations.

To ensure ethical, reproducible, and compliant research:
1. All demonstration samples committed to this repository are synthetic dialogues generated programmatically using text-to-speech engines and audio synthesis scripts.
2. Large audio files (> 10 MB) are excluded from git tracking via `.gitignore`.

---

## Plugging In Public Datasets

You can train and evaluate the acoustic anomaly detector and speech transcriber on standard, ethically sourced speech datasets:

### 1. Benign Telephone & Conversational Speech
- **LibriSpeech ASR Corpus** (CC BY 4.0):
  - Large collection of 16 kHz read English speech derived from LibriVox audiobooks.
  - Download: [openslr.org/12](https://www.openslr.org/12/)
  - Recommended subset for anomaly baseline: `train-clean-100` (100 hours of clear speech).
- **Mozilla Common Voice** (CC-0 / Public Domain):
  - Crowdsourced multilingual speech across thousands of speakers and acoustic environments.
  - Download: [commonvoice.mozilla.org](https://commonvoice.mozilla.org/)
- **VoxCeleb / Switchboard Telephone Corpus**:
  - For simulating 8 kHz telephony codecs and band-pass filtering (G.711 / AMR).

### 2. Vishing & Scam Text Transcripts
- **FTC Scam Call Transcripts / Consumer Sentinel Network**:
  - Public transcripts published by the U.S. Federal Trade Commission (FTC) illustrating common imposter scams, IRS impersonation, and tech-support scams.
- **Scam-Halt / Telephony Fraud Corpus Transcripts**:
  - Academic corpora containing anonymized scam call dialogue transcripts for text cue validation.

---

## Directory Organization

```text
data/
├── README.md               # Dataset documentation & ethical guidelines
├── samples/                # Lightweight test samples for tests & demo (< 2 MB)
│   ├── sample_benign.wav   # Synthetic normal conversation sample
│   └── sample_scam.wav     # Synthetic vishing call sample
├── synthetic/              # Generated via scripts/generate_synthetic_data.py (gitignored)
└── raw/                    # User-downloaded external corpora (gitignored)
```

## Running Synthetic Data Generation
To generate fresh benign and scam synthetic audio samples:
```bash
python scripts/generate_synthetic_data.py --num-samples 20 --output-dir data/synthetic
```
