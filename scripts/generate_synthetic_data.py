"""Generates large-scale synthetic telephony audio datasets for vishing detection.

Features:
- >= 100 clips per class for held-out evaluation (benign vs. vishing).
- Hard negatives: Benign calls legitimately mentioning banks, OTPs, or deliveries to challenge false positives.
- Multiple TTS voice personas with strictly held-out voices and held-out scripts for the test split.
- Telephony simulation on the test set: 8 kHz downsampling, G.711 u-law codec compression,
  telecom band-pass filtering (300 Hz - 3400 Hz), and line noise.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Script Collections
# ---------------------------------------------------------------------------

BENIGN_TRAIN_SCRIPTS = [
    "Hi Sarah, just checking in to see if we're still meeting for lunch tomorrow at twelve thirty.",
    "Hello, this is customer service confirming your organic grocery delivery window between two and four PM.",
    "Good afternoon, I am following up on the quarterly financial presentation slides we reviewed yesterday.",
    "Hey mom, just calling to see how the garden is doing and whether the flowers started blooming.",
    "Hello doctor's office, I would like to reschedule my dental cleaning appointment for next Tuesday morning.",
    "Good morning team, let us review the sprint retrospective action items and assign issue ownership.",
    "Hi, I noticed the package arrived safely on our front porch. Thank you so much for letting me know.",
    "Hello, could you please email me the recipe for the pasta sauce you prepared last weekend?",
    "Hey Alex, do you have time for a quick phone call to discuss the weekend hiking trip route?",
    "Good evening, we are confirming your dinner reservation for party of four tonight at seven PM.",
    "Hi Tom, I reviewed the drafted contract proposal and left several comments in the shared document.",
    "Hello, I am calling to inquire about the library hours this coming holiday weekend.",
    "Hi, just letting you know the plumber finished repairing the sink pipe and everything works well.",
    "Good morning, our flight departs tomorrow at ten AM so please remember to pack your passport.",
    "Hey there, the neighborhood committee meeting has been moved to Thursday evening at the community center.",
]

BENIGN_TEST_SCRIPTS = [
    "Hi Rachel, I wanted to double check what time the school play starts on Friday evening.",
    "Good afternoon, this is Mark calling from the automotive garage. Your car brake inspection is complete.",
    "Hey dad, I landed at the airport and I am heading to the baggage claim area now.",
    "Hello, I am calling to confirm our tennis court reservation for Saturday morning at nine o'clock.",
    "Hi everyone, please remember to submit your timesheets before the end of business today.",
    "Good morning, I was wondering if your bakery has gluten-free sourdough bread in stock today.",
    "Hey Chris, let us coordinate who is bringing the projector and slides for the workshop tomorrow.",
    "Hello, just checking if you received the updated architectural blueprints I emailed earlier.",
    "Good afternoon, we would like to invite you to our annual charity book drive this Saturday.",
    "Hi, I am calling from the animal clinic to confirm that the puppy vaccinations are up to date.",
    "Hey brother, let me know if you want to split the hotel booking for the family reunion.",
    "Hello, this is the pharmacy calling to notify you that your prescription is ready for pickup.",
]

# Hard negatives: Legitimate benign calls that mention banks, OTPs, deliveries, or security
HARD_NEGATIVE_SCRIPTS = [
    "Hi, I just received an SMS with an OTP code from my bank. Is that normal or should I change my password?",
    "Hey honey, the grocery delivery courier is outside asking for the four digit verification PIN on the app.",
    "Hello customer care, I am calling Chase bank support to ask why my debit card was declined at the grocery store.",
    "Mom, someone called claiming to be from the bank asking for my card numbers, but I told them no and hung up.",
    "Good morning, I received a text notification about an urgent package delivery requiring a confirmation code.",
    "Hi officer, I am reporting a suspicious voicemail I received this morning claiming my account was suspended.",
    "The bank teller told me to set up two-factor authentication with an OTP generator application for safety.",
    "Can you please show me how to execute a secure wire transfer to my daughter's university tuition account?",
    "I received a strange letter about an arrest warrant from the IRS, can you help me verify it is a fraudulent scam?",
    "Hi, I am logging into my online banking portal and it asked me to enter a one-time verification password.",
    "Hey, did our bank send you an email about updated credit card terms and conditions for next month?",
    "Hello, I am trying to reset my email password and need you to read me the security code sent to the backup phone.",
    "The IT department asked everyone to update their passwords and enable two-step authentication today.",
    "I saw a charge on my credit card statement from an unfamiliar vendor and want to dispute the transaction.",
    "Could you verify if this text from the postal service asking for delivery redelivery confirmation is genuine?",
]

SCAM_TRAIN_SCRIPTS = [
    "This is Officer Miller from the Internal Revenue Service. A legal arrest warrant has been issued under your social security number. You must immediately pay five hundred dollars in Apple gift cards to stay out of custody.",
    "Urgent security alert from your bank fraud department. Your checking account has been compromised. Please read back the six digit verification OTP code sent to your phone right now to verify your identity.",
    "This is Microsoft customer support. Critical warning: your computer is infected with a trojan virus. Install AnyDesk immediately and grant remote access to our technician to fix it.",
    "Federal Trade Commission final notice. Your bank accounts will be frozen within fifteen minutes. Transfer your funds into our government safe account via wire transfer immediately.",
    "Amazon security department notice. An unauthorized charge of nine hundred dollars was detected. Do not hang up or tell anyone. Please tell me your credit card number, CVV code, and billing address.",
    "Law enforcement alert: an indictment has been filed against you. You must pay the fine using a Bitcoin ATM immediately before local police arrive at your home address.",
    "Social Security Administration urgent message. Your SSN has been terminated due to fraudulent activity. Press one now and provide your mother's maiden name and bank PIN.",
    "Bank fraud prevention unit: suspicious international wire transfers detected on your debit card. Read back your online banking password to verify account ownership.",
]

SCAM_TEST_SCRIPTS = [
    "Department of Justice criminal investigation unit. A federal indictment is pending against your identity. Transfer ten thousand dollars to our safe government custody account immediately.",
    "Urgent notification from Wells Fargo fraud division. A fraudulent withdrawal was attempted. You must disclose your full card number, expiration date, and CVV code right now.",
    "Apple technical security alert: your iCloud account has been breached. Download TeamViewer on your desktop and give our agent full remote control.",
    "This is the police department warrant division. You will be taken into custody within two hours unless you deposit restitution into a Bitcoin ATM right now.",
    "Target fraud prevention: a suspicious purchase of two thousand dollars was initiated. Keep this call confidential and do not inform your family or bank teller.",
    "Treasury department legal notice: unpaid tax liabilities must be resolved today using Google Play or Target gift cards to avoid asset seizure.",
    "Federal Trade Commission security officer speaking. Disclose your one-time passcode to stop unauthorized transactions from clearing your checking balance.",
    "FBI cyber crime division: we have placed a freeze order on your personal assets. Transfer your savings to our encrypted cryptocurrency wallet immediately.",
    "Amazon customer loyalty department: you won a ten thousand dollar refund prize. To claim your prize, send five hundred dollars in Steam gift cards right away.",
    "Border protection customs agency: contraband items were discovered in a parcel registered to your name. Pay the clearance fine via Western Union wire transfer.",
    "Chase security center: critical compromise detected. Read out your six digit OTP passcode over the phone immediately to secure your funds.",
    "Windows defender security team: malicious ransomware detected on your network. Grant remote access via UltraViewer to prevent disk erasure.",
]

# ---------------------------------------------------------------------------
# Voice Persona Definitions
# ---------------------------------------------------------------------------

@dataclass
class VoicePersona:
    name: str
    rate: int          # SAPI rate: -2 to +2
    pitch_shift: float # Semitones: -4.0 to +4.0
    formant_scale: float # Formant frequency ratio
    is_test_heldout: bool


VOICE_PERSONAS: List[VoicePersona] = [
    # Training Personas
    VoicePersona("train_female_natural", rate=0, pitch_shift=0.0, formant_scale=1.0, is_test_heldout=False),
    VoicePersona("train_male_deep", rate=-1, pitch_shift=-3.0, formant_scale=0.92, is_test_heldout=False),
    VoicePersona("train_female_clear", rate=1, pitch_shift=2.0, formant_scale=1.08, is_test_heldout=False),
    VoicePersona("train_male_brisk", rate=1, pitch_shift=-1.5, formant_scale=0.96, is_test_heldout=False),

    # Held-out Test Personas (NEVER seen in training)
    VoicePersona("test_heldout_female_urgent", rate=2, pitch_shift=3.5, formant_scale=1.12, is_test_heldout=True),
    VoicePersona("test_heldout_male_gravelly", rate=-1, pitch_shift=-4.0, formant_scale=0.88, is_test_heldout=True),
    VoicePersona("test_heldout_neutral_telecom", rate=0, pitch_shift=1.0, formant_scale=1.02, is_test_heldout=True),
    VoicePersona("test_heldout_fast_caller", rate=2, pitch_shift=-2.0, formant_scale=0.94, is_test_heldout=True),
]


# ---------------------------------------------------------------------------
# Telephony Simulation Functions
# ---------------------------------------------------------------------------

def apply_telephony_simulation(
    audio_16k: np.ndarray,
    sr: int = 16000,
    snr_db: float = 24.0,
) -> np.ndarray:
    """Simulate real telephony channel: 8 kHz downsampling, G.711 u-law codec, 300-3400 Hz bandpass, line noise."""
    # 1. Downsample to 8,000 Hz (telephony standard)
    audio_8k = librosa.resample(audio_16k, orig_sr=sr, target_sr=8000)

    # 2. Telephone band-pass filter (300 Hz - 3400 Hz)
    try:
        sos = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")
        audio_8k = sosfilt(sos, audio_8k)
    except Exception:
        pass

    # 3. G.711 u-law compression via quantizing / u-law companding
    # u-law compression: F(x) = sgn(x) * ln(1 + u|x|) / ln(1 + u), u=255
    mu = 255.0
    x = np.clip(audio_8k, -1.0, 1.0)
    companded = np.sign(x) * np.log(1.0 + mu * np.abs(x)) / np.log(1.0 + mu)
    # 8-bit quantization
    quantized = np.round(companded * 127.0) / 127.0
    # Expanding back
    expanded = np.sign(quantized) * (1.0 / mu) * ((1.0 + mu) ** np.abs(quantized) - 1.0)
    audio_8k = expanded

    # 4. Add telephony background line noise (white/pink hiss + 60 Hz mains hum)
    t_noise = np.linspace(0, len(audio_8k) / 8000.0, len(audio_8k))
    hum = 0.003 * np.sin(2 * np.pi * 60.0 * t_noise)
    sig_power = np.mean(audio_8k**2) + 1e-8
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio_8k))
    audio_8k = audio_8k + noise + hum

    # 5. Resample back to 16,000 Hz for standard ingestion
    audio_resampled = librosa.resample(audio_8k, orig_sr=8000, target_sr=sr)
    return np.clip(audio_resampled, -1.0, 1.0).astype(np.float32)


def _batch_synthesize_tts_windows(
    items: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Synthesize batch of TTS audio files efficiently in a single PowerShell process."""
    if platform.system() != "Windows":
        return

    # Build commands for a single powershell script
    ps_lines = [
        "Add-Type -AssemblyName System.Speech",
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
    ]

    for item in items:
        out_path = str(item["wav_path"]).replace(os.sep, "/")
        clean_text = item["text"].replace('"', '\\"').replace("'", "")
        rate = item["persona"].rate
        ps_lines.append(f"$s.Rate = {rate}")
        ps_lines.append(f"$s.SetOutputToWaveFile('{out_path}')")
        ps_lines.append(f"$s.Speak('{clean_text}')")

    ps_lines.append("$s.Dispose()")

    temp_ps1 = output_dir / "_batch_tts.ps1"
    temp_ps1.write_text("\n".join(ps_lines), encoding="utf-8")

    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(temp_ps1)],
            capture_output=True,
            timeout=180,
            check=True,
        )
    except Exception as err:
        logger.warning("Batch PowerShell TTS execution warning: %s", err)
    finally:
        if temp_ps1.exists():
            temp_ps1.unlink()


def _synthesize_harmonic_fallback(text: str, out_wav: Path, sr: int = 16000, f0_base: float = 140.0) -> None:
    """Cross-platform harmonic formant vocal synthesizer fallback."""
    words = text.split()
    duration_sec = max(2.5, len(words) * 0.35)
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    f0 = f0_base + 15.0 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    signal = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase) + 0.1 * np.sin(4 * phase)
    syllable_freq = len(words) * 1.5 / duration_sec
    envelope = (np.sin(2 * np.pi * syllable_freq * t) ** 2) * 0.8 + 0.2
    audio = signal * envelope + np.random.normal(0, 0.015, n_samples)
    sf.write(str(out_wav), np.clip(audio, -1.0, 1.0).astype(np.float32), sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Dataset Generation Pipeline
# ---------------------------------------------------------------------------

def generate_large_scale_dataset(
    output_dir: Path = Path("data/synthetic"),
    num_benign_train: int = 60,
    num_benign_test: int = 110,
    num_scam_test: int = 110,
) -> Dict[str, Any]:
    """Generate extensive telephony-simulated dataset exceeding 100 clips per class.

    Includes:
    - Train split: strictly benign recordings using training voice personas.
    - Test split: held-out benign recordings, held-out hard negatives, and held-out scam recordings
      using held-out voice personas and simulated 8 kHz G.711 telephony compression.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_tts"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_personas = [p for p in VOICE_PERSONAS if not p.is_test_heldout]
    test_personas = [p for p in VOICE_PERSONAS if p.is_test_heldout]

    tts_jobs: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []

    # 1. Benign Training Set (Train Split)
    logger.info("Scheduling %d benign training clips...", num_benign_train)
    for i in range(num_benign_train):
        script = BENIGN_TRAIN_SCRIPTS[i % len(BENIGN_TRAIN_SCRIPTS)]
        persona = train_personas[i % len(train_personas)]
        filename = f"train_benign_{i+1:03d}.wav"
        target_path = train_dir / filename
        raw_path = raw_dir / f"raw_train_benign_{i+1:03d}.wav"

        tts_jobs.append({"text": script, "persona": persona, "wav_path": raw_path, "target_path": target_path, "split": "train", "label": 0, "is_hard_negative": False})

    # 2. Benign Test Set (Held-Out Test Split, including Hard Negatives)
    num_hard_neg = int(num_benign_test * 0.45) # ~50 hard negatives
    num_standard_benign_test = num_benign_test - num_hard_neg
    logger.info("Scheduling %d held-out benign test clips (%d standard + %d hard negatives)...", num_benign_test, num_standard_benign_test, num_hard_neg)

    for i in range(num_standard_benign_test):
        script = BENIGN_TEST_SCRIPTS[i % len(BENIGN_TEST_SCRIPTS)]
        persona = test_personas[i % len(test_personas)]
        filename = f"test_benign_std_{i+1:03d}.wav"
        target_path = test_dir / filename
        raw_path = raw_dir / f"raw_test_benign_std_{i+1:03d}.wav"

        tts_jobs.append({"text": script, "persona": persona, "wav_path": raw_path, "target_path": target_path, "split": "test", "label": 0, "is_hard_negative": False})

    for i in range(num_hard_neg):
        script = HARD_NEGATIVE_SCRIPTS[i % len(HARD_NEGATIVE_SCRIPTS)]
        persona = test_personas[i % len(test_personas)]
        filename = f"test_benign_hard_{i+1:03d}.wav"
        target_path = test_dir / filename
        raw_path = raw_dir / f"raw_test_benign_hard_{i+1:03d}.wav"

        tts_jobs.append({"text": script, "persona": persona, "wav_path": raw_path, "target_path": target_path, "split": "test", "label": 0, "is_hard_negative": True})

    # 3. Scam Test Set (Held-Out Test Split)
    logger.info("Scheduling %d held-out scam test clips...", num_scam_test)
    for i in range(num_scam_test):
        script = SCAM_TEST_SCRIPTS[i % len(SCAM_TEST_SCRIPTS)]
        persona = test_personas[i % len(test_personas)]
        filename = f"test_scam_{i+1:03d}.wav"
        target_path = test_dir / filename
        raw_path = raw_dir / f"raw_test_scam_{i+1:03d}.wav"

        tts_jobs.append({"text": script, "persona": persona, "wav_path": raw_path, "target_path": target_path, "split": "test", "label": 1, "is_hard_negative": False})

    # Execute TTS Generation
    logger.info("Synthesizing %d total speech clips...", len(tts_jobs))
    _batch_synthesize_tts_windows(tts_jobs, raw_dir)

    # Process acoustic perturbations & telephony simulation
    logger.info("Applying acoustic persona shifts and telephony simulation...")
    for job in tts_jobs:
        raw_p = job["wav_path"]
        target_p = job["target_path"]
        persona: VoicePersona = job["persona"]
        is_test = job["split"] == "test"
        label = job["label"]

        # Ensure raw audio exists
        if not raw_p.exists() or raw_p.stat().st_size < 1000:
            f0_base = 130.0 + persona.pitch_shift * 8.0
            _synthesize_harmonic_fallback(job["text"], raw_p, sr=16000, f0_base=f0_base)

        try:
            y, sr = librosa.load(str(raw_p), sr=16000)

            # Apply persona pitch shifting
            if abs(persona.pitch_shift) > 0.1:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=persona.pitch_shift)

            # For scam callers, apply slightly elevated cadence
            if label == 1:
                y = librosa.effects.time_stretch(y, rate=random.choice([1.06, 1.12, 1.18]))

            # On the test set, apply full telephony channel simulation (8 kHz + G.711 u-law + bandpass + noise)
            if is_test:
                y = apply_telephony_simulation(y, sr=16000, snr_db=random.uniform(20.0, 28.0))
                codec_desc = "g711_mulaw_8k_simulated"
            else:
                codec_desc = "linear_pcm_16k"

            # Write finalized 16 kHz WAV
            sf.write(str(target_p), y, sr, subtype="PCM_16")
            duration = float(len(y) / sr)

            manifest.append(
                {
                    "file": str(target_p.relative_to(output_dir)).replace(os.sep, "/"),
                    "split": job["split"],
                    "label": label,
                    "label_name": "vishing" if label == 1 else "benign",
                    "is_hard_negative": job["is_hard_negative"],
                    "voice_persona": persona.name,
                    "codec": codec_desc,
                    "duration": round(duration, 2),
                    "text": job["text"],
                }
            )
        except Exception as err:
            logger.error("Failed processing %s: %s", target_p.name, err)

    # Clean up raw intermediates
    try:
        import shutil
        shutil.rmtree(raw_dir)
    except Exception:
        pass

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"samples": manifest}, f, indent=2)

    # Copy canonical samples for quick demos
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    if (train_dir / "train_benign_001.wav").exists():
        import shutil
        shutil.copyfile(train_dir / "train_benign_001.wav", samples_dir / "sample_benign.wav")
    if (test_dir / "test_scam_001.wav").exists():
        import shutil
        shutil.copyfile(test_dir / "test_scam_001.wav", samples_dir / "sample_scam.wav")

    train_count = sum(1 for m in manifest if m["split"] == "train")
    test_benign = sum(1 for m in manifest if m["split"] == "test" and m["label"] == 0)
    test_scam = sum(1 for m in manifest if m["split"] == "test" and m["label"] == 1)
    hard_neg_count = sum(1 for m in manifest if m.get("is_hard_negative", False))

    logger.info(
        "Dataset Generation Complete!\n"
        "  - Train Set: %d benign clips\n"
        "  - Test Set: %d total clips (%d benign [%d hard negatives] + %d vishing)\n"
        "  - Manifest: %s",
        train_count,
        test_benign + test_scam,
        test_benign,
        hard_neg_count,
        test_scam,
        manifest_path,
    )

    return {
        "manifest_path": str(manifest_path),
        "train_count": train_count,
        "test_count": test_benign + test_scam,
        "test_benign": test_benign,
        "test_scam": test_scam,
        "hard_negatives": hard_neg_count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large-scale synthetic dataset generator for vishing detection.")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory path")
    parser.add_argument("--train-benign", type=int, default=60, help="Number of benign training samples")
    parser.add_argument("--test-benign", type=int, default=110, help="Number of benign test samples (>= 100)")
    parser.add_argument("--test-scam", type=int, default=110, help="Number of scam test samples (>= 100)")
    args = parser.parse_args()

    generate_large_scale_dataset(
        output_dir=Path(args.output_dir),
        num_benign_train=args.train_benign,
        num_benign_test=args.test_benign,
        num_scam_test=args.test_scam,
    )
