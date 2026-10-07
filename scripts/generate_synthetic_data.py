"""Generates synthetic benign conversational audio and scripted vishing (scam) audio.

Uses Windows System.Speech SAPI5 (offline) when available, with a cross-platform
harmonic formant acoustic fallback for Linux/CI, and applies acoustic variations
(telephony filtering, pitch/speed perturbations, line noise).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BENIGN_SCRIPTS = [
    "Hi Sarah, just calling to see if we're still meeting for lunch tomorrow at twelve thirty.",
    "Hello, this is David from customer support confirming your grocery delivery window between two and four PM.",
    "Good afternoon, I am following up on the quarterly report slides we discussed yesterday afternoon.",
    "Hey mom, just checking in to see how your garden is doing and if you received the photos I sent.",
    "Hello doctor's office, I would like to reschedule my routine dental checkup for next Tuesday if possible.",
    "Good morning team, let us review the sprint retrospective action items and assign ownership.",
    "Hi, I noticed the package arrived on my porch. Thank you very much for letting me know.",
    "Hello, could you please email me the recipe for the pasta sauce you made last weekend?",
]

SCAM_SCRIPTS = [
    "This is Officer Miller from the Internal Revenue Service. A legal arrest warrant has been issued under your social security number. You must immediately pay five hundred dollars in Apple gift cards to stay out of custody.",
    "Urgent security alert from your bank fraud department. Your checking account has been compromised. Please read back the six digit verification OTP code sent to your phone right now to verify your identity.",
    "This is Microsoft customer support. Critical warning: your computer is infected with a trojan virus. Install AnyDesk immediately and grant remote access to our technician to fix it.",
    "Federal Trade Commission final notice. Your bank accounts will be frozen within fifteen minutes. Transfer your funds into our government safe account via wire transfer immediately.",
    "Amazon security department notice. An unauthorized charge of nine hundred dollars was detected. Do not hang up or tell anyone. Please tell me your credit card number, CVV code, and billing address.",
    "Law enforcement alert: an indictment has been filed against you. You must pay the fine using a Bitcoin ATM immediately before local police arrive at your home address.",
]


def _synthesize_tts_windows(text: str, out_wav: Path) -> bool:
    """Synthesize speech using Windows built-in SAPI5 SpeechSynthesizer."""
    if platform.system() != "Windows":
        return False
    try:
        clean_text = text.replace('"', '\\"').replace("'", "")
        ps_cmd = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {random.randint(-1, 2)}; "
            f"$s.SetOutputToWaveFile('{str(out_wav).replace(os.sep, '/')}'); "
            f"$s.Speak('{clean_text}'); "
            f"$s.Dispose()"
        )
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=15)
        return res.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 1000
    except Exception as err:
        logger.debug("Windows SAPI synthesis failed: %s", err)
        return False


def _synthesize_harmonic_fallback(text: str, out_wav: Path, sr: int = 16000) -> None:
    """Cross-platform harmonic formant vocal synthesizer for CI/Linux environments without TTS."""
    words = text.split()
    duration_sec = max(2.5, len(words) * 0.4)
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Base pitch with natural intonation contour
    f0 = 130.0 + 15.0 * np.sin(2 * np.pi * 0.4 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    # Formant synthesis (vocal tract approximation)
    signal = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase) + 0.1 * np.sin(4 * phase)

    # Modulate envelope with pseudo-syllable bursts
    syllable_freq = len(words) * 1.5 / duration_sec
    envelope = (np.sin(2 * np.pi * syllable_freq * t) ** 2) * 0.8 + 0.2
    signal = signal * envelope

    # Add light acoustic breath noise
    noise = np.random.normal(0, 0.02, n_samples)
    audio = signal + noise
    audio = np.clip(audio, -1.0, 1.0)

    sf.write(str(out_wav), audio, sr, subtype="PCM_16")


def add_acoustic_perturbations(
    wav_path: Path,
    is_scam: bool = False,
    apply_telephone_codec: bool = True,
) -> None:
    """Apply pitch, tempo, noise, and telephony bandpass perturbations."""
    try:
        y, sr = librosa.load(str(wav_path), sr=16000)

        # Scammers often exhibit either rushed speech (high tempo) or artificial pitch manipulation
        if is_scam:
            speed_factor = random.choice([1.08, 1.15, 1.20])
            y = librosa.effects.time_stretch(y, rate=speed_factor)
            n_steps = random.choice([-2, 1, 2, 3])
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

        # Add telephone line band-pass effect (300 Hz - 3400 Hz)
        if apply_telephone_codec and len(y) > 512:
            try:
                from scipy.signal import butter, sosfilt
                sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
                y = sosfilt(sos, y)
            except Exception:
                pass

        # Add telephony background hiss / ambient noise
        noise_level = random.uniform(0.003, 0.012)
        noise = np.random.normal(0, noise_level, len(y))
        y = y + noise
        y = np.clip(y, -0.99, 0.99)

        sf.write(str(wav_path), y, sr, subtype="PCM_16")
    except Exception as err:
        logger.warning("Could not apply perturbations to %s: %s", wav_path.name, err)


def generate_dataset(
    output_dir: Path = Path("data/synthetic"),
    num_benign: int = 8,
    num_scam: int = 8,
) -> Dict[str, Any]:
    """Generate balanced synthetic benign and scam datasets with metadata manifest."""
    benign_dir = output_dir / "benign"
    scam_dir = output_dir / "scam"
    benign_dir.mkdir(parents=True, exist_ok=True)
    scam_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []

    logger.info("Generating %d benign speech samples...", num_benign)
    for i in range(num_benign):
        text = BENIGN_SCRIPTS[i % len(BENIGN_SCRIPTS)]
        out_wav = benign_dir / f"benign_{i+1:03d}.wav"
        success = _synthesize_tts_windows(text, out_wav)
        if not success:
            _synthesize_harmonic_fallback(text, out_wav)
        add_acoustic_perturbations(out_wav, is_scam=False)

        duration = librosa.get_duration(path=str(out_wav))
        manifest.append(
            {
                "file": str(out_wav.relative_to(output_dir)),
                "label": 0,
                "label_name": "benign",
                "text": text,
                "duration": round(duration, 2),
            }
        )

    logger.info("Generating %d vishing (scam) speech samples...", num_scam)
    for i in range(num_scam):
        text = SCAM_SCRIPTS[i % len(SCAM_SCRIPTS)]
        out_wav = scam_dir / f"scam_{i+1:03d}.wav"
        success = _synthesize_tts_windows(text, out_wav)
        if not success:
            _synthesize_harmonic_fallback(text, out_wav)
        add_acoustic_perturbations(out_wav, is_scam=True)

        duration = librosa.get_duration(path=str(out_wav))
        manifest.append(
            {
                "file": str(out_wav.relative_to(output_dir)),
                "label": 1,
                "label_name": "vishing",
                "text": text,
                "duration": round(duration, 2),
            }
        )

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"samples": manifest}, f, indent=2)

    # Also export two canonical reference samples for immediate testing/demos
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    ref_benign = samples_dir / "sample_benign.wav"
    ref_scam = samples_dir / "sample_scam.wav"
    if (benign_dir / "benign_001.wav").exists():
        import shutil
        shutil.copyfile(benign_dir / "benign_001.wav", ref_benign)
    if (scam_dir / "scam_001.wav").exists():
        import shutil
        shutil.copyfile(scam_dir / "scam_001.wav", ref_scam)

    logger.info("Generated %d total synthetic audio files in %s", len(manifest), output_dir)
    return {"manifest_path": str(manifest_path), "total_samples": len(manifest)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic audio dataset generator for vishing detection.")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory path")
    parser.add_argument("--num-benign", type=int, default=8, help="Number of benign samples to generate")
    parser.add_argument("--num-scam", type=int, default=8, help="Number of vishing samples to generate")
    args = parser.parse_args()

    generate_dataset(Path(args.output_dir), args.num_benign, args.num_scam)
