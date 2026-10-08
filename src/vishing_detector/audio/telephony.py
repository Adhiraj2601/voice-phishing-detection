"""Telephony channel simulation: 8 kHz downsampling, G.711 mu-law compression, bandpass filtering, and line noise."""

from __future__ import annotations

import logging
import random
from typing import Optional

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)


def apply_telephony_simulation(
    audio: np.ndarray,
    sr: int = 16000,
    snr_db: Optional[float] = None,
    filter_low: float = 300.0,
    filter_high: float = 3400.0,
    target_output_sr: int = 16000,
) -> np.ndarray:
    """Simulate telephony transmission channel characteristics.

    Steps:
    1. Downsample to 8,000 Hz (PSTN standard telephony sampling).
    2. Telecom frequency bandpass filter (300 Hz - 3400 Hz, 4th order Butterworth).
    3. ITU-T G.711 mu-law companding & 8-bit quantization (mu=255).
    4. Telephone channel line noise (randomized 15 - 30 dB SNR + 60 Hz hum).
    5. Resample back to target_output_sr (16,000 Hz) for pipeline ingestion.

    Args:
        audio: Input 1D audio waveform array.
        sr: Input sampling rate in Hz (default 16000).
        snr_db: Signal-to-noise ratio in dB. If None, drawn uniformly from [15.0, 30.0].
        filter_low: Lower cutoff frequency for telecom passband (default 300.0 Hz).
        filter_high: Upper cutoff frequency for telecom passband (default 3400.0 Hz).
        target_output_sr: Output sampling rate in Hz (default 16000).

    Returns:
        Processed 1D audio waveform as float32 in range [-1.0, 1.0].
    """
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    # 1. Downsample to 8,000 Hz
    if sr != 8000:
        audio_8k = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=8000)
    else:
        audio_8k = audio.astype(np.float32).copy()

    # 2. Telephone band-pass filter (300 Hz - 3400 Hz)
    try:
        sos = butter(4, [filter_low, filter_high], btype="bandpass", fs=8000, output="sos")
        audio_8k = sosfilt(sos, audio_8k)
    except Exception as exc:
        logger.debug("Bandpass filtering fallback: %s", exc)

    # 3. ITU-T G.711 mu-law companding and 8-bit quantization
    # F(x) = sgn(x) * ln(1 + mu*|x|) / ln(1 + mu), mu=255
    mu = 255.0
    x = np.clip(audio_8k, -1.0, 1.0)
    companded = np.sign(x) * np.log(1.0 + mu * np.abs(x)) / np.log(1.0 + mu)
    quantized = np.round(companded * 127.0) / 127.0
    expanded = np.sign(quantized) * (1.0 / mu) * ((1.0 + mu) ** np.abs(quantized) - 1.0)
    audio_8k = expanded

    # 4. Telephony background line noise (white/pink hiss + 60 Hz hum)
    if snr_db is None:
        actual_snr = float(random.uniform(15.0, 30.0))
    else:
        actual_snr = float(snr_db)

    t_noise = np.linspace(0, len(audio_8k) / 8000.0, len(audio_8k), endpoint=False)
    hum = 0.003 * np.sin(2.0 * np.pi * 60.0 * t_noise)
    sig_power = float(np.mean(audio_8k**2)) + 1e-8
    noise_power = sig_power / (10.0 ** (actual_snr / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), len(audio_8k))
    audio_8k = audio_8k + noise + hum

    # 5. Resample back to target sampling rate (16 kHz)
    if target_output_sr != 8000:
        audio_out = librosa.resample(audio_8k, orig_sr=8000, target_sr=target_output_sr)
    else:
        audio_out = audio_8k

    return np.clip(audio_out, -1.0, 1.0).astype(np.float32)
