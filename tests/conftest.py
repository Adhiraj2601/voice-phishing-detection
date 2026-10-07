"""Pytest fixtures and synthetic test audio generators."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sample_rate() -> int:
    return 16000


@pytest.fixture
def synthetic_sine_wav(sample_rate: int) -> Generator[Path, None, None]:
    """Create a temporary 4.0-second 440 Hz pure tone WAV file at 16 kHz."""
    duration_sec = 4.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Sine wave with varying envelope
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) * (1.0 + 0.2 * np.sin(2 * np.pi * 2 * t))
    audio_int16 = (signal * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_int16, sample_rate, subtype="PCM_16")
        tmp_path = Path(tmp.name)

    yield tmp_path

    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def short_wav(sample_rate: int) -> Generator[Path, None, None]:
    """Create a temporary 1.0-second WAV file."""
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * 880 * t)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, (signal * 32767).astype(np.int16), sample_rate, subtype="PCM_16")
        tmp_path = Path(tmp.name)

    yield tmp_path

    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def silent_wav(sample_rate: int) -> Generator[Path, None, None]:
    """Create a temporary silent WAV file."""
    duration_sec = 2.0
    silence = np.zeros(int(sample_rate * duration_sec), dtype=np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, silence, sample_rate, subtype="PCM_16")
        tmp_path = Path(tmp.name)

    yield tmp_path

    if tmp_path.exists():
        tmp_path.unlink()
