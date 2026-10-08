"""Unit tests for telephony channel simulation."""

import numpy as np

from vishing_detector.audio.telephony import apply_telephony_simulation


def test_telephony_simulation_basic():
    sr = 16000
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # 440 Hz test sine wave
    sine = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    simulated = apply_telephony_simulation(sine, sr=sr, snr_db=25.0)

    assert len(simulated) == len(sine)
    assert simulated.dtype == np.float32
    assert -1.0 <= simulated.min() and simulated.max() <= 1.0
    # Telephony noise and filtering alters the pristine sine wave
    assert not np.allclose(simulated, sine, atol=1e-3)


def test_telephony_simulation_empty():
    empty = np.zeros(0, dtype=np.float32)
    res = apply_telephony_simulation(empty, sr=16000)
    assert len(res) == 0


def test_telephony_simulation_randomized_snr():
    sr = 16000
    audio = np.random.normal(0, 0.2, sr).astype(np.float32)
    # Calling without explicit snr_db tests uniform random SNR in [15.0, 30.0]
    out = apply_telephony_simulation(audio, sr=sr, snr_db=None)

    assert len(out) == len(audio)
    assert np.isfinite(out).all()
