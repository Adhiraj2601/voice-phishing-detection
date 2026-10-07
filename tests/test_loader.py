"""Unit tests for audio loader, normalization, and window chunking."""

from pathlib import Path

import numpy as np
import pytest
from pydub import AudioSegment

from vishing_detector.audio.loader import AudioChunk, AudioLoader


def test_loader_initialization():
    loader = AudioLoader(target_sample_rate=16000, window_duration_sec=3.0, hop_duration_sec=1.0)
    assert loader.target_sample_rate == 16000
    assert loader.window_duration_sec == 3.0
    assert loader.hop_duration_sec == 1.0


def test_loader_load_wav(synthetic_sine_wav: Path):
    loader = AudioLoader(target_sample_rate=16000)
    segment, samples = loader.load(synthetic_sine_wav)

    assert isinstance(segment, AudioSegment)
    assert segment.frame_rate == 16000
    assert segment.channels == 1
    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32
    assert -1.0 <= samples.min() <= samples.max() <= 1.0


def test_loader_chunk_stream(synthetic_sine_wav: Path):
    loader = AudioLoader(window_duration_sec=2.0, hop_duration_sec=1.0)
    segment, _ = loader.load(synthetic_sine_wav)
    chunks = list(loader.chunk_stream(segment))

    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, AudioChunk)
        assert chunk.index == i
        assert chunk.duration <= 2.05
        assert len(chunk.samples) > 0
        assert chunk.sample_rate == 16000


def test_loader_short_audio_chunking(short_wav: Path):
    loader = AudioLoader(window_duration_sec=3.0, hop_duration_sec=1.0)
    segment, _ = loader.load(short_wav)
    chunks = list(loader.chunk_stream(segment))

    # Short audio should yield exactly one chunk without error
    assert len(chunks) == 1
    assert chunks[0].start_time == 0.0
    assert chunks[0].duration <= 1.1


def test_loader_file_not_found():
    loader = AudioLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent_audio_file.wav")


def test_loader_unsupported_format(tmp_path: Path):
    fake_txt = tmp_path / "test.txt"
    fake_txt.write_text("not audio")
    loader = AudioLoader()
    with pytest.raises(ValueError, match="Unsupported format"):
        loader.load(fake_txt)
