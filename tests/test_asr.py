"""Unit tests for speech-to-text abstraction and mock transcriber."""

from pathlib import Path

from pydub import AudioSegment

from vishing_detector.asr.transcriber import (
    MockTranscriber,
    TranscriptionResult,
    TranscriptionSegment,
    get_transcriber,
)
from vishing_detector.audio.loader import AudioLoader


def test_mock_transcriber_chunk(synthetic_sine_wav: Path):
    loader = AudioLoader()
    seg, _ = loader.load(synthetic_sine_wav)
    chunk = loader.chunk_all(seg)[0]

    transcriber = MockTranscriber(predefined_script="urgent IRS alert")
    res = transcriber.transcribe_chunk(chunk)

    assert isinstance(res, TranscriptionResult)
    assert "IRS" in res.full_text
    assert len(res.segments) == 1
    assert res.segments[0].start_time == chunk.start_time
    assert res.segments[0].end_time == chunk.end_time


def test_mock_transcriber_segment():
    seg = AudioSegment.silent(duration=2000, frame_rate=16000)
    transcriber = MockTranscriber()
    res = transcriber.transcribe_audio_segment(seg)

    assert isinstance(res, TranscriptionResult)
    assert res.duration == 2.0


def test_transcriber_factory():
    t_mock = get_transcriber("mock")
    assert isinstance(t_mock, MockTranscriber)

    t_vosk = get_transcriber("vosk", model_path="models/non_existent")
    assert t_vosk is not None
