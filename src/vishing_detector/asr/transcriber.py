"""Speech-to-text (ASR) abstraction layer with offline-capable implementations."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from pydub import AudioSegment

from vishing_detector.audio.loader import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A timestamped text phrase or word within a transcription.

    Attributes:
        text: Transcribed text snippet.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        confidence: Confidence score in range [0.0, 1.0].
    """

    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """Full transcription result containing aggregated text and segment breakdown.

    Attributes:
        full_text: Complete concatenated transcript text.
        segments: List of timestamped TranscriptionSegment instances.
        duration: Total audio duration in seconds.
    """

    full_text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    duration: float = 0.0


class Transcriber(ABC):
    """Abstract base class for speech recognition backends."""

    @abstractmethod
    def transcribe_chunk(self, chunk: AudioChunk) -> TranscriptionResult:
        """Transcribe an individual streaming audio chunk."""
        raise NotImplementedError

    @abstractmethod
    def transcribe_audio_segment(self, segment: AudioSegment) -> TranscriptionResult:
        """Transcribe a full PyDub AudioSegment."""
        raise NotImplementedError

    def transcribe_file(self, file_path: Union[str, Path]) -> TranscriptionResult:
        """Transcribe an audio file from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        segment = AudioSegment.from_file(str(path))
        return self.transcribe_audio_segment(segment)


class MockTranscriber(Transcriber):
    """Deterministic mock transcriber for testing and fast offline pipelines without ASR weights."""

    def __init__(self, predefined_script: Optional[str] = None) -> None:
        """Initialize mock transcriber.

        Args:
            predefined_script: Optional transcript text to return. If None, synthesizes text.
        """
        self.predefined_script = predefined_script

    def transcribe_chunk(self, chunk: AudioChunk) -> TranscriptionResult:
        text = self.predefined_script or "hello how are you doing today"
        seg = TranscriptionSegment(
            text=text,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            confidence=0.95,
        )
        return TranscriptionResult(full_text=text, segments=[seg], duration=chunk.duration)

    def transcribe_audio_segment(self, segment: AudioSegment) -> TranscriptionResult:
        duration = len(segment) / 1000.0
        text = self.predefined_script or "hello how are you doing today"
        seg = TranscriptionSegment(text=text, start_time=0.0, end_time=duration, confidence=0.95)
        return TranscriptionResult(full_text=text, segments=[seg], duration=duration)


class VoskTranscriber(Transcriber):
    """Offline speech-to-text transcriber using Vosk Kaldi-based models.

    Chosen for:
    - 100% offline and low-resource (CPU-friendly, ~50MB model footprint).
    - Fast streaming inference with zero GPU / heavy PyTorch requirement.
    - Word-level and phrase-level timestamp extraction.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize Vosk transcriber.

        Args:
            model_path: Path to unpacked Vosk model directory. If None or not found,
                        gracefully falls back to mock transcriber with logged warning.
        """
        self.model_path = Path(model_path) if model_path else None
        self._model = None
        self._is_ready = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempt to load Vosk model."""
        try:
            from vosk import Model  # type: ignore

            if self.model_path and self.model_path.exists():
                logger.info("Loading Vosk model from: %s", self.model_path)
                self._model = Model(str(self.model_path))
                self._is_ready = True
            else:
                logger.warning(
                    "Vosk model path not found (%s). Transcriptions will use heuristic fallback.",
                    self.model_path,
                )
        except Exception as err:
            logger.warning("Failed to initialize Vosk: %s", err)
            self._model = None
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """Whether a real Vosk model is loaded and ready."""
        return self._is_ready

    def transcribe_chunk(self, chunk: AudioChunk) -> TranscriptionResult:
        """Transcribe an audio chunk with Vosk or heuristic fallback."""
        if not self._is_ready or self._model is None:
            return MockTranscriber().transcribe_chunk(chunk)

        try:
            from vosk import KaldiRecognizer  # type: ignore

            # Ensure 16 kHz 16-bit mono PCM bytes
            raw_bytes = chunk.raw_segment.set_frame_rate(16000).set_channels(1).raw_data
            rec = KaldiRecognizer(self._model, 16000)
            rec.SetWords(True)

            rec.AcceptWaveform(raw_bytes)
            res = json.loads(rec.FinalResult())
            text = res.get("text", "").strip()

            segments = []
            if text:
                segments.append(
                    TranscriptionSegment(
                        text=text,
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        confidence=0.90,
                    )
                )

            return TranscriptionResult(
                full_text=text, segments=segments, duration=chunk.duration
            )
        except Exception as err:
            logger.error("Vosk chunk transcription error: %s", err)
            return MockTranscriber().transcribe_chunk(chunk)

    def transcribe_audio_segment(self, segment: AudioSegment) -> TranscriptionResult:
        """Transcribe a full audio segment."""
        duration = len(segment) / 1000.0
        if not self._is_ready or self._model is None:
            return MockTranscriber().transcribe_audio_segment(segment)

        try:
            from vosk import KaldiRecognizer  # type: ignore

            seg_16k = segment.set_frame_rate(16000).set_channels(1)
            raw_bytes = seg_16k.raw_data
            rec = KaldiRecognizer(self._model, 16000)
            rec.SetWords(True)

            # Stream through recognizer in chunks
            step = 4000
            for i in range(0, len(raw_bytes), step):
                rec.AcceptWaveform(raw_bytes[i : i + step])

            final = json.loads(rec.FinalResult())
            full_text = final.get("text", "").strip()
            segments: List[TranscriptionSegment] = []

            # Extract word-level timestamps if available
            words = final.get("result", [])
            for w in words:
                word_text = w.get("word", "")
                w_start = float(w.get("start", 0.0))
                w_end = float(w.get("end", 0.0))
                conf = float(w.get("conf", 1.0))
                segments.append(
                    TranscriptionSegment(
                        text=word_text,
                        start_time=w_start,
                        end_time=w_end,
                        confidence=conf,
                    )
                )

            if not segments and full_text:
                segments.append(
                    TranscriptionSegment(
                        text=full_text,
                        start_time=0.0,
                        end_time=duration,
                        confidence=0.90,
                    )
                )

            return TranscriptionResult(
                full_text=full_text, segments=segments, duration=duration
            )
        except Exception as err:
            logger.error("Vosk audio segment transcription error: %s", err)
            return MockTranscriber().transcribe_audio_segment(segment)


def get_transcriber(
    backend: str = "vosk",
    model_path: Optional[Union[str, Path]] = None,
    mock_script: Optional[str] = None,
) -> Transcriber:
    """Factory function to instantiate swappable transcriber backends.

    Args:
        backend: Name of the ASR backend ('vosk', 'mock').
        model_path: Path to offline model directory (for vosk).
        mock_script: Optional predefined text when using mock backend.

    Returns:
        Configured Transcriber instance.
    """
    if backend.lower() == "mock":
        return MockTranscriber(predefined_script=mock_script)
    elif backend.lower() == "vosk":
        return VoskTranscriber(model_path=model_path)
    else:
        logger.warning("Unrecognized ASR backend '%s'; defaulting to MockTranscriber", backend)
        return MockTranscriber(predefined_script=mock_script)
