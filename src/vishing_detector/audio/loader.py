"""Audio ingestion, preprocessing, normalization, and windowed chunking module."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

# Auto-detect ffmpeg in common Windows WinGet / user locations before pydub initializes
def _configure_ffmpeg_path() -> None:
    if shutil.which("ffmpeg") is not None:
        return
    winget_packages = Path(os.path.expanduser("~")) / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_packages.exists():
        for ffmpeg_candidate in winget_packages.glob("**/ffmpeg.exe"):
            ffmpeg_dir = str(ffmpeg_candidate.parent)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            return

_configure_ffmpeg_path()

import numpy as np
from pydub import AudioSegment, silence

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents a discrete temporal slice of audio in a streaming pipeline.

    Attributes:
        index: Sequential chunk index (0-based).
        start_time: Start timestamp of the chunk in seconds.
        end_time: End timestamp of the chunk in seconds.
        duration: Duration of this chunk in seconds.
        sample_rate: Sample rate in Hz.
        samples: Normalized mono float32 numpy array in range [-1.0, 1.0].
        raw_segment: Underlying PyDub AudioSegment for ASR/export.
    """

    index: int
    start_time: float
    end_time: float
    duration: float
    sample_rate: int
    samples: np.ndarray
    raw_segment: AudioSegment


class AudioLoader:
    """Handles audio loading, format conversion, loudness normalization, and stream chunking."""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        target_loudness_db: float = -20.0,
        silence_thresh_db: float = -40.0,
        min_silence_len_ms: int = 400,
        window_duration_sec: float = 3.0,
        hop_duration_sec: float = 1.0,
    ) -> None:
        """Initialize the audio ingestion loader.

        Args:
            target_sample_rate: Output sample rate (Hz). Defaults to 16000 (standard for ASR).
            target_channels: Number of audio channels (1 for mono). Defaults to 1.
            target_loudness_db: Target dBFS loudness for normalization. Defaults to -20.0.
            silence_thresh_db: dBFS threshold below which audio is considered silence.
            min_silence_len_ms: Minimum silence duration to trim.
            window_duration_sec: Duration of each analysis window chunk in seconds.
            hop_duration_sec: Hop / stride duration between consecutive chunks in seconds.
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.target_loudness_db = target_loudness_db
        self.silence_thresh_db = silence_thresh_db
        self.min_silence_len_ms = min_silence_len_ms
        self.window_duration_sec = window_duration_sec
        self.hop_duration_sec = hop_duration_sec

    def load(
        self,
        file_path: Union[str, Path],
        trim_silence: bool = True,
        normalize: bool = True,
    ) -> Tuple[AudioSegment, np.ndarray]:
        """Load an audio file, convert to target format, normalize loudness, and return samples.

        Args:
            file_path: Path to the input audio file (WAV, MP3, M4A, OGG).
            trim_silence: Whether to trim leading/trailing dead silence.
            normalize: Whether to normalize loudness to target dBFS.

        Returns:
            Tuple of (preprocessed PyDub AudioSegment, float32 numpy array [-1.0, 1.0]).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{ext}'. Supported formats: {self.SUPPORTED_FORMATS}"
            )

        logger.info("Loading audio file: %s (format: %s)", path.name, ext)
        try:
            # Load with pydub (format inferred from extension)
            format_name = ext.lstrip(".")
            segment = AudioSegment.from_file(str(path), format=format_name)
        except Exception as err:
            logger.warning("PyDub from_file failed with %s; falling back to WAV loader", err)
            segment = AudioSegment.from_file(str(path))

        # Standardize channels and sample rate
        if segment.channels != self.target_channels:
            segment = segment.set_channels(self.target_channels)
        if segment.frame_rate != self.target_sample_rate:
            segment = segment.set_frame_rate(self.target_sample_rate)

        # Normalize loudness
        if normalize and segment.max_possible_amplitude > 0:
            segment = self.normalize_loudness(segment, self.target_loudness_db)

        # Trim leading and trailing silence
        if trim_silence and len(segment) > self.min_silence_len_ms:
            segment = self.strip_silence(segment)

        # Convert segment to float32 numpy array
        samples = self.segment_to_numpy(segment)
        logger.info(
            "Preprocessed audio: duration=%.2fs, sr=%d, channels=%d, samples=%d",
            segment.duration_seconds,
            segment.frame_rate,
            segment.channels,
            len(samples),
        )
        return segment, samples

    @staticmethod
    def normalize_loudness(segment: AudioSegment, target_db: float = -20.0) -> AudioSegment:
        """Normalize audio volume to a target dBFS level."""
        change_in_db = target_db - segment.dBFS
        # Prevent runaway amplification on near-silent clips
        if np.isinf(change_in_db) or np.isnan(change_in_db):
            return segment
        bounded_change = float(np.clip(change_in_db, -30.0, 30.0))
        return segment.apply_gain(bounded_change)

    def strip_silence(self, segment: AudioSegment) -> AudioSegment:
        """Trim silence from the beginning and end of the audio clip."""
        nonsilent_ranges = silence.detect_nonsilent(
            segment,
            min_silence_len=self.min_silence_len_ms,
            silence_thresh=self.silence_thresh_db,
        )
        if not nonsilent_ranges:
            return segment

        start_ms = nonsilent_ranges[0][0]
        end_ms = nonsilent_ranges[-1][1]
        trimmed = segment[start_ms:end_ms]
        # Return original if trimming reduced audio to less than 0.5s
        return trimmed if len(trimmed) >= 500 else segment

    @staticmethod
    def segment_to_numpy(segment: AudioSegment) -> np.ndarray:
        """Convert a mono PyDub AudioSegment to a float32 numpy array in [-1.0, 1.0]."""
        samples = np.array(segment.get_array_of_samples())
        if segment.sample_width == 2:
            return (samples / 32768.0).astype(np.float32)
        elif segment.sample_width == 4:
            return (samples / 2147483648.0).astype(np.float32)
        elif segment.sample_width == 1:
            return ((samples - 128) / 128.0).astype(np.float32)
        else:
            max_val = float(2 ** (segment.sample_width * 8 - 1))
            return (samples / max_val).astype(np.float32)

    def chunk_stream(
        self,
        segment: AudioSegment,
    ) -> Generator[AudioChunk, None, None]:
        """Split audio into overlapping temporal chunks to simulate real-time streaming.

        Args:
            segment: Preprocessed mono AudioSegment at target_sample_rate.

        Yields:
            AudioChunk instances containing slice metadata, timestamps, and numpy samples.
        """
        total_len_ms = len(segment)
        window_ms = int(self.window_duration_sec * 1000)
        hop_ms = int(self.hop_duration_sec * 1000)

        # If audio is shorter than window, yield single chunk padded or original
        if total_len_ms <= window_ms:
            samples = self.segment_to_numpy(segment)
            yield AudioChunk(
                index=0,
                start_time=0.0,
                end_time=float(total_len_ms) / 1000.0,
                duration=float(total_len_ms) / 1000.0,
                sample_rate=self.target_sample_rate,
                samples=samples,
                raw_segment=segment,
            )
            return

        chunk_idx = 0
        start_ms = 0
        while start_ms < total_len_ms:
            end_ms = min(start_ms + window_ms, total_len_ms)
            chunk_segment = segment[start_ms:end_ms]
            chunk_samples = self.segment_to_numpy(chunk_segment)

            yield AudioChunk(
                index=chunk_idx,
                start_time=float(start_ms) / 1000.0,
                end_time=float(end_ms) / 1000.0,
                duration=float(len(chunk_segment)) / 1000.0,
                sample_rate=self.target_sample_rate,
                samples=chunk_samples,
                raw_segment=chunk_segment,
            )

            chunk_idx += 1
            if end_ms >= total_len_ms:
                break
            start_ms += hop_ms

    def chunk_all(self, segment: AudioSegment) -> List[AudioChunk]:
        """Eagerly materialize all chunks from an audio segment into a list."""
        return list(self.chunk_stream(segment))
