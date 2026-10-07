"""Acoustic feature extraction using Librosa for voice anomaly detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np
import pandas as pd

from vishing_detector.audio.loader import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class AcousticFeatures:
    """Container for acoustic features extracted from an audio chunk.

    Attributes:
        chunk_index: Index of the source chunk.
        start_time: Start timestamp of the chunk in seconds.
        end_time: End timestamp of the chunk in seconds.
        vector: 1D numpy array of fixed-length numeric features.
        feature_names: List of strings corresponding to each feature dimension.
    """

    chunk_index: int
    start_time: float
    end_time: float
    vector: np.ndarray
    feature_names: List[str]

    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Convert features and chunk metadata into a flat dictionary."""
        data: Dict[str, Union[float, int]] = {
            "chunk_index": self.chunk_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        for name, val in zip(self.feature_names, self.vector, strict=False):
            data[name] = float(val)
        return data


class AcousticFeatureExtractor:
    """Extracts fixed-length acoustic representations from audio chunks.

    Calculates:
    - MFCCs (13 coefficients), along with first-order (delta) and second-order (delta-delta) derivatives.
    - Chroma pitch class statistics (12 bins).
    - Spectral characteristics: Centroid, Bandwidth, Rolloff, and Contrast.
    - Zero-Crossing Rate (ZCR) and Root Mean Square (RMS) energy.
    - Pitch (F0) tracking via YIN algorithm, including voiced fraction.
    - Temporal prosodic metrics: pause ratio and estimated syllable/speaking rate.
    """

    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 50.0,
        fmax: float = 8000.0,
        pitch_method: str = "yin",
    ) -> None:
        """Initialize acoustic feature extractor.

        Args:
            n_mfcc: Number of Mel-Frequency Cepstral Coefficients to compute.
            n_fft: FFT window size.
            hop_length: Number of audio samples between adjacent STFT columns.
            fmin: Minimum frequency bound for spectral and pitch analysis.
            fmax: Maximum frequency bound for spectral and pitch analysis.
            pitch_method: Method for fundamental frequency estimation ('yin' or 'pyin').
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.pitch_method = pitch_method

        self._feature_names = self._generate_feature_names()

    @property
    def feature_names(self) -> List[str]:
        """List of feature names matching the output vector order."""
        return list(self._feature_names)

    @property
    def feature_dimension(self) -> int:
        """Total number of scalar feature dimensions."""
        return len(self._feature_names)

    def _generate_feature_names(self) -> List[str]:
        """Construct deterministic ordered list of feature names."""
        names: List[str] = []

        # MFCCs + deltas
        for i in range(1, self.n_mfcc + 1):
            names.append(f"mfcc_{i}_mean")
            names.append(f"mfcc_{i}_std")
        for i in range(1, self.n_mfcc + 1):
            names.append(f"mfcc_delta_{i}_mean")
            names.append(f"mfcc_delta_{i}_std")
        for i in range(1, self.n_mfcc + 1):
            names.append(f"mfcc_delta2_{i}_mean")
            names.append(f"mfcc_delta2_{i}_std")

        # Chroma (12 pitch classes)
        for i in range(1, 13):
            names.append(f"chroma_{i}_mean")
            names.append(f"chroma_{i}_std")

        # Spectral shapes
        names.extend(["spectral_centroid_mean", "spectral_centroid_std"])
        names.extend(["spectral_bandwidth_mean", "spectral_bandwidth_std"])
        names.extend(["spectral_rolloff_mean", "spectral_rolloff_std"])
        names.extend(["spectral_contrast_mean", "spectral_contrast_std"])

        # Energy & ZCR
        names.extend(["zcr_mean", "zcr_std"])
        names.extend(["rms_mean", "rms_std"])

        # Pitch & Prosody
        names.extend(["pitch_f0_mean", "pitch_f0_std", "voiced_ratio"])
        names.extend(["pause_ratio", "speaking_rate_est"])

        return names

    def extract_chunk(self, chunk: AudioChunk) -> AcousticFeatures:
        """Extract acoustic features from a single AudioChunk.

        Args:
            chunk: AudioChunk instance containing normalized samples and metadata.

        Returns:
            AcousticFeatures with 1D numpy vector and column names.
        """
        y = chunk.samples
        sr = chunk.sample_rate

        # Guard against empty or silent audio
        if len(y) < self.hop_length or np.all(y == 0):
            zero_vec = np.zeros(self.feature_dimension, dtype=np.float32)
            return AcousticFeatures(
                chunk_index=chunk.index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                vector=zero_vec,
                feature_names=self.feature_names,
            )

        features: List[float] = []

        # 1. MFCCs and derivatives
        try:
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=min(self.n_fft, len(y)),
                hop_length=self.hop_length,
            )
            # Delta and delta-delta
            mfcc_delta = librosa.feature.delta(mfcc, order=1)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            for i in range(self.n_mfcc):
                features.append(float(np.mean(mfcc[i])))
                features.append(float(np.std(mfcc[i])))
            for i in range(self.n_mfcc):
                features.append(float(np.mean(mfcc_delta[i])))
                features.append(float(np.std(mfcc_delta[i])))
            for i in range(self.n_mfcc):
                features.append(float(np.mean(mfcc_delta2[i])))
                features.append(float(np.std(mfcc_delta2[i])))
        except Exception as err:
            logger.debug("MFCC extraction failed: %s; falling back to zeros", err)
            features.extend([0.0] * (self.n_mfcc * 6))

        # 2. Chroma
        try:
            chroma = librosa.feature.chroma_stft(
                y=y,
                sr=sr,
                n_fft=min(self.n_fft, len(y)),
                hop_length=self.hop_length,
            )
            for i in range(12):
                features.append(float(np.mean(chroma[i])))
                features.append(float(np.std(chroma[i])))
        except Exception:
            features.extend([0.0] * 24)

        # 3. Spectral features
        try:
            sc = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=min(self.n_fft, len(y)), hop_length=self.hop_length
            )
            features.append(float(np.mean(sc)))
            features.append(float(np.std(sc)))
        except Exception:
            features.extend([0.0, 0.0])

        try:
            sb = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=min(self.n_fft, len(y)), hop_length=self.hop_length
            )
            features.append(float(np.mean(sb)))
            features.append(float(np.std(sb)))
        except Exception:
            features.extend([0.0, 0.0])

        try:
            s_roll = librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=min(self.n_fft, len(y)), hop_length=self.hop_length
            )
            features.append(float(np.mean(s_roll)))
            features.append(float(np.std(s_roll)))
        except Exception:
            features.extend([0.0, 0.0])

        try:
            s_contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, n_fft=min(self.n_fft, len(y)), hop_length=self.hop_length
            )
            features.append(float(np.mean(s_contrast)))
            features.append(float(np.std(s_contrast)))
        except Exception:
            features.extend([0.0, 0.0])

        # 4. ZCR and RMS Energy
        try:
            zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)
            features.append(float(np.mean(zcr)))
            features.append(float(np.std(zcr)))
        except Exception:
            features.extend([0.0, 0.0])

        try:
            rms = librosa.feature.rms(
                y=y, frame_length=min(self.n_fft, len(y)), hop_length=self.hop_length
            )
            features.append(float(np.mean(rms)))
            features.append(float(np.std(rms)))
        except Exception:
            features.extend([0.0, 0.0])

        # 5. Pitch (F0) tracking via YIN
        f0_mean, f0_std, voiced_ratio = self._extract_pitch(y, sr)
        features.extend([f0_mean, f0_std, voiced_ratio])

        # 6. Prosody & Speaking Rate
        pause_ratio, speaking_rate = self._estimate_prosody(y, sr)
        features.extend([pause_ratio, speaking_rate])

        # Sanitize any NaNs or Infs
        vector = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        return AcousticFeatures(
            chunk_index=chunk.index,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            vector=vector,
            feature_names=self.feature_names,
        )

    def _extract_pitch(self, y: np.ndarray, sr: int) -> Tuple[float, float, float]:
        """Estimate fundamental frequency (F0) and voiced fraction using YIN."""
        try:
            f0 = librosa.yin(
                y=y,
                fmin=max(self.fmin, 65.0),
                fmax=min(self.fmax, 500.0),
                sr=sr,
                hop_length=self.hop_length,
            )
            voiced = f0[~np.isnan(f0)]
            voiced_ratio = float(len(voiced) / len(f0)) if len(f0) > 0 else 0.0
            if len(voiced) > 0:
                return float(np.mean(voiced)), float(np.std(voiced)), voiced_ratio
            return 0.0, 0.0, 0.0
        except Exception as err:
            logger.debug("Pitch tracking fallback: %s", err)
            return 0.0, 0.0, 0.0

    def _estimate_prosody(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """Estimate pause ratio and speaking rate (syllable envelope bursts per second)."""
        try:
            # Frame-level RMS energy
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            if len(rms) == 0:
                return 0.0, 0.0

            # Dynamic silence threshold based on median energy
            threshold = max(float(np.median(rms)) * 0.4, 1e-4)
            silence_frames = np.sum(rms < threshold)
            pause_ratio = float(silence_frames / len(rms))

            # Approximate speaking rate via peaks in the smoothed energy envelope
            # Peaks roughly correspond to vowel nuclei / syllable centers
            duration_sec = len(y) / sr
            if duration_sec > 0.2:
                # Count sign changes in derivative of smoothed energy envelope
                smoothed = np.convolve(rms, np.ones(5) / 5, mode="same")
                diff = np.diff(smoothed)
                peaks = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0]
                speaking_rate = float(len(peaks) / duration_sec)
            else:
                speaking_rate = 0.0

            return pause_ratio, speaking_rate
        except Exception:
            return 0.0, 0.0

    def extract_chunks(self, chunks: List[AudioChunk]) -> pd.DataFrame:
        """Extract features across a batch of chunks into a structured pandas DataFrame.

        Args:
            chunks: List of AudioChunk instances.

        Returns:
            DataFrame with columns: chunk_index, start_time, end_time, followed by all acoustic features.
        """
        records = []
        for chunk in chunks:
            feat = self.extract_chunk(chunk)
            records.append(feat.to_dict())

        return pd.DataFrame(records)
