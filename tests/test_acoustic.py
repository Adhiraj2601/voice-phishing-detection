"""Unit tests for acoustic feature extraction using Librosa."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vishing_detector.audio.loader import AudioLoader
from vishing_detector.features.acoustic import AcousticFeatureExtractor, AcousticFeatures


def test_extractor_feature_names():
    extractor = AcousticFeatureExtractor(n_mfcc=13)
    names = extractor.feature_names
    assert len(names) == extractor.feature_dimension
    assert "mfcc_1_mean" in names
    assert "mfcc_delta_1_mean" in names
    assert "chroma_1_mean" in names
    assert "spectral_centroid_mean" in names
    assert "pitch_f0_mean" in names
    assert "pause_ratio" in names


def test_extractor_extract_chunk(synthetic_sine_wav: Path):
    loader = AudioLoader()
    seg, _ = loader.load(synthetic_sine_wav)
    chunks = loader.chunk_all(seg)
    assert len(chunks) > 0

    extractor = AcousticFeatureExtractor()
    feat = extractor.extract_chunk(chunks[0])

    assert isinstance(feat, AcousticFeatures)
    assert len(feat.vector) == extractor.feature_dimension
    assert not np.isnan(feat.vector).any()
    assert not np.isinf(feat.vector).any()
    assert feat.chunk_index == 0


def test_extractor_extract_silent_chunk(silent_wav: Path):
    loader = AudioLoader()
    seg, _ = loader.load(silent_wav, trim_silence=False)
    chunks = loader.chunk_all(seg)

    extractor = AcousticFeatureExtractor()
    feat = extractor.extract_chunk(chunks[0])

    assert len(feat.vector) == extractor.feature_dimension
    assert not np.isnan(feat.vector).any()


def test_extractor_batch_dataframe(synthetic_sine_wav: Path):
    loader = AudioLoader()
    seg, _ = loader.load(synthetic_sine_wav)
    chunks = loader.chunk_all(seg)

    extractor = AcousticFeatureExtractor()
    df = extractor.extract_chunks(chunks)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(chunks)
    assert "chunk_index" in df.columns
    assert "spectral_rolloff_mean" in df.columns
