"""End-to-end streaming detection pipeline coordinating audio, acoustics, ASR, NLP, and fusion."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Union

import yaml
from pydub import AudioSegment

from vishing_detector.anomaly.detector import AcousticAnomalyDetector
from vishing_detector.asr.transcriber import Transcriber, get_transcriber
from vishing_detector.audio.loader import AudioLoader
from vishing_detector.features.acoustic import AcousticFeatureExtractor
from vishing_detector.fusion.scorer import (
    CallExplanation,
    ChunkRiskAssessment,
    RiskScorer,
)
from vishing_detector.nlp.scam_cues import ScamCueAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class StreamingEvent:
    """Event emitted at each temporal step during simulated real-time streaming.

    Attributes:
        chunk_index: Current window index.
        start_time: Window start timestamp in seconds.
        end_time: Window end timestamp in seconds.
        transcript_snippet: Text transcribed in this window.
        cumulative_transcript: Accumulated transcript up to this point.
        assessment: Risk evaluation for this window.
        elapsed_processing_ms: Compute time taken to process this chunk.
    """

    chunk_index: int
    start_time: float
    end_time: float
    transcript_snippet: str
    cumulative_transcript: str
    assessment: ChunkRiskAssessment
    elapsed_processing_ms: float


@dataclass
class PipelineResult:
    """Complete forensic results for an analyzed call.

    Attributes:
        audio_file: Path to analyzed audio file (if applicable).
        audio_duration: Total duration of audio in seconds.
        total_chunks: Number of analysis chunks processed.
        cumulative_transcript: Complete concatenated transcript.
        chunk_assessments: Timeline of risk assessments per chunk.
        explanation: Final forensic explanation and risk tier.
        total_processing_time_sec: Wall-clock runtime for processing.
    """

    audio_file: Optional[str]
    audio_duration: float
    total_chunks: int
    cumulative_transcript: str
    chunk_assessments: List[ChunkRiskAssessment] = field(default_factory=list)
    explanation: Optional[CallExplanation] = None
    total_processing_time_sec: float = 0.0


class VishingDetectionPipeline:
    """Coordinates end-to-end voice phishing detection across multi-modal components."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        anomaly_model_path: Optional[Union[str, Path]] = None,
        asr_backend: str = "vosk",
        vosk_model_path: Optional[Union[str, Path]] = None,
        lexicon_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the pipeline from configuration or explicit model paths.

        Args:
            config_path: Path to YAML config file (e.g. config/default.yaml).
            anomaly_model_path: Path to trained joblib anomaly model.
            asr_backend: ASR engine backend ('vosk' or 'mock').
            vosk_model_path: Path to Vosk model directory.
            lexicon_path: Path to scam lexicon YAML file.
        """
        self.config = self._load_config(config_path)

        # 1. Audio loader
        audio_cfg = self.config.get("audio", {})
        self.loader = AudioLoader(
            target_sample_rate=audio_cfg.get("target_sample_rate", 16000),
            target_channels=audio_cfg.get("target_channels", 1),
            target_loudness_db=audio_cfg.get("normalize_db", -20.0),
            silence_thresh_db=audio_cfg.get("silence_thresh_db", -40.0),
            min_silence_len_ms=audio_cfg.get("min_silence_len_ms", 400),
            window_duration_sec=audio_cfg.get("window_duration_sec", 3.0),
            hop_duration_sec=audio_cfg.get("hop_duration_sec", 1.0),
        )

        # 2. Acoustic feature extractor
        feat_cfg = self.config.get("features", {})
        self.feature_extractor = AcousticFeatureExtractor(
            n_mfcc=feat_cfg.get("n_mfcc", 13),
            n_fft=feat_cfg.get("n_fft", 2048),
            hop_length=feat_cfg.get("hop_length", 512),
            pitch_method=feat_cfg.get("pitch_method", "yin"),
        )

        # 3. ASR transcriber
        asr_cfg = self.config.get("asr", {})
        backend = asr_backend or asr_cfg.get("backend", "vosk")
        model_p = vosk_model_path or asr_cfg.get("model_path")
        self.transcriber: Transcriber = get_transcriber(
            backend=backend,
            model_path=model_p,
        )

        # 4. Scam NLP analyzer
        nlp_cfg = self.config.get("nlp", {})
        lex_p = lexicon_path or nlp_cfg.get("lexicon_path", "config/scam_lexicon.yaml")
        self.nlp_analyzer = ScamCueAnalyzer(lexicon_path=lex_p)

        # 5. Acoustic anomaly detector
        anom_cfg = self.config.get("anomaly", {})
        model_save_path = anomaly_model_path or anom_cfg.get("model_save_path", "models/acoustic_anomaly_model.joblib")
        self.anomaly_detector = self._init_anomaly_detector(model_save_path)

        # 6. Multi-modal Risk Scorer
        fusion_cfg = self.config.get("fusion", {})
        weights = fusion_cfg.get("weights", {})
        thresholds = fusion_cfg.get("alert_thresholds", {})
        fusion_mode = fusion_cfg.get("fusion_mode", "stacker")
        stacker_p = Path(fusion_cfg.get("model_path", "models/stacker_model.joblib"))
        stacker_obj = None
        if stacker_p.exists():
            try:
                from vishing_detector.fusion.stacker import LogisticRiskStacker
                stacker_obj = LogisticRiskStacker.load(stacker_p)
            except Exception as err:
                logger.warning("Failed loading stacker model (%s): %s", stacker_p, err)

        elevated_th = float(fusion_cfg.get("calibrated_threshold", thresholds.get("elevated", 50.0)))
        self.scorer = RiskScorer(
            weight_acoustic=weights.get("acoustic_anomaly", 0.35),
            weight_text=weights.get("text_cues", 0.65),
            smoothing_alpha=fusion_cfg.get("smoothing_alpha", 0.4),
            threshold_low=thresholds.get("low", 25.0),
            threshold_elevated=elevated_th,
            threshold_high=thresholds.get("high", 70.0),
            threshold_critical=thresholds.get("critical", 85.0),
            fusion_mode=fusion_mode,
            stacker=stacker_obj,
        )

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> dict:
        """Load configuration dictionary from YAML file."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as err:
                logger.warning("Error reading config %s: %s; using defaults", config_path, err)
        return {}

    def _init_anomaly_detector(
        self, model_save_path: Union[str, Path]
    ) -> AcousticAnomalyDetector:
        """Load saved anomaly detector if exists, or instantiate unfitted with feature names."""
        p = Path(model_save_path)
        if p.exists():
            try:
                return AcousticAnomalyDetector.load(p)
            except Exception as err:
                logger.warning("Failed loading saved anomaly model (%s): %s", p, err)

        # Create unfitted instance ready to train or baseline
        return AcousticAnomalyDetector(
            model_type="isolation_forest",
            feature_names=self.feature_extractor.feature_names,
        )

    def stream_file(
        self,
        file_path: Optional[Union[str, Path]] = None,
        preloaded_segment: Optional[AudioSegment] = None,
    ) -> Generator[StreamingEvent, None, None]:
        """Process an audio file in simulated real-time, yielding StreamingEvents per window.

        Args:
            file_path: Optional path to the target audio file.
            preloaded_segment: Optional preloaded AudioSegment.

        Yields:
            StreamingEvent for each time window.
        """
        if preloaded_segment is not None:
            segment = preloaded_segment
        elif file_path is not None:
            segment, _ = self.loader.load(file_path)
        else:
            raise ValueError("Either file_path or preloaded_segment must be provided.")

        self.scorer.reset()

        cumulative_texts: List[str] = []

        for chunk in self.loader.chunk_stream(segment):
            t0 = time.perf_counter()

            # Step 1: Acoustic features
            acoustic_feats = self.feature_extractor.extract_chunk(chunk)

            # Step 2: Anomaly prediction
            if self.anomaly_detector.is_fitted:
                anomaly_res = self.anomaly_detector.predict_chunk(acoustic_feats.vector)
            else:
                # Default baseline inference if model is not yet fitted
                from vishing_detector.anomaly.detector import AnomalyInferenceResult

                anomaly_res = AnomalyInferenceResult(
                    is_anomaly=False,
                    anomaly_score=0.2,
                    raw_score=0.1,
                    top_deviating_features=[],
                )

            # Step 3: Speech-to-text
            asr_res = self.transcriber.transcribe_chunk(chunk)
            chunk_text = asr_res.full_text
            if chunk_text:
                cumulative_texts.append(chunk_text)

            current_transcript = " ".join(cumulative_texts)

            # Step 4: NLP scam analysis on cumulative conversation
            nlp_res = self.nlp_analyzer.analyze_text(current_transcript)

            # Step 5: Multi-modal fusion
            assessment = self.scorer.compute_chunk_risk(
                chunk_index=chunk.index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                anomaly_result=anomaly_res,
                nlp_result=nlp_res,
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            yield StreamingEvent(
                chunk_index=chunk.index,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                transcript_snippet=chunk_text,
                cumulative_transcript=current_transcript,
                assessment=assessment,
                elapsed_processing_ms=elapsed_ms,
            )

    def process_file(self, file_path: Union[str, Path]) -> PipelineResult:
        """Run complete forensic analysis on an audio file synchronously.

        Args:
            file_path: Path to the target audio file.

        Returns:
            PipelineResult with all chunk assessments and final explanation.
        """
        t_start = time.perf_counter()
        assessments: List[ChunkRiskAssessment] = []
        final_transcript = ""

        segment, _ = self.loader.load(file_path)
        duration_sec = len(segment) / 1000.0

        for event in self.stream_file(preloaded_segment=segment):
            assessments.append(event.assessment)
            final_transcript = event.cumulative_transcript

        explanation = self.scorer.generate_explanation(
            assessments=assessments,
            full_transcript=final_transcript,
        )

        total_runtime = time.perf_counter() - t_start

        return PipelineResult(
            audio_file=str(file_path),
            audio_duration=duration_sec,
            total_chunks=len(assessments),
            cumulative_transcript=final_transcript,
            chunk_assessments=assessments,
            explanation=explanation,
            total_processing_time_sec=total_runtime,
        )
