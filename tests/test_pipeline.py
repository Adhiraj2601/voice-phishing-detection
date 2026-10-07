"""Unit tests for end-to-end VishingDetectionPipeline."""

from pathlib import Path

from vishing_detector.pipeline import PipelineResult, StreamingEvent, VishingDetectionPipeline


def test_pipeline_stream_and_process(synthetic_sine_wav: Path):
    pipeline = VishingDetectionPipeline(
        config_path="config/default.yaml",
        asr_backend="mock",
    )

    # 1. Test streaming generator
    events = list(pipeline.stream_file(synthetic_sine_wav))
    assert len(events) >= 1
    for ev in events:
        assert isinstance(ev, StreamingEvent)
        assert ev.assessment.instantaneous_risk >= 0.0
        assert ev.elapsed_processing_ms > 0.0

    # 2. Test full process_file
    result = pipeline.process_file(synthetic_sine_wav)
    assert isinstance(result, PipelineResult)
    assert result.audio_duration > 0.0
    assert result.total_chunks == len(events)
    assert result.explanation is not None
    assert result.total_processing_time_sec > 0.0
