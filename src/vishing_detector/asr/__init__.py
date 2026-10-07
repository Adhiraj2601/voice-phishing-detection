"""Speech recognition module."""

from vishing_detector.asr.transcriber import (
    MockTranscriber,
    Transcriber,
    TranscriptionResult,
    TranscriptionSegment,
    VoskTranscriber,
    get_transcriber,
)

__all__ = [
    "Transcriber",
    "TranscriptionSegment",
    "TranscriptionResult",
    "MockTranscriber",
    "VoskTranscriber",
    "get_transcriber",
]
