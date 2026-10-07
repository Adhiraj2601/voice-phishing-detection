"""Downloads lightweight offline Vosk speech recognition model."""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
TARGET_DIR = Path("models")


def download_and_extract_vosk_model(target_dir: Path = TARGET_DIR) -> Path:
    """Download and extract the lightweight Vosk English model (~40MB)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    model_dir = target_dir / "vosk-model-small-en-us-0.15"

    if model_dir.exists():
        logger.info("Vosk model already exists at: %s", model_dir)
        return model_dir

    zip_path = target_dir / "vosk-model-small-en-us-0.15.zip"
    logger.info("Downloading Vosk model from %s...", VOSK_MODEL_URL)
    urllib.request.urlretrieve(VOSK_MODEL_URL, str(zip_path))

    logger.info("Extracting %s...", zip_path)
    with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
        zip_ref.extractall(str(target_dir))

    if zip_path.exists():
        zip_path.unlink()

    logger.info("Vosk model ready at: %s", model_dir)
    return model_dir


if __name__ == "__main__":
    download_and_extract_vosk_model()
