"""Trains, calibrates, and optimizes the LogisticRiskStacker on validation split data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from vishing_detector.anomaly.detector import AcousticAnomalyDetector
from vishing_detector.audio.loader import AudioLoader
from vishing_detector.features.acoustic import AcousticFeatureExtractor
from vishing_detector.fusion.stacker import LogisticRiskStacker
from vishing_detector.nlp.scam_cues import ScamCueAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_and_calibrate_stacker(
    dataset_dir: Path = Path("data/synthetic"),
    output_model_path: Path = Path("models/stacker_model.joblib"),
    config_path: Path = Path("config/default.yaml"),
) -> Dict[str, Any]:
    """Extract features from validation clips, train logistic stacker with CV, and calibrate threshold."""
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    val_samples = [s for s in manifest_data["samples"] if s["split"] == "val"]
    logger.info("Extracting multi-modal signals for %d validation clips...", len(val_samples))

    # Initialize components
    loader = AudioLoader()
    extractor = AcousticFeatureExtractor()
    detector = AcousticAnomalyDetector.load("models/acoustic_anomaly_model.joblib")
    analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")

    feature_matrix: List[np.ndarray] = []
    labels: List[int] = []
    hard_neg_flags: List[bool] = []

    for s in val_samples:
        file_path = dataset_dir / s["file"]
        if not file_path.exists():
            continue

        try:
            seg, _ = loader.load(file_path)
            chunks = loader.chunk_all(seg)
            chunk_vecs = [extractor.extract_chunk(c).vector for c in chunks]
            ac_scores = detector.score_anomaly(np.array(chunk_vecs))

            nlp_res = analyzer.analyze_text(s["text"], intent_aware=True)

            feats = LogisticRiskStacker.extract_features(ac_scores, nlp_res)
            feature_matrix.append(feats)
            labels.append(s["label"])
            hard_neg_flags.append(s.get("is_hard_negative", False))
        except Exception as err:
            logger.warning("Error processing %s: %s", file_path.name, err)

    X_val = np.array(feature_matrix, dtype=np.float32)
    y_val = np.array(labels, dtype=int)
    hard_neg_mask = (y_val == 0) & np.array(hard_neg_flags, dtype=bool)

    logger.info("Extracted %d validation feature vectors (dim: %d)", len(X_val), X_val.shape[1])

    # Best streaming alpha selected on validation (tested across [0.25, 0.35, 0.40, 0.50])
    best_alpha = 0.40

    # Train stacker
    stacker = LogisticRiskStacker(
        random_state=42,
        cv_folds=5,
        streaming_alpha=best_alpha,
    )
    stacker.fit(X_val, y_val, hard_neg_mask=hard_neg_mask)

    # Save trained stacker model
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    stacker.save(output_model_path)

    # Save results to json
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "stacker_validation_results.json"
    report_data = {
        "val_metrics": stacker.val_metrics,
        "calibrated_threshold": stacker.calibrated_threshold,
        "calibrated_risk_threshold_100": round(stacker.calibrated_threshold * 100.0, 2),
        "streaming_alpha": best_alpha,
        "num_val_samples": len(y_val),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Update config/default.yaml
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("fusion", {})
        cfg["fusion"]["model_path"] = str(output_model_path).replace("\\", "/")
        cfg["fusion"]["fusion_mode"] = "stacker"
        cfg["fusion"]["calibrated_threshold"] = float(stacker.calibrated_threshold * 100.0)
        cfg["fusion"]["smoothing_alpha"] = float(best_alpha)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        logger.info("Updated %s with stacker threshold %.2f and smoothing_alpha %.2f",
                    config_path, stacker.calibrated_threshold * 100.0, best_alpha)

    logger.info("Stacker training complete! Metrics on validation: %s", stacker.val_metrics)
    return report_data


if __name__ == "__main__":
    train_and_calibrate_stacker()
