"""Trains, calibrates, and compares unsupervised acoustic anomaly models on telephony speech.

Compares:
- Isolation Forest
- One-Class SVM (RBF kernel)

Process:
1. Train exclusively on telephony benign audio (train split).
2. Evaluate and calibrate thresholds on telephony validation split (benign, hard negatives, scam).
3. Report ROC-AUC, F1, Precision, Recall, and Hard-Negative FPR for both on validation.
4. Persist the better model and update config/default.yaml.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_curve

from vishing_detector.anomaly.detector import AcousticAnomalyDetector
from vishing_detector.audio.loader import AudioLoader
from vishing_detector.features.acoustic import AcousticFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_features_from_split(
    manifest_path: Path,
    dataset_dir: Path,
    target_split: str,
    loader: AudioLoader,
    extractor: AcousticFeatureExtractor,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, List[np.ndarray]]]:
    """Extract chunk features from all clips in a specific split.

    Returns:
        all_chunks: Array of feature vectors across all chunks in split.
        clip_records: List of per-clip metadata dicts.
        clip_chunk_map: Map of clip file path to list of chunk feature vectors.
    """
    import joblib
    cache_path = dataset_dir / f".features_cache_{target_split}.joblib"
    if cache_path.exists() and cache_path.stat().st_mtime > manifest_path.stat().st_mtime:
        logger.info("Loading cached features for split '%s' from %s", target_split, cache_path.name)
        cached = joblib.load(cache_path)
        return cached["X_mat"], cached["clip_records"], cached["clip_chunk_map"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    samples = [s for s in manifest_data["samples"] if s["split"] == target_split]
    logger.info("Extracting features for split '%s' (%d clips)...", target_split, len(samples))

    all_chunk_vectors: List[np.ndarray] = []
    clip_records: List[Dict[str, Any]] = []
    clip_chunk_map: Dict[str, List[np.ndarray]] = {}

    for s in samples:
        file_path = dataset_dir / s["file"]
        if not file_path.exists():
            continue

        try:
            seg, _ = loader.load(file_path)
            chunks = loader.chunk_all(seg)
            chunk_vecs: List[np.ndarray] = []
            for c in chunks:
                feats = extractor.extract_chunk(c)
                chunk_vecs.append(feats.vector)
                all_chunk_vectors.append(feats.vector)

            clip_chunk_map[s["file"]] = chunk_vecs
            clip_records.append({
                "file": s["file"],
                "label": s["label"],
                "is_hard_negative": s.get("is_hard_negative", False),
                "voice_id": s.get("voice_id", "unknown"),
                "duration": s.get("duration", 0.0),
                "num_chunks": len(chunk_vecs),
            })
        except Exception as err:
            logger.warning("Error processing %s: %s", file_path.name, err)

    X_mat = np.array(all_chunk_vectors) if all_chunk_vectors else np.zeros((0, len(extractor.feature_names)))
    logger.info("Split '%s': extracted %d total chunks (%d features) across %d clips.",
                target_split, len(X_mat), X_mat.shape[1] if len(X_mat) else 0, len(clip_records))

    try:
        joblib.dump({"X_mat": X_mat, "clip_records": clip_records, "clip_chunk_map": clip_chunk_map}, cache_path)
    except Exception:
        pass

    return X_mat, clip_records, clip_chunk_map


def evaluate_model_on_val(
    detector: AcousticAnomalyDetector,
    val_records: List[Dict[str, Any]],
    val_chunk_map: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    """Evaluate an acoustic anomaly model on the validation split and calibrate threshold."""
    # Compute per-call scores (80th percentile of chunk anomaly scores, consistent with evaluate.py)
    y_true: List[int] = []
    y_scores: List[float] = []
    hard_neg_flags: List[bool] = []

    for r in val_records:
        chunks = val_chunk_map.get(r["file"], [])
        if not chunks:
            continue
        chunk_mat = np.array(chunks)
        scores = detector.score_anomaly(chunk_mat)
        # Call-level score
        call_score = float(np.percentile(scores, 80)) if len(scores) > 1 else float(scores[0])
        y_true.append(r["label"])
        y_scores.append(call_score)
        hard_neg_flags.append(r["is_hard_negative"])

    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)
    hard_neg_arr = np.array(hard_neg_flags)

    # ROC AUC
    fpr_arr, tpr_arr, roc_th = roc_curve(y_true_arr, y_scores_arr)
    roc_auc = float(auc(fpr_arr, tpr_arr))

    # Calibrate threshold on validation: target 10% FPR on val benign (benign + hard negatives)
    benign_scores = y_scores_arr[y_true_arr == 0]
    target_th = float(np.percentile(benign_scores, 90.0))

    # Evaluate at target 10% FPR threshold
    y_pred = (y_scores_arr >= target_th).astype(int)
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    benign_mask = y_true_arr == 0
    fpr_benign = float(np.mean(y_pred[benign_mask] == 1)) if np.any(benign_mask) else 0.0

    hard_neg_mask = benign_mask & hard_neg_arr
    fpr_hard = float(np.mean(y_pred[hard_neg_mask] == 1)) if np.any(hard_neg_mask) else 0.0

    detector.calibrated_threshold = target_th

    return {
        "model_type": detector.model_type,
        "calibrated_threshold": round(target_th, 4),
        "val_roc_auc": round(roc_auc, 4),
        "val_f1": round(f1, 4),
        "val_precision": round(prec, 4),
        "val_recall": round(rec, 4),
        "val_fpr_hard_negatives": round(fpr_hard, 4),
        "val_fpr_plain_benign": round(fpr_benign, 4),
        "num_val_clips": len(y_true),
    }


def train_and_compare_acoustic_models(
    dataset_dir: Path = Path("data/synthetic"),
    output_model_path: Path = Path("models/acoustic_anomaly_model.joblib"),
    config_path: Path = Path("config/default.yaml"),
) -> Dict[str, Any]:
    """Train IsolationForest and OneClassSVM on telephony train data, evaluate on val, and save winner."""
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    loader = AudioLoader()
    extractor = AcousticFeatureExtractor()

    # 1. Extract Train features (benign only)
    X_train, train_records, _ = extract_features_from_split(
        manifest_path, dataset_dir, "train", loader, extractor
    )

    # 2. Extract Val features (benign + hard neg + scam)
    _, val_records, val_chunk_map = extract_features_from_split(
        manifest_path, dataset_dir, "val", loader, extractor
    )

    logger.info("Training candidate A: IsolationForest...")
    iforest = AcousticAnomalyDetector(
        model_type="isolation_forest",
        contamination=0.05,
        random_state=42,
        feature_names=extractor.feature_names,
    )
    iforest.fit(X_train, calibrate_threshold=False)
    metrics_iforest = evaluate_model_on_val(iforest, val_records, val_chunk_map)

    logger.info("Training candidate B: OneClassSVM...")
    ocsvm = AcousticAnomalyDetector(
        model_type="one_class_svm",
        nu=0.05,
        gamma="scale",
        feature_names=extractor.feature_names,
    )
    ocsvm.fit(X_train, calibrate_threshold=False)
    metrics_ocsvm = evaluate_model_on_val(ocsvm, val_records, val_chunk_map)

    logger.info("Validation Comparison Results:")
    logger.info("  IsolationForest: ROC-AUC = %.4f, F1 = %.4f, Prec = %.4f, Rec = %.4f, Hard-Neg FPR = %.1f%%, Thresh = %.4f",
                metrics_iforest["val_roc_auc"], metrics_iforest["val_f1"], metrics_iforest["val_precision"],
                metrics_iforest["val_recall"], metrics_iforest["val_fpr_hard_negatives"] * 100, metrics_iforest["calibrated_threshold"])
    logger.info("  OneClassSVM:     ROC-AUC = %.4f, F1 = %.4f, Prec = %.4f, Rec = %.4f, Hard-Neg FPR = %.1f%%, Thresh = %.4f",
                metrics_ocsvm["val_roc_auc"], metrics_ocsvm["val_f1"], metrics_ocsvm["val_precision"],
                metrics_ocsvm["val_recall"], metrics_ocsvm["val_fpr_hard_negatives"] * 100, metrics_ocsvm["calibrated_threshold"])

    # Determine winner based on validation F1 and ROC-AUC
    if metrics_iforest["val_f1"] >= metrics_ocsvm["val_f1"]:
        winner_name = "isolation_forest"
        winner_detector = iforest
        winner_metrics = metrics_iforest
    else:
        winner_name = "one_class_svm"
        winner_detector = ocsvm
        winner_metrics = metrics_ocsvm

    logger.info("Selected winning model: %s (Val F1: %.4f, ROC-AUC: %.4f)",
                winner_name, winner_metrics["val_f1"], winner_metrics["val_roc_auc"])

    # Save winning model
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    winner_detector.save(output_model_path)

    # Also save comparison report
    comparison_report_path = Path("results/acoustic_model_comparison_val.json")
    comparison_report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "isolation_forest": metrics_iforest,
        "one_class_svm": metrics_ocsvm,
        "selected_model": winner_name,
        "calibrated_threshold": winner_metrics["calibrated_threshold"],
    }
    with open(comparison_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Update config/default.yaml with the selected model and calibrated threshold
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("anomaly", {})
        cfg["anomaly"]["model_type"] = winner_name
        cfg["anomaly"]["calibrated_threshold"] = float(winner_metrics["calibrated_threshold"])
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        logger.info("Updated %s with calibrated threshold %.4f and model_type '%s'",
                    config_path, winner_metrics["calibrated_threshold"], winner_name)

    return report_data


if __name__ == "__main__":
    train_and_compare_acoustic_models()
