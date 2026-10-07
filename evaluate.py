"""System evaluation engine: computes Precision, Recall, F1, ROC-AUC, Latency, and saves figures."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from vishing_detector.pipeline import VishingDetectionPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_full_evaluation(
    test_dir: Path = Path("data/synthetic"),
    output_dir: Path = Path("docs/figures"),
    config_path: Path = Path("config/default.yaml"),
) -> Dict[str, Any]:
    """Execute evaluation benchmark across benign and vishing call datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = test_dir / "manifest.json"
    if not manifest_file.exists():
        logger.info("Manifest not found. Generating synthetic test dataset...")
        from scripts.generate_synthetic_data import generate_dataset

        generate_dataset(output_dir=test_dir, num_benign=10, num_scam=10)

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])

    logger.info("Evaluating on %d test samples...", len(samples))
    pipeline = VishingDetectionPipeline(config_path=config_path, asr_backend="mock")

    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[float] = []
    latencies: List[float] = []
    acoustic_scores_benign: List[float] = []
    acoustic_scores_scam: List[float] = []

    sample_benign_assessment = None
    sample_scam_assessment = None

    for item in samples:
        rel_path = item["file"]
        label = item["label"]
        script_text = item.get("text", "")
        wav_path = test_dir / rel_path

        if not wav_path.exists():
            continue

        # In mock ASR mode, set transcript to the ground truth script for reproducibility
        pipeline.transcriber.predefined_script = script_text

        result = pipeline.process_file(wav_path)
        expl = result.explanation
        if expl is None:
            continue

        score = expl.overall_risk_score
        pred_label = 1 if expl.is_threat else 0

        y_true.append(label)
        y_pred.append(pred_label)
        y_scores.append(score / 100.0)

        # Collect acoustic scores
        for a in result.chunk_assessments:
            if label == 0:
                acoustic_scores_benign.append(a.acoustic_score)
            else:
                acoustic_scores_scam.append(a.acoustic_score)

        if label == 1 and expl.first_alert_time is not None:
            latencies.append(expl.first_alert_time)

        if label == 0 and sample_benign_assessment is None:
            sample_benign_assessment = result.chunk_assessments
        if label == 1 and sample_scam_assessment is None:
            sample_scam_assessment = result.chunk_assessments

    # Compute classification metrics
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))
    cm = confusion_matrix(y_true, y_pred)
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "roc_auc": round(roc_auc, 4),
        "mean_latency_sec": round(mean_latency, 2),
        "samples_evaluated": len(y_true),
    }

    # Generate figures
    _plot_roc_curve(fpr, tpr, roc_auc, output_dir / "roc_curve.png")
    _plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    _plot_anomaly_distribution(
        acoustic_scores_benign, acoustic_scores_scam, output_dir / "anomaly_distribution.png"
    )
    if sample_benign_assessment and sample_scam_assessment:
        _plot_score_over_time(
            sample_benign_assessment, sample_scam_assessment, output_dir / "score_over_time.png"
        )

    logger.info("Evaluation Complete! Results: %s", metrics)
    return metrics


def _plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, save_path: Path) -> None:
    plt.figure(figsize=(6, 5), facecolor="#0e1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")

    plt.plot(fpr, tpr, color="#38bdf8", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#6b7280", lw=1.5, linestyle="--", label="Random Chance")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate", color="white", fontsize=11)
    plt.ylabel("True Positive Rate", color="white", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC)", color="white", fontsize=12, pad=12)
    plt.tick_params(colors="white")
    plt.grid(True, color="#30363d", alpha=0.5)
    plt.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, facecolor="#0e1117")
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(5, 4.5), facecolor="#0e1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Benign (0)", "Vishing (1)"],
        yticklabels=["Benign (0)", "Vishing (1)"],
        ax=ax,
        annot_kws={"size": 14, "weight": "bold", "color": "white"},
    )
    plt.ylabel("True Label", color="white", fontsize=11)
    plt.xlabel("Predicted Label", color="white", fontsize=11)
    plt.title("Evaluation Confusion Matrix", color="white", fontsize=12, pad=12)
    plt.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, facecolor="#0e1117")
    plt.close()


def _plot_anomaly_distribution(
    benign_scores: List[float], scam_scores: List[float], save_path: Path
) -> None:
    plt.figure(figsize=(7, 4), facecolor="#0e1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")

    if benign_scores:
        sns.kdeplot(
            benign_scores,
            fill=True,
            color="#10b981",
            alpha=0.4,
            label="Benign Speech",
            ax=ax,
            lw=2,
        )
    if scam_scores:
        sns.kdeplot(
            scam_scores,
            fill=True,
            color="#ef4444",
            alpha=0.4,
            label="Vishing Audio",
            ax=ax,
            lw=2,
        )

    plt.xlabel("Normalized Acoustic Anomaly Score", color="white", fontsize=11)
    plt.ylabel("Density", color="white", fontsize=11)
    plt.title("Acoustic Anomaly Score Distribution", color="white", fontsize=12, pad=12)
    plt.tick_params(colors="white")
    plt.grid(True, color="#30363d", alpha=0.5)
    plt.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, facecolor="#0e1117")
    plt.close()


def _plot_score_over_time(
    benign_assessments: list, scam_assessments: list, save_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    t_scam = [a.start_time for a in scam_assessments]
    s_scam = [a.smoothed_risk for a in scam_assessments]

    t_benign = [a.start_time for a in benign_assessments]
    s_benign = [a.smoothed_risk for a in benign_assessments]

    ax.plot(
        t_scam,
        s_scam,
        color="#ef4444",
        lw=2.5,
        marker="o",
        label="Vishing Call (Threat Progression)",
    )
    ax.plot(
        t_benign,
        s_benign,
        color="#10b981",
        lw=2.5,
        marker="s",
        label="Benign Conversation (Baseline)",
    )

    ax.axhline(50, color="#f59e0b", linestyle="--", alpha=0.7, label="Alert Threshold (Elevated)")
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Time Elapsed (seconds)", color="white", fontsize=11)
    ax.set_ylabel("Running Risk Score (0-100)", color="white", fontsize=11)
    ax.set_title("Streaming Detection Latency & Threat Evolution", color="white", fontsize=12, pad=12)
    ax.tick_params(colors="white")
    ax.grid(True, color="#30363d", alpha=0.5)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, facecolor="#0e1117")
    plt.close()


if __name__ == "__main__":
    metrics = run_full_evaluation()
    print("\nBenchmark Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
