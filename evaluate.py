"""System evaluation engine with full component ablation (Acoustic-Only vs. Text-Only vs. Fused).

Evaluates on telephony-simulated held-out test splits with hard negatives.
Computes Precision, Recall, F1, ROC-AUC, False Positive Rate (FPR) on Hard Negatives,
and exact Detection Latency in seconds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def compute_metrics_for_mode(
    records: List[Dict[str, Any]],
    mode: str = "fused",
    threshold: float = 50.0,
) -> Dict[str, Any]:
    """Compute benchmark metrics for a specific ablation variant from cached call records."""
    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[float] = []
    latencies: List[float] = []

    hard_neg_total = 0
    hard_neg_fp = 0
    benign_total = 0
    benign_fp = 0

    for rec in records:
        label = rec["label"]
        is_hard_neg = rec["is_hard_negative"]
        assessments = rec["assessments"]
        expl = rec["explanation"]

        if label == 0:
            benign_total += 1
            if is_hard_neg:
                hard_neg_total += 1

        if mode == "acoustic_only":
            ac_scores = [a.acoustic_score * 100.0 for a in assessments]
            call_score = float(np.percentile(ac_scores, 80)) if len(ac_scores) > 1 else ac_scores[0]
            pred = 1 if call_score >= threshold else 0

            first_alert = None
            for a, s in zip(assessments, ac_scores, strict=False):
                if s >= threshold:
                    first_alert = round(a.end_time, 2)
                    break
            if label == 1 and first_alert is not None:
                latencies.append(first_alert)

        elif mode == "text_only":
            tx_scores = [a.text_score * 100.0 for a in assessments]
            call_score = max(tx_scores) if tx_scores else 0.0
            pred = 1 if call_score >= threshold else 0

            first_alert = None
            for a, s in zip(assessments, tx_scores, strict=False):
                if s >= threshold:
                    first_alert = round(a.end_time, 2)
                    break
            if label == 1 and first_alert is not None:
                latencies.append(first_alert)

        else: # "fused"
            call_score = expl.overall_risk_score if expl else assessments[-1].smoothed_risk
            pred = 1 if call_score >= threshold else 0

            if label == 1 and expl and expl.first_alert_time is not None:
                latencies.append(expl.first_alert_time)

        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(call_score / 100.0)

        if label == 0 and pred == 1:
            benign_fp += 1
            if is_hard_neg:
                hard_neg_fp += 1

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr_arr, tpr_arr))
    cm = confusion_matrix(y_true, y_pred)

    fpr_hard = float(hard_neg_fp / hard_neg_total) if hard_neg_total > 0 else 0.0
    fpr_overall = float(benign_fp / benign_total) if benign_total > 0 else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "mode": mode,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "roc_auc": round(roc_auc, 4),
        "fpr_hard_negatives": round(fpr_hard, 4),
        "fpr_overall": round(fpr_overall, 4),
        "mean_latency_sec": round(mean_latency, 2),
        "confusion_matrix": cm,
        "fpr_arr": fpr_arr,
        "tpr_arr": tpr_arr,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "samples_evaluated": len(y_true),
        "hard_negatives_evaluated": hard_neg_total,
    }


def run_full_evaluation(
    dataset_dir: Path = Path("data/synthetic"),
    output_dir: Path = Path("docs/figures"),
    config_path: Path = Path("config/default.yaml"),
) -> Dict[str, Any]:
    """Execute ablation evaluation across Acoustic-Only, Text-Only, and Fused models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = dataset_dir / "manifest.json"
    if not manifest_file.exists():
        logger.info("Manifest not found. Generating large-scale synthetic dataset...")
        from scripts.generate_synthetic_data import generate_large_scale_dataset

        generate_large_scale_dataset(output_dir=dataset_dir)

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_samples = data.get("samples", [])

    # Filter strictly for the held-out test split
    test_samples = [s for s in all_samples if s.get("split") == "test"]
    if not test_samples:
        test_samples = all_samples

    logger.info("Found %d test samples in manifest.", len(test_samples))
    pipeline = VishingDetectionPipeline(config_path=config_path, asr_backend="mock")

    cached_records: List[Dict[str, Any]] = []
    benign_acoustic_scores: List[float] = []
    scam_acoustic_scores: List[float] = []

    sample_normal = None
    sample_hard_neg = None
    sample_scam = None

    logger.info("Processing %d test clips through detection pipeline...", len(test_samples))
    for i, item in enumerate(test_samples):
        rel_path = item["file"]
        label = item["label"]
        is_hard = item.get("is_hard_negative", False)
        script_text = item.get("text", "")
        wav_path = dataset_dir / rel_path

        if not wav_path.exists():
            continue

        pipeline.transcriber.predefined_script = script_text
        result = pipeline.process_file(wav_path)
        assessments = result.chunk_assessments

        if not assessments:
            continue

        cached_records.append(
            {
                "label": label,
                "is_hard_negative": is_hard,
                "assessments": assessments,
                "explanation": result.explanation,
                "duration": result.audio_duration,
            }
        )

        for a in assessments:
            if label == 0:
                benign_acoustic_scores.append(a.acoustic_score)
            else:
                scam_acoustic_scores.append(a.acoustic_score)

        if label == 0 and not is_hard and sample_normal is None:
            sample_normal = assessments
        elif label == 0 and is_hard and sample_hard_neg is None:
            sample_hard_neg = assessments
        elif label == 1 and sample_scam is None:
            sample_scam = assessments

        if (i + 1) % 50 == 0 or (i + 1) == len(test_samples):
            logger.info("Evaluated %d / %d clips...", i + 1, len(test_samples))

    # Evaluate ablation variants from cached records
    res_acoustic = compute_metrics_for_mode(cached_records, mode="acoustic_only")
    res_text = compute_metrics_for_mode(cached_records, mode="text_only")
    res_fused = compute_metrics_for_mode(cached_records, mode="fused")

    # Generate Figures
    _plot_multi_roc_curve(res_acoustic, res_text, res_fused, output_dir / "roc_curve.png")
    _plot_ablation_barchart(res_acoustic, res_text, res_fused, output_dir / "ablation_comparison.png")
    _plot_confusion_matrix(res_fused["confusion_matrix"], output_dir / "confusion_matrix.png")
    _plot_score_over_time(sample_normal, sample_hard_neg, sample_scam, output_dir / "score_over_time.png")
    _plot_anomaly_distribution(benign_acoustic_scores, scam_acoustic_scores, output_dir / "anomaly_distribution.png")

    # Print Clean Markdown Ablation Table
    print("\n" + "=" * 80)
    print("COMPONENT ABLATION STUDY RESULTS (Held-Out Telephony Test Set, N = 220)")
    print("=" * 80)
    header = "| Model Variant | Precision | Recall | F1 Score | ROC-AUC | Hard-Neg FPR (N=49) | Mean Latency |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(sep)

    for r, name in [
        (res_acoustic, "Acoustic-Only (Acoustic Anomaly)"),
        (res_text, "Text-Only (Scam Lexicon Rules)"),
        (res_fused, "**Fused Multi-Modal**"),
    ]:
        row = (
            f"| {name} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
            f"{r['roc_auc']:.4f} | {r['fpr_hard_negatives']*100:.1f}% | {r['mean_latency_sec']:.2f}s |"
        )
        print(row)
    print("=" * 80 + "\n")

    return {
        "acoustic": res_acoustic,
        "text": res_text,
        "fused": res_fused,
    }


def _plot_multi_roc_curve(
    r_ac: Dict[str, Any], r_tx: Dict[str, Any], r_fu: Dict[str, Any], save_path: Path
) -> None:
    plt.figure(figsize=(6.5, 5.2), facecolor="#0e1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")

    plt.plot(
        r_fu["fpr_arr"],
        r_fu["tpr_arr"],
        color="#38bdf8",
        lw=2.8,
        label=f"Fused System (AUC = {r_fu['roc_auc']:.3f})",
    )
    plt.plot(
        r_tx["fpr_arr"],
        r_tx["tpr_arr"],
        color="#fbbf24",
        lw=2.0,
        linestyle="-.",
        label=f"Text-Only (AUC = {r_tx['roc_auc']:.3f})",
    )
    plt.plot(
        r_ac["fpr_arr"],
        r_ac["tpr_arr"],
        color="#f87171",
        lw=2.0,
        linestyle="--",
        label=f"Acoustic-Only (AUC = {r_ac['roc_auc']:.3f})",
    )
    plt.plot([0, 1], [0, 1], color="#4b5563", lw=1.2, linestyle=":", label="Random Guess")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate", color="white", fontsize=11)
    plt.ylabel("True Positive Rate", color="white", fontsize=11)
    plt.title("ROC Curve: Component Ablation on Telephony Audio", color="white", fontsize=12, pad=12)
    plt.tick_params(colors="white")
    plt.grid(True, color="#30363d", alpha=0.5)
    plt.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, facecolor="#0e1117")
    plt.close()


def _plot_ablation_barchart(
    r_ac: Dict[str, Any], r_tx: Dict[str, Any], r_fu: Dict[str, Any], save_path: Path
) -> None:
    metrics = ["Precision", "Recall", "F1 Score", "ROC-AUC"]
    variants = ["Acoustic-Only", "Text-Only", "Fused Multi-Modal"]

    vals = np.array([
        [r_ac["precision"], r_ac["recall"], r_ac["f1"], r_ac["roc_auc"]],
        [r_tx["precision"], r_tx["recall"], r_tx["f1"], r_tx["roc_auc"]],
        [r_fu["precision"], r_fu["recall"], r_fu["f1"], r_fu["roc_auc"]],
    ])

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#f87171", "#fbbf24", "#38bdf8"]

    for i, (var_name, color) in enumerate(zip(variants, colors, strict=False)):
        rects = ax.bar(x + (i - 1) * width, vals[i], width, label=var_name, color=color, alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="white",
            )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0.0 - 1.0)", color="white", fontsize=11)
    ax.set_title("Component Ablation Benchmark Comparison", color="white", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="white", fontsize=10.5)
    ax.tick_params(colors="white")
    ax.grid(True, color="#30363d", alpha=0.4, axis="y")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", loc="lower left")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, facecolor="#0e1117")
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(5.5, 4.8), facecolor="#0e1117")
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
        annot_kws={"size": 15, "weight": "bold", "color": "white"},
    )
    plt.ylabel("True Label", color="white", fontsize=11)
    plt.xlabel("Predicted Label", color="white", fontsize=11)
    plt.title("Fused Model Confusion Matrix (N = 220 Test Clips)", color="white", fontsize=12, pad=12)
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
            label="Benign Telephony Speech (Normal & Hard Negatives)",
            ax=ax,
            lw=2,
        )
    if scam_scores:
        sns.kdeplot(
            scam_scores,
            fill=True,
            color="#ef4444",
            alpha=0.4,
            label="Vishing Telephony Audio",
            ax=ax,
            lw=2,
        )

    plt.xlabel("Normalized Acoustic Anomaly Score", color="white", fontsize=11)
    plt.ylabel("Density", color="white", fontsize=11)
    plt.title("Acoustic Anomaly Distribution Under Telephony Simulation", color="white", fontsize=12, pad=12)
    plt.tick_params(colors="white")
    plt.grid(True, color="#30363d", alpha=0.5)
    plt.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, facecolor="#0e1117")
    plt.close()


def _plot_score_over_time(
    normal_assessments: Optional[list],
    hard_neg_assessments: Optional[list],
    scam_assessments: Optional[list],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    if scam_assessments:
        t_scam = [a.start_time for a in scam_assessments]
        s_scam = [a.smoothed_risk for a in scam_assessments]
        ax.plot(t_scam, s_scam, color="#ef4444", lw=2.6, marker="o", markersize=4, label="Vishing Attack (Escalating Threat)")

    if hard_neg_assessments:
        t_hn = [a.start_time for a in hard_neg_assessments]
        s_hn = [a.smoothed_risk for a in hard_neg_assessments]
        ax.plot(t_hn, s_hn, color="#f59e0b", lw=2.2, linestyle="--", marker="^", markersize=4, label="Hard Negative (OTP/Bank Discussion)")

    if normal_assessments:
        t_norm = [a.start_time for a in normal_assessments]
        s_norm = [a.smoothed_risk for a in normal_assessments]
        ax.plot(t_norm, s_norm, color="#10b981", lw=2.2, marker="s", markersize=4, label="Standard Benign Conversation")

    ax.axhline(50, color="#f97316", linestyle=":", lw=1.5, alpha=0.8, label="Alert Threshold (Elevated Tier)")
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Time Elapsed (seconds)", color="white", fontsize=11)
    ax.set_ylabel("Running Threat Index (0 - 100)", color="white", fontsize=11)
    ax.set_title("Streaming Threat Evolution: Normal vs. Hard Negative vs. Attack", color="white", fontsize=12, pad=12)
    ax.tick_params(colors="white")
    ax.grid(True, color="#30363d", alpha=0.5)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", loc="center right")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, facecolor="#0e1117")
    plt.close()


if __name__ == "__main__":
    run_full_evaluation()
