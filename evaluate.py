"""System evaluation engine with full component ablation, bootstrap CIs, and error analysis.

Evaluates 5 distinct system variants on held-out telephony test data (N = 300):
1. Acoustic-Only (Isolation Forest on telephony audio)
2. Text-Only Baseline (Legacy bare-keyword matching)
3. Text-Only Intent-Aware (Proximity demand matching + inquiry dampener)
4. Fused Hand-Tuned (Legacy heuristic weights + synergy multiplier)
5. Fused Learned Stacker (Platt-calibrated Logistic Regression meta-classifier)

Metrics computed:
- Precision, Recall, F1 Score
- ROC-AUC and PR-AUC
- Hard-Negative False Positive Rate (N = 64)
- Plain-Benign False Positive Rate (N = 86)
- Exact Detection Latency (Mean & Median seconds from chunk timestamps)
- 95% Bootstrap Confidence Intervals (1,000 resamples) for F1, ROC-AUC, and Hard-Negative FPR
- Per-voice-variant breakdown across held-out speaker profiles
- Granular error analysis: top 5 false positives and top 5 false negatives
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from vishing_detector.anomaly.detector import AcousticAnomalyDetector
from vishing_detector.audio.loader import AudioLoader
from vishing_detector.features.acoustic import AcousticFeatureExtractor
from vishing_detector.fusion.scorer import RiskScorer
from vishing_detector.fusion.stacker import LogisticRiskStacker
from vishing_detector.nlp.scam_cues import ScamCueAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    hard_neg_mask: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for F1, ROC-AUC, and Hard-Negative FPR."""
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    boot_f1: List[float] = []
    boot_roc: List[float] = []
    boot_fpr_hard: List[float] = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        ys_b = y_scores[idx]
        hn_b = hard_neg_mask[idx]

        # F1
        boot_f1.append(float(f1_score(yt_b, yp_b, zero_division=0)))

        # ROC-AUC (requires both classes present in resample)
        if len(np.unique(yt_b)) > 1:
            fpr, tpr, _ = roc_curve(yt_b, ys_b)
            boot_roc.append(float(auc(fpr, tpr)))
        else:
            boot_roc.append(0.5)

        # Hard-Negative FPR
        if np.any(hn_b):
            boot_fpr_hard.append(float(np.mean(yp_b[hn_b] == 1)))
        else:
            boot_fpr_hard.append(0.0)

    f1_ci = (float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5)))
    roc_ci = (float(np.percentile(boot_roc, 2.5)), float(np.percentile(boot_roc, 97.5)))
    fpr_hard_ci = (float(np.percentile(boot_fpr_hard, 2.5)), float(np.percentile(boot_fpr_hard, 97.5)))

    return {
        "f1_ci": (round(f1_ci[0], 4), round(f1_ci[1], 4)),
        "roc_auc_ci": (round(roc_ci[0], 4), round(roc_ci[1], 4)),
        "fpr_hard_ci": (round(fpr_hard_ci[0], 4), round(fpr_hard_ci[1], 4)),
    }


def evaluate_variant(
    records: List[Dict[str, Any]],
    variant_name: str,
    threshold: float,
) -> Dict[str, Any]:
    """Compute detailed benchmark metrics for a specific variant across test records."""
    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[float] = []
    latencies: List[float] = []
    hard_neg_flags: List[bool] = []
    plain_benign_flags: List[bool] = []

    hard_neg_total = 0
    hard_neg_fp = 0
    plain_benign_total = 0
    plain_benign_fp = 0

    for rec in records:
        label = rec["label"]
        is_hard = rec["is_hard_negative"]
        call_score = rec["scores"][variant_name]
        chunk_timeline = rec["chunk_timelines"][variant_name]

        pred = 1 if call_score >= threshold else 0

        # Latency on true positives
        if label == 1:
            first_alert = None
            for end_time, sc in chunk_timeline:
                if sc >= threshold:
                    first_alert = end_time
                    break
            if first_alert is not None:
                latencies.append(first_alert)

        if label == 0:
            if is_hard:
                hard_neg_total += 1
                hard_neg_flags.append(True)
                plain_benign_flags.append(False)
                if pred == 1:
                    hard_neg_fp += 1
            else:
                plain_benign_total += 1
                hard_neg_flags.append(False)
                plain_benign_flags.append(True)
                if pred == 1:
                    plain_benign_fp += 1
        else:
            hard_neg_flags.append(False)
            plain_benign_flags.append(False)

        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(call_score / 100.0)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_scores_arr = np.array(y_scores)
    hard_neg_arr = np.array(hard_neg_flags)

    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    fpr_arr, tpr_arr, _ = roc_curve(y_true_arr, y_scores_arr)
    roc_auc = float(auc(fpr_arr, tpr_arr))
    pr_auc = float(average_precision_score(y_true_arr, y_scores_arr))
    prec_curve, rec_curve, _ = precision_recall_curve(y_true_arr, y_scores_arr)

    cm = confusion_matrix(y_true_arr, y_pred_arr)

    fpr_hard = float(hard_neg_fp / hard_neg_total) if hard_neg_total > 0 else 0.0
    fpr_plain = float(plain_benign_fp / plain_benign_total) if plain_benign_total > 0 else 0.0

    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    median_lat = float(np.median(latencies)) if latencies else 0.0

    # 95% Bootstrap CIs
    ci_dict = compute_bootstrap_ci(y_true_arr, y_pred_arr, y_scores_arr, hard_neg_arr)

    return {
        "variant": variant_name,
        "calibrated_threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "fpr_hard_negatives": round(fpr_hard, 4),
        "hard_neg_fp_count": hard_neg_fp,
        "hard_neg_total": hard_neg_total,
        "fpr_plain_benign": round(fpr_plain, 4),
        "plain_benign_fp_count": plain_benign_fp,
        "plain_benign_total": plain_benign_total,
        "mean_latency_sec": round(mean_lat, 2),
        "median_latency_sec": round(median_lat, 2),
        "bootstrap_ci": ci_dict,
        "confusion_matrix": cm.tolist(),
        "fpr_arr": fpr_arr,
        "tpr_arr": tpr_arr,
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
    }


def run_full_evaluation(
    dataset_dir: Path = Path("data/synthetic"),
    output_figures_dir: Path = Path("docs/figures"),
    results_json_path: Path = Path("results/results_v0.2.json"),
    config_path: Path = Path("config/default.yaml"),
) -> Dict[str, Any]:
    """Execute evaluation across the 5 ablation variants and generate figures & tables."""
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    results_json_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_file = dataset_dir / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_samples = [s for s in data.get("samples", []) if s.get("split") == "test"]
    logger.info("Found %d test samples in manifest.", len(test_samples))

    # Initialize modular engines
    loader = AudioLoader()
    extractor = AcousticFeatureExtractor()
    anomaly_detector = AcousticAnomalyDetector.load("models/acoustic_anomaly_model.joblib")
    nlp_analyzer = ScamCueAnalyzer(lexicon_path="config/scam_lexicon.yaml")
    stacker = LogisticRiskStacker.load("models/stacker_model.joblib")

    scorer_hand = RiskScorer(fusion_mode="hand_tuned")
    scorer_stack = RiskScorer(fusion_mode="stacker", stacker=stacker)

    # Frozen thresholds selected on validation split
    threshold_acoustic = round(anomaly_detector.calibrated_threshold * 100.0, 2)  # ~49.93
    threshold_text_legacy = 35.0
    threshold_text_intent = 35.0
    threshold_fused_hand = 50.0
    threshold_fused_stacker = round(stacker.calibrated_threshold * 100.0, 2)     # ~36.50

    cached_records: List[Dict[str, Any]] = []
    benign_ac_scores: List[float] = []
    hard_neg_ac_scores: List[float] = []
    scam_ac_scores: List[float] = []

    # Stream samples for visual timeline plot
    sample_normal_timeline = None
    sample_hard_timeline = None
    sample_scam_timeline = None

    logger.info("Evaluating all %d test clips across 5 variants...", len(test_samples))
    for item in test_samples:
        wav_path = dataset_dir / item["file"]
        if not wav_path.exists():
            continue

        label = item["label"]
        is_hard = item.get("is_hard_negative", False)
        voice_id = item.get("voice_id", "unknown")
        text = item.get("text", "")

        seg, _ = loader.load(wav_path)
        chunks = loader.chunk_all(seg)
        chunk_vecs = [extractor.extract_chunk(c).vector for c in chunks]

        # 1. Acoustic chunk scoring
        chunk_ac_scores = anomaly_detector.score_anomaly(np.array(chunk_vecs))
        call_ac_score = float(np.percentile(chunk_ac_scores, 80)) if len(chunk_ac_scores) > 1 else float(chunk_ac_scores[0])
        call_ac_score_100 = call_ac_score * 100.0

        if label == 0:
            if is_hard:
                hard_neg_ac_scores.extend(chunk_ac_scores)
            else:
                benign_ac_scores.extend(chunk_ac_scores)
        else:
            scam_ac_scores.extend(chunk_ac_scores)

        # 2. Text scoring: Legacy vs Intent-Aware
        nlp_legacy = nlp_analyzer.analyze_text(text, intent_aware=False)
        nlp_intent = nlp_analyzer.analyze_text(text, intent_aware=True)
        call_tx_legacy_100 = float(nlp_legacy.combined_text_score * 100.0)
        call_tx_intent_100 = float(nlp_intent.combined_text_score * 100.0)

        # 3. Fused Hand-Tuned Streaming
        scorer_hand.reset()
        hand_timeline: List[Tuple[float, float]] = []
        for c_idx, (c, _ac_s) in enumerate(zip(chunks, chunk_ac_scores, strict=False)):
            anom_res = anomaly_detector.predict_chunk(extractor.extract_chunk(c).vector)
            ass = scorer_hand.compute_chunk_risk(c_idx, c.start_time, c.end_time, anom_res, nlp_intent)
            hand_timeline.append((round(c.end_time, 2), ass.smoothed_risk))
        call_fused_hand_100 = hand_timeline[-1][1] if hand_timeline else 0.0

        # 4. Fused Learned Stacker Streaming
        scorer_stack.reset()
        stack_timeline: List[Tuple[float, float]] = []
        for c_idx, (c, _ac_s) in enumerate(zip(chunks, chunk_ac_scores, strict=False)):
            anom_res = anomaly_detector.predict_chunk(extractor.extract_chunk(c).vector)
            ass = scorer_stack.compute_chunk_risk(c_idx, c.start_time, c.end_time, anom_res, nlp_intent)
            stack_timeline.append((round(c.end_time, 2), ass.smoothed_risk))
        call_fused_stack_100 = stack_timeline[-1][1] if stack_timeline else 0.0

        # Timelines for individual modes for latency evaluation
        ac_timeline = [(round(c.end_time, 2), sc * 100.0) for c, sc in zip(chunks, chunk_ac_scores, strict=False)]
        tx_legacy_timeline = [(round(c.end_time, 2), call_tx_legacy_100) for c in chunks]
        tx_intent_timeline = [(round(c.end_time, 2), call_tx_intent_100) for c in chunks]

        # Record timeline samples for score-over-time visualization
        if label == 0 and not is_hard and sample_normal_timeline is None:
            sample_normal_timeline = stack_timeline
        elif label == 0 and is_hard and sample_hard_timeline is None:
            sample_hard_timeline = stack_timeline
        elif label == 1 and sample_scam_timeline is None:
            sample_scam_timeline = stack_timeline

        cached_records.append({
            "file": item["file"],
            "label": label,
            "is_hard_negative": is_hard,
            "voice_id": voice_id,
            "text": text,
            "scores": {
                "acoustic_only": call_ac_score_100,
                "text_only_legacy": call_tx_legacy_100,
                "text_only_intent": call_tx_intent_100,
                "fused_hand_tuned": call_fused_hand_100,
                "fused_stacker": call_fused_stack_100,
            },
            "chunk_timelines": {
                "acoustic_only": ac_timeline,
                "text_only_legacy": tx_legacy_timeline,
                "text_only_intent": tx_intent_timeline,
                "fused_hand_tuned": hand_timeline,
                "fused_stacker": stack_timeline,
            },
            "nlp_intent_matches": [m.matched_text for m in nlp_intent.cue_matches],
            "nlp_intent_inquiry": nlp_intent.inquiry_detected,
        })

    # Evaluate the 5 variants
    res_acoustic = evaluate_variant(cached_records, "acoustic_only", threshold_acoustic)
    res_tx_legacy = evaluate_variant(cached_records, "text_only_legacy", threshold_text_legacy)
    res_tx_intent = evaluate_variant(cached_records, "text_only_intent", threshold_text_intent)
    res_fused_hand = evaluate_variant(cached_records, "fused_hand_tuned", threshold_fused_hand)
    res_fused_stack = evaluate_variant(cached_records, "fused_stacker", threshold_fused_stacker)

    # Per-voice-variant breakdown on test
    voice_ids = sorted(list(set(r["voice_id"] for r in cached_records)))
    voice_breakdown: Dict[str, Dict[str, Any]] = {}
    for v_id in voice_ids:
        v_records = [r for r in cached_records if r["voice_id"] == v_id]
        v_res = evaluate_variant(v_records, "fused_stacker", threshold_fused_stacker)
        voice_breakdown[v_id] = {
            "n_samples": len(v_records),
            "precision": v_res["precision"],
            "recall": v_res["recall"],
            "f1": v_res["f1"],
            "roc_auc": v_res["roc_auc"],
            "fpr_hard_negatives": v_res["fpr_hard_negatives"],
        }

    # Error analysis: Top 5 False Positives and Top 5 False Negatives for Fused Stacker
    benign_records = [r for r in cached_records if r["label"] == 0]
    benign_records.sort(key=lambda r: r["scores"]["fused_stacker"], reverse=True)
    top_false_positives = [
        {
            "file": r["file"],
            "text": r["text"],
            "voice_id": r["voice_id"],
            "is_hard_negative": r["is_hard_negative"],
            "fused_score": round(r["scores"]["fused_stacker"], 2),
            "acoustic_score": round(r["scores"]["acoustic_only"], 2),
            "text_score": round(r["scores"]["text_only_intent"], 2),
            "cues_fired": r["nlp_intent_matches"],
            "inquiry_detected": r["nlp_intent_inquiry"],
        }
        for r in benign_records[:5]
    ]

    scam_records = [r for r in cached_records if r["label"] == 1]
    scam_records.sort(key=lambda r: r["scores"]["fused_stacker"])
    top_false_negatives = [
        {
            "file": r["file"],
            "text": r["text"],
            "voice_id": r["voice_id"],
            "fused_score": round(r["scores"]["fused_stacker"], 2),
            "acoustic_score": round(r["scores"]["acoustic_only"], 2),
            "text_score": round(r["scores"]["text_only_intent"], 2),
            "cues_fired": r["nlp_intent_matches"],
        }
        for r in scam_records[:5]
    ]

    # Save results JSON
    json_output = {
        "dataset_metadata": {
            "total_test_samples": len(cached_records),
            "benign_test_samples": len(benign_records),
            "hard_negative_samples": sum(1 for r in benign_records if r["is_hard_negative"]),
            "plain_benign_samples": sum(1 for r in benign_records if not r["is_hard_negative"]),
            "scam_test_samples": len(scam_records),
            "distinct_heldout_voice_variants": len(voice_ids),
        },
        "variants": {
            "acoustic_only": {k: v for k, v in res_acoustic.items() if not k.endswith("_arr") and not k.endswith("_curve") and not k.startswith("y_")},
            "text_only_legacy": {k: v for k, v in res_tx_legacy.items() if not k.endswith("_arr") and not k.endswith("_curve") and not k.startswith("y_")},
            "text_only_intent": {k: v for k, v in res_tx_intent.items() if not k.endswith("_arr") and not k.endswith("_curve") and not k.startswith("y_")},
            "fused_hand_tuned": {k: v for k, v in res_fused_hand.items() if not k.endswith("_arr") and not k.endswith("_curve") and not k.startswith("y_")},
            "fused_stacker": {k: v for k, v in res_fused_stack.items() if not k.endswith("_arr") and not k.endswith("_curve") and not k.startswith("y_")},
        },
        "per_voice_breakdown": voice_breakdown,
        "error_analysis": {
            "worst_false_positives": top_false_positives,
            "worst_false_negatives": top_false_negatives,
        },
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    logger.info("Saved benchmark evaluation results to %s", results_json_path)

    # Generate Figures
    _plot_all_figures(
        res_acoustic,
        res_tx_legacy,
        res_tx_intent,
        res_fused_hand,
        res_fused_stack,
        benign_ac_scores,
        hard_neg_ac_scores,
        scam_ac_scores,
        sample_normal_timeline,
        sample_hard_timeline,
        sample_scam_timeline,
        output_figures_dir,
    )

    # Print Clean Markdown Ablation Table
    _print_results_table(
        res_acoustic,
        res_tx_legacy,
        res_tx_intent,
        res_fused_hand,
        res_fused_stack,
        len(cached_records),
        res_fused_stack["hard_neg_total"],
    )

    return json_output


def _print_results_table(
    r_ac: Dict[str, Any],
    r_tx_l: Dict[str, Any],
    r_tx_i: Dict[str, Any],
    r_fu_h: Dict[str, Any],
    r_fu_s: Dict[str, Any],
    total_samples: int,
    hard_neg_n: int,
) -> None:
    print("\n" + "=" * 95)
    print(f"COMPONENT ABLATION STUDY RESULTS (Held-Out Telephony Test Set, N = {total_samples})")
    print("=" * 95)
    header = f"| Model Variant | Precision | Recall | F1 Score [95% CI] | ROC-AUC [95% CI] | Hard-Neg FPR (N={hard_neg_n}) | Mean Latency |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(sep)

    variants = [
        ("Acoustic-Only (Acoustic Anomaly)", r_ac),
        ("Text-Only (Legacy Bare Keywords)", r_tx_l),
        ("Text-Only (Intent-Aware Rules)", r_tx_i),
        ("Fused Multi-Modal (Hand-Tuned)", r_fu_h),
        ("**Fused Multi-Modal (Learned Stacker)**", r_fu_s),
    ]

    for name, r in variants:
        f1_ci = f"[{r['bootstrap_ci']['f1_ci'][0]:.3f}, {r['bootstrap_ci']['f1_ci'][1]:.3f}]"
        roc_ci = f"[{r['bootstrap_ci']['roc_auc_ci'][0]:.3f}, {r['bootstrap_ci']['roc_auc_ci'][1]:.3f}]"
        row = (
            f"| {name} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} {f1_ci} | "
            f"{r['roc_auc']:.4f} {roc_ci} | {r['fpr_hard_negatives']*100:.1f}% ({r['hard_neg_fp_count']}/{r['hard_neg_total']}) | {r['mean_latency_sec']:.2f}s |"
        )
        print(row)
    print("=" * 95 + "\n")


def _plot_all_figures(
    r_ac: Dict[str, Any],
    r_tx_l: Dict[str, Any],
    r_tx_i: Dict[str, Any],
    r_fu_h: Dict[str, Any],
    r_fu_s: Dict[str, Any],
    benign_ac: List[float],
    hard_neg_ac: List[float],
    scam_ac: List[float],
    s_normal: Optional[List[Tuple[float, float]]],
    s_hard: Optional[List[Tuple[float, float]]],
    s_scam: Optional[List[Tuple[float, float]]],
    fig_dir: Path,
) -> None:
    # 1. ROC Curve
    plt.figure(figsize=(7, 5.5), facecolor="#0e1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")
    plt.plot(r_fu_s["fpr_arr"], r_fu_s["tpr_arr"], color="#38bdf8", lw=2.6, label=f"Fused Stacker (AUC = {r_fu_s['roc_auc']:.3f})")
    plt.plot(r_fu_h["fpr_arr"], r_fu_h["tpr_arr"], color="#818cf8", lw=2.0, linestyle=":", label=f"Fused Hand-Tuned (AUC = {r_fu_h['roc_auc']:.3f})")
    plt.plot(r_tx_i["fpr_arr"], r_tx_i["tpr_arr"], color="#34d399", lw=2.2, label=f"Text Intent-Aware (AUC = {r_tx_i['roc_auc']:.3f})")
    plt.plot(r_tx_l["fpr_arr"], r_tx_l["tpr_arr"], color="#fbbf24", lw=1.8, linestyle="-.", label=f"Text Legacy (AUC = {r_tx_l['roc_auc']:.3f})")
    plt.plot(r_ac["fpr_arr"], r_ac["tpr_arr"], color="#f87171", lw=1.8, linestyle="--", label=f"Acoustic-Only (AUC = {r_ac['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], color="#4b5563", lw=1.2, linestyle="--")
    plt.title("ROC Curves: 5-Variant Ablation on Held-Out Test Audio", color="white", fontsize=12, pad=12)
    plt.xlabel("False Positive Rate", color="white")
    plt.ylabel("True Positive Rate", color="white")
    plt.tick_params(colors="white")
    plt.grid(True, color="#30363d", alpha=0.5)
    plt.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(fig_dir / "roc_curve_comparison.png"), dpi=200, facecolor="#0e1117")
    plt.savefig(str(fig_dir / "roc_curve.png"), dpi=200, facecolor="#0e1117")
    plt.close()

    # 2. Precision-Recall Curve
    plt.figure(figsize=(7, 5.5), facecolor="#0e1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")
    plt.plot(r_fu_s["rec_curve"], r_fu_s["prec_curve"], color="#38bdf8", lw=2.6, label=f"Fused Stacker (PR-AUC = {r_fu_s['pr_auc']:.3f})")
    plt.plot(r_tx_i["rec_curve"], r_tx_i["prec_curve"], color="#34d399", lw=2.2, label=f"Text Intent-Aware (PR-AUC = {r_tx_i['pr_auc']:.3f})")
    plt.plot(r_tx_l["rec_curve"], r_tx_l["prec_curve"], color="#fbbf24", lw=1.8, linestyle="-.", label=f"Text Legacy (PR-AUC = {r_tx_l['pr_auc']:.3f})")
    plt.plot(r_ac["rec_curve"], r_ac["prec_curve"], color="#f87171", lw=1.8, linestyle="--", label=f"Acoustic-Only (PR-AUC = {r_ac['pr_auc']:.3f})")
    plt.title("Precision-Recall Curves across Ablation Variants", color="white", fontsize=12, pad=12)
    plt.xlabel("Recall", color="white")
    plt.ylabel("Precision", color="white")
    plt.tick_params(colors="white")
    plt.grid(True, color="#30363d", alpha=0.5)
    plt.legend(loc="lower left", facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(fig_dir / "precision_recall_curves.png"), dpi=200, facecolor="#0e1117")
    plt.close()

    # 3. Ablation Bar Chart
    plt.figure(figsize=(9, 5.2), facecolor="#0e1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")
    names = ["Acoustic-Only", "Text Legacy", "Text Intent", "Fused Hand", "Fused Stacker"]
    var_list = [r_ac, r_tx_l, r_tx_i, r_fu_h, r_fu_s]

    x = np.arange(len(names))
    width = 0.20

    p_vals = [v["precision"] for v in var_list]
    r_vals = [v["recall"] for v in var_list]
    f1_vals = [v["f1"] for v in var_list]
    hn_vals = [v["fpr_hard_negatives"] for v in var_list]

    plt.bar(x - 1.5 * width, p_vals, width, label="Precision", color="#60a5fa")
    plt.bar(x - 0.5 * width, r_vals, width, label="Recall", color="#34d399")
    plt.bar(x + 0.5 * width, f1_vals, width, label="F1 Score", color="#f59e0b")
    plt.bar(x + 1.5 * width, hn_vals, width, label="Hard-Neg FPR (Lower is Better)", color="#f87171")

    plt.title("Component Ablation: Performance & False Alarm Rates", color="white", fontsize=12, pad=12)
    plt.xticks(x, names, color="white", fontsize=10)
    plt.yticks(color="white")
    plt.ylabel("Score (0.0 - 1.0)", color="white")
    plt.grid(axis="y", color="#30363d", alpha=0.5)
    plt.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    plt.tight_layout()
    plt.savefig(str(fig_dir / "ablation_comparison.png"), dpi=200, facecolor="#0e1117")
    plt.close()

    # 4. Confusion Matrix (Fused Stacker)
    plt.figure(figsize=(5.5, 4.5), facecolor="#0e1117")
    ax = plt.gca()
    cm = np.array(r_fu_s["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Benign", "Scam"],
        yticklabels=["Benign", "Scam"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix: Fused Stacker (N = {int(np.sum(cm))})", color="white", fontsize=11, pad=12)
    plt.xlabel("Predicted Label", color="white")
    plt.ylabel("True Label", color="white")
    plt.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(str(fig_dir / "confusion_matrix.png"), dpi=200, facecolor="#0e1117")
    plt.close()

    # 5. Score Over Time Streaming Timeline
    plt.figure(figsize=(8, 4.5), facecolor="#0e1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")
    if s_scam:
        t_scam, sc_scam = zip(*s_scam, strict=False)
        plt.plot(t_scam, sc_scam, color="#ef4444", lw=2.4, marker="o", label="Scam Call Timeline")
    if s_hard:
        t_hard, sc_hard = zip(*s_hard, strict=False)
        plt.plot(t_hard, sc_hard, color="#f59e0b", lw=2.0, marker="s", label="Hard Negative Call (Bank OTP Inquiry)")
    if s_normal:
        t_norm, sc_norm = zip(*s_normal, strict=False)
        plt.plot(t_norm, sc_norm, color="#10b981", lw=2.0, marker="^", label="Standard Benign Call")

    plt.axhline(r_fu_s["calibrated_threshold"], color="#38bdf8", linestyle="--", lw=1.5, label=f"Alert Threshold ({r_fu_s['calibrated_threshold']:.1f})")
    plt.title("Simulated Streaming Threat Score Evolution Over Time", color="white", fontsize=12, pad=12)
    plt.xlabel("Time in Stream (Seconds)", color="white")
    plt.ylabel("Calibrated Risk Score (0 - 100)", color="white")
    plt.ylim([-5, 105])
    plt.tick_params(colors="white")
    plt.grid(True, color="#30363d", alpha=0.5)
    plt.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(fig_dir / "score_over_time.png"), dpi=200, facecolor="#0e1117")
    plt.close()

    # 6. Anomaly Score Distributions
    plt.figure(figsize=(7.5, 4.8), facecolor="#0e1117")
    ax = plt.gca()
    ax.set_facecolor("#161b22")
    if benign_ac:
        sns.kdeplot(benign_ac, color="#34d399", fill=True, alpha=0.3, label=f"Plain Benign (Mean={np.mean(benign_ac):.3f})")
    if hard_neg_ac:
        sns.kdeplot(hard_neg_ac, color="#fbbf24", fill=True, alpha=0.3, label=f"Hard Negatives (Mean={np.mean(hard_neg_ac):.3f})")
    if scam_ac:
        sns.kdeplot(scam_ac, color="#f87171", fill=True, alpha=0.3, label=f"Scam Audio (Mean={np.mean(scam_ac):.3f})")

    plt.title("Acoustic Anomaly Score Distributions by Audio Class", color="white", fontsize=12, pad=12)
    plt.xlabel("Acoustic Anomaly Score (0.0 - 1.0)", color="white")
    plt.ylabel("Density", color="white")
    plt.tick_params(colors="white")
    plt.grid(True, color="#30363d", alpha=0.5)
    plt.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    plt.tight_layout()
    plt.savefig(str(fig_dir / "anomaly_distribution.png"), dpi=200, facecolor="#0e1117")
    plt.close()


if __name__ == "__main__":
    run_full_evaluation()
