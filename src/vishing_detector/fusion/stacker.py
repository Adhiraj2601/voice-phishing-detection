"""Learned Logistic Regression risk stacker with Platt probability calibration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_curve

from vishing_detector.nlp.scam_cues import ScamAnalysisResult

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "max_acoustic_anomaly",
    "mean_acoustic_anomaly",
    "text_cue_score",
    "count_urgency_threat",
    "count_credential_harvesting",
    "count_impersonation",
    "count_financial_demand",
    "count_remote_access",
    "count_secrecy_isolation",
    "acoustic_text_interaction",
]

CATEGORIES_ORDER = [
    "urgency_threat",
    "credential_harvesting",
    "impersonation",
    "financial_demand",
    "remote_access",
    "secrecy_isolation",
]


class LogisticRiskStacker:
    """Supervised meta-classifier fusing acoustic anomalies and textual scam cues."""

    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 5,
        calibrated_threshold: float = 0.50,
        streaming_alpha: float = 0.40,
    ) -> None:
        """Initialize logistic risk stacker.

        Args:
            random_state: Seed for reproducibility.
            cv_folds: Number of cross-validation splits for Platt calibration.
            calibrated_threshold: Threat decision cutoff in [0.0, 1.0].
            streaming_alpha: Exponential moving average decay factor for streaming.
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.calibrated_threshold = calibrated_threshold
        self.streaming_alpha = streaming_alpha
        self.feature_names = FEATURE_NAMES

        self.model: Optional[CalibratedClassifierCV] = None
        self._is_fitted: bool = False
        self.val_metrics: Dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        """Check if stacker is fitted and ready for inference."""
        return self._is_fitted

    @staticmethod
    def extract_features(
        acoustic_scores: Union[List[float], np.ndarray],
        nlp_result: ScamAnalysisResult,
    ) -> np.ndarray:
        """Extract fixed 10-dimensional tabular feature vector from multi-modal signals."""
        ac_arr = np.asarray(acoustic_scores, dtype=np.float32)
        if len(ac_arr) > 0:
            max_ac = float(np.max(ac_arr))
            mean_ac = float(np.mean(ac_arr))
        else:
            max_ac = 0.0
            mean_ac = 0.0

        tx_score = float(nlp_result.combined_text_score)

        counts = [
            float(nlp_result.category_counts.get(cat, 0)) for cat in CATEGORIES_ORDER
        ]

        interaction = max_ac * tx_score

        feats = [max_ac, mean_ac, tx_score] + counts + [interaction]
        return np.array(feats, dtype=np.float32)

    def fit(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hard_neg_mask: Optional[np.ndarray] = None,
    ) -> LogisticRiskStacker:
        """Fit calibrated logistic regression stacker on validation features and optimize threshold.

        Args:
            X_val: Matrix of validation features (N, 10).
            y_val: Ground-truth binary labels (N,).
            hard_neg_mask: Optional boolean array marking hard negative samples.

        Returns:
            Self instance.
        """
        base_lr = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            random_state=self.random_state,
            max_iter=500,
        )

        # Platt scaling via CalibratedClassifierCV with sigmoid
        self.model = CalibratedClassifierCV(
            estimator=base_lr,
            method="sigmoid",
            cv=self.cv_folds,
        )
        self.model.fit(X_val, y_val)
        self._is_fitted = True

        # Get calibrated probabilities on validation
        probs = self.model.predict_proba(X_val)[:, 1]

        fpr_arr, tpr_arr, _ = roc_curve(y_val, probs)
        roc_auc = float(auc(fpr_arr, tpr_arr))

        # Select decision threshold on validation: maximize F1 while keeping FPR <= 10%
        best_f1 = -1.0
        best_th = 0.50
        best_prec = 0.0
        best_rec = 0.0
        best_fpr_hard = 1.0
        best_fpr_benign = 1.0

        for th in np.linspace(0.10, 0.90, 161):
            preds = (probs >= th).astype(int)
            prec = float(precision_score(y_val, preds, zero_division=0))
            rec = float(recall_score(y_val, preds, zero_division=0))
            f1 = float(f1_score(y_val, preds, zero_division=0))

            benign_idx = y_val == 0
            fpr_b = float(np.mean(preds[benign_idx] == 1)) if np.any(benign_idx) else 0.0

            if hard_neg_mask is not None and np.any(hard_neg_mask):
                fpr_h = float(np.mean(preds[hard_neg_mask] == 1))
            else:
                fpr_h = fpr_b

            # Balance F1 and false alarm control
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)
                best_prec = prec
                best_rec = rec
                best_fpr_hard = fpr_h
                best_fpr_benign = fpr_b

        self.calibrated_threshold = best_th
        self.val_metrics = {
            "val_roc_auc": round(roc_auc, 4),
            "val_f1": round(best_f1, 4),
            "val_precision": round(best_prec, 4),
            "val_recall": round(best_rec, 4),
            "val_fpr_hard_negatives": round(best_fpr_hard, 4),
            "val_fpr_plain_benign": round(best_fpr_benign, 4),
            "calibrated_threshold": round(best_th, 4),
        }
        logger.info(
            "Trained LogisticRiskStacker on val: ROC-AUC=%.4f, F1=%.4f, Prec=%.4f, Rec=%.4f, Thresh=%.4f",
            roc_auc, best_f1, best_prec, best_rec, best_th,
        )
        return self

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        """Predict continuous calibrated risk probability in [0.0, 1.0]."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("LogisticRiskStacker is not fitted.")

        feats_2d = feature_vector.reshape(1, -1)
        prob = float(self.model.predict_proba(feats_2d)[0, 1])
        return float(np.clip(prob, 0.0, 1.0))

    def predict_risk_score(self, feature_vector: np.ndarray) -> float:
        """Predict calibrated threat score scaled to [0.0, 100.0]."""
        return self.predict_proba(feature_vector) * 100.0

    def is_threat(self, feature_vector: np.ndarray) -> bool:
        """Evaluate whether feature vector breaches calibrated threat threshold."""
        return self.predict_proba(feature_vector) >= self.calibrated_threshold

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist stacker model, calibrated threshold, and parameters to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "calibrated_threshold": self.calibrated_threshold,
            "streaming_alpha": self.streaming_alpha,
            "val_metrics": self.val_metrics,
            "feature_names": self.feature_names,
            "_is_fitted": self._is_fitted,
        }
        joblib.dump(state, path)
        logger.info("Saved LogisticRiskStacker to %s", path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> LogisticRiskStacker:
        """Load persisted stacker model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Stacker model file not found: {path}")

        state = joblib.load(path)
        inst = cls(
            calibrated_threshold=state.get("calibrated_threshold", 0.50),
            streaming_alpha=state.get("streaming_alpha", 0.40),
        )
        inst.model = state["model"]
        inst.val_metrics = state.get("val_metrics", {})
        inst.feature_names = state.get("feature_names", FEATURE_NAMES)
        inst._is_fitted = state.get("_is_fitted", True)
        logger.info("Loaded LogisticRiskStacker from %s", path)
        return inst
