"""Unsupervised acoustic anomaly detection models for vishing voice pattern identification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)


@dataclass
class AnomalyInferenceResult:
    """Detailed anomaly scoring outcome for an audio chunk.

    Attributes:
        is_anomaly: Binary flag (True if score exceeds threshold).
        anomaly_score: Normalized anomaly severity in [0.0, 1.0].
        raw_score: Raw decision function output from underlying estimator.
        top_deviating_features: List of (feature_name, z_score_deviation) for explainability.
    """

    is_anomaly: bool
    anomaly_score: float
    raw_score: float
    top_deviating_features: List[Tuple[str, float]]


class AcousticAnomalyDetector:
    """Unsupervised anomaly detector operating on acoustic feature representations.

    Supports:
    - Isolation Forest (default, robust against high dimensionality and outliers).
    - One-Class SVM with RBF kernel (boundary learning around normal speech).
    - StandardScaler normalization fitted strictly on normal conversational data.
    - Automatic percentile-based threshold calibration.
    - Model persistence via joblib.
    """

    SUPPORTED_MODELS = {"isolation_forest", "one_class_svm"}

    def __init__(
        self,
        model_type: str = "isolation_forest",
        contamination: float = 0.05,
        random_state: int = 42,
        nu: float = 0.05,
        gamma: str = "scale",
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize acoustic anomaly detector.

        Args:
            model_type: Either 'isolation_forest' or 'one_class_svm'.
            contamination: Expected proportion of outliers in the training data (for iForest).
            random_state: Seed for reproducibility.
            nu: Margin parameter for OneClassSVM.
            gamma: Kernel coefficient for OneClassSVM.
            feature_names: Optional list of acoustic feature column names for explainability.
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choose from {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.contamination = contamination
        self.random_state = random_state
        self.nu = nu
        self.gamma = gamma
        self.feature_names = feature_names

        self.scaler = StandardScaler()
        self.model: Optional[Union[IsolationForest, OneClassSVM]] = None
        self.calibrated_threshold: float = 0.5
        self._is_fitted: bool = False

        self._init_estimator()

    def _init_estimator(self) -> None:
        """Construct the underlying scikit-learn anomaly estimator."""
        if self.model_type == "isolation_forest":
            self.model = IsolationForest(
                n_estimators=150,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "one_class_svm":
            self.model = OneClassSVM(
                kernel="rbf",
                nu=self.nu,
                gamma=self.gamma,
            )

    @property
    def is_fitted(self) -> bool:
        """Check whether the scaler and model have been trained."""
        return self._is_fitted

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        calibrate_threshold: bool = True,
        threshold_percentile: float = 95.0,
    ) -> AcousticAnomalyDetector:
        """Fit scaler and anomaly model on normal/benign acoustic feature vectors.

        Args:
            X: Matrix of acoustic features (n_samples, n_features).
            calibrate_threshold: Whether to compute decision boundary at target percentile.
            threshold_percentile: Percentile of normal scores used as anomaly cutoff (default 95th).

        Returns:
            Self instance.
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                # Filter out metadata columns if present
                meta_cols = {"chunk_index", "start_time", "end_time"}
                self.feature_names = [col for col in X.columns if col not in meta_cols]
            X_mat = X[self.feature_names].values
        else:
            X_mat = np.asarray(X)

        if len(X_mat) == 0:
            raise ValueError("Training dataset cannot be empty.")

        logger.info(
            "Fitting %s anomaly detector on %d benign acoustic samples (%d features)",
            self.model_type,
            X_mat.shape[0],
            X_mat.shape[1],
        )

        X_scaled = self.scaler.fit_transform(X_mat)
        self.model.fit(X_scaled)
        self._is_fitted = True

        if calibrate_threshold:
            # Calibrate threshold on training distribution
            train_scores = self.score_anomaly(X_mat)
            self.calibrated_threshold = float(np.percentile(train_scores, threshold_percentile))
            logger.info(
                "Calibrated anomaly cutoff threshold at %.2fth percentile: %.4f",
                threshold_percentile,
                self.calibrated_threshold,
            )

        return self

    def score_anomaly(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Compute continuous normalized anomaly scores in [0.0, 1.0], where 1 is highly anomalous.

        Args:
            X: Input feature vectors.

        Returns:
            1D array of normalized anomaly scores in range [0.0, 1.0].
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() before scoring.")

        if isinstance(X, pd.DataFrame):
            feature_cols = self.feature_names or [c for c in X.columns if c not in {"chunk_index", "start_time", "end_time"}]
            X_mat = X[feature_cols].values
        else:
            X_mat = np.asarray(X)
            if X_mat.ndim == 1:
                X_mat = X_mat.reshape(1, -1)

        X_scaled = self.scaler.transform(X_mat)

        # In scikit-learn, decision_function returns negative values for anomalies
        # and positive values for normal samples.
        raw_scores = self.model.decision_function(X_scaled)

        # Invert and apply sigmoid/logistic normalization so higher = more anomalous
        # Sigmoid: 1 / (1 + exp(alpha * raw_score))
        normalized = 1.0 / (1.0 + np.exp(3.0 * raw_scores))
        return np.clip(normalized, 0.0, 1.0)

    def predict_chunk(
        self,
        feature_vector: np.ndarray,
        top_k_features: int = 4,
    ) -> AnomalyInferenceResult:
        """Perform inference on a single audio chunk vector and compute feature attributions.

        Args:
            feature_vector: 1D numpy array of acoustic features.
            top_k_features: Number of top deviating features to return for explanation.

        Returns:
            AnomalyInferenceResult instance.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() before scoring.")

        vec_2d = feature_vector.reshape(1, -1)
        scaled_vec = self.scaler.transform(vec_2d)[0]
        raw_score = float(self.model.decision_function(scaled_vec.reshape(1, -1))[0])
        score = float(self.score_anomaly(vec_2d)[0])
        is_anomaly = score >= self.calibrated_threshold

        # Determine top deviating features via absolute Z-scores from benign scaler
        top_features: List[Tuple[str, float]] = []
        if self.feature_names and len(self.feature_names) == len(scaled_vec):
            abs_z = np.abs(scaled_vec)
            top_indices = np.argsort(abs_z)[::-1][:top_k_features]
            for idx in top_indices:
                feat_name = self.feature_names[idx]
                z_val = float(scaled_vec[idx])
                top_features.append((feat_name, z_val))

        return AnomalyInferenceResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            raw_score=raw_score,
            top_deviating_features=top_features,
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist model, scaler, threshold, and configuration to disk using joblib.

        Args:
            filepath: Target file path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_type": self.model_type,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "nu": self.nu,
            "gamma": self.gamma,
            "scaler": self.scaler,
            "model": self.model,
            "calibrated_threshold": self.calibrated_threshold,
            "feature_names": self.feature_names,
            "_is_fitted": self._is_fitted,
        }
        joblib.dump(state, path)
        logger.info("Saved acoustic anomaly model to: %s", path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> AcousticAnomalyDetector:
        """Load serialized model and state from disk.

        Args:
            filepath: Path to saved joblib file.

        Returns:
            Reconstituted AcousticAnomalyDetector instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = joblib.load(path)
        detector = cls(
            model_type=state["model_type"],
            contamination=state.get("contamination", 0.05),
            random_state=state.get("random_state", 42),
            nu=state.get("nu", 0.05),
            gamma=state.get("gamma", "scale"),
            feature_names=state.get("feature_names"),
        )
        detector.scaler = state["scaler"]
        detector.model = state["model"]
        detector.calibrated_threshold = state.get("calibrated_threshold", 0.5)
        detector._is_fitted = state.get("_is_fitted", True)

        logger.info("Loaded acoustic anomaly model from: %s", path)
        return detector
