"""Unit tests for unsupervised acoustic anomaly detection models."""

from pathlib import Path

import numpy as np

from vishing_detector.anomaly.detector import AcousticAnomalyDetector


def test_isolation_forest_training_and_scoring(tmp_path: Path):
    feature_names = [f"feat_{i}" for i in range(20)]
    np.random.seed(42)
    # Generate synthetic normal distribution
    X_train = np.random.normal(loc=0.0, scale=1.0, size=(100, 20))

    detector = AcousticAnomalyDetector(
        model_type="isolation_forest",
        feature_names=feature_names,
        random_state=42,
    )
    detector.fit(X_train, calibrate_threshold=True, threshold_percentile=90.0)

    assert detector.is_fitted
    assert 0.0 < detector.calibrated_threshold < 1.0

    # Normal sample
    x_normal = np.random.normal(loc=0.0, scale=1.0, size=20)
    res_normal = detector.predict_chunk(x_normal)
    assert 0.0 <= res_normal.anomaly_score <= 1.0

    # Outlier sample (extreme deviation)
    x_outlier = np.ones(20) * 10.0
    res_outlier = detector.predict_chunk(x_outlier)
    assert res_outlier.anomaly_score > res_normal.anomaly_score
    assert len(res_outlier.top_deviating_features) == 4

    # Save and reload persistence
    save_file = tmp_path / "model.joblib"
    detector.save(save_file)
    assert save_file.exists()

    loaded = AcousticAnomalyDetector.load(save_file)
    assert loaded.is_fitted
    assert loaded.model_type == "isolation_forest"


def test_one_class_svm_training():
    feature_names = [f"feat_{i}" for i in range(10)]
    X_train = np.random.normal(loc=0.0, scale=1.0, size=(50, 10))

    detector = AcousticAnomalyDetector(
        model_type="one_class_svm",
        feature_names=feature_names,
    )
    detector.fit(X_train)
    assert detector.is_fitted

    scores = detector.score_anomaly(X_train[:5])
    assert len(scores) == 5
    assert (scores >= 0.0).all() and (scores <= 1.0).all()
