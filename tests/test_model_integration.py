"""Integration tests with real trained model artifacts."""
from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_THRESHOLDS,
    FEATURES,
    MODEL_LABELS,
    MODEL_SMOTENC_KEY,
    MODEL_UNWEIGHTED_KEY,
    MODELS_DIR,
)
from core.model_loader import compute_file_sha256, load_model_bundle
from core.prediction_engine import (
    classify_scores,
    evaluate_dual_models,
    extract_positive_class_probabilities,
)


class RealModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models, cls.metadata = load_model_bundle(MODELS_DIR)

    def test_both_models_loaded(self):
        self.assertIn(MODEL_UNWEIGHTED_KEY, self.models)
        self.assertIn(MODEL_SMOTENC_KEY, self.models)
        self.assertTrue(hasattr(self.models[MODEL_UNWEIGHTED_KEY], "predict_proba"))
        self.assertTrue(hasattr(self.models[MODEL_SMOTENC_KEY], "predict_proba"))

    def test_model_artifact_checksums(self):
        m1_path = MODELS_DIR / "rf_unweighted.joblib"
        m2_path = MODELS_DIR / "rf_smotenc_025.joblib"
        self.assertTrue(m1_path.is_file())
        self.assertTrue(m2_path.is_file())
        
        m1_hash = compute_file_sha256(m1_path)
        m2_hash = compute_file_sha256(m2_path)
        
        # Verify exact SHA256 integrity
        self.assertEqual(m1_hash, "CE9B055DFC0DAADF439C04ABCEEDA42C859D3443A097E259F962D97B2140CBE1")
        self.assertEqual(m2_hash, "CBAD36F93B57A1951F4FDA7DB61D1AD2E52FD555CC13453E02074130536C1DA9")

    def test_real_inference_unweighted_model(self):
        sample_df = pd.DataFrame([{
            "time_to_tca": 2.45,
            "max_risk_estimate": -4.20,
            "max_risk_scaling": 12.5,
            "miss_distance": 480.0,
            "mahalanobis_distance": 8.5,
            "c_obs_used": 42.0,
            "c_object_type": "DEBRIS",
        }])
        scores = extract_positive_class_probabilities(self.models[MODEL_UNWEIGHTED_KEY], sample_df)
        self.assertEqual(len(scores), 1)
        self.assertTrue(0.0 <= scores[0] <= 1.0)
        self.assertTrue(np.isfinite(scores[0]))

    def test_real_inference_smotenc_model(self):
        sample_df = pd.DataFrame([{
            "time_to_tca": 2.45,
            "max_risk_estimate": -4.20,
            "max_risk_scaling": 12.5,
            "miss_distance": 480.0,
            "mahalanobis_distance": 8.5,
            "c_obs_used": 42.0,
            "c_object_type": "DEBRIS",
        }])
        scores = extract_positive_class_probabilities(self.models[MODEL_SMOTENC_KEY], sample_df)
        self.assertEqual(len(scores), 1)
        self.assertTrue(0.0 <= scores[0] <= 1.0)
        self.assertTrue(np.isfinite(scores[0]))

    def test_feature_order_is_strict(self):
        # Even if DataFrame columns are shuffled or contain extra target fields,
        # extract_positive_class_probabilities selects strictly the canonical 7 features
        shuffled_df = pd.DataFrame([{
            "extra_field": "ignore_me",
            "c_object_type": "PAYLOAD",
            "miss_distance": 1500.0,
            "risk": -10.5,
            "time_to_tca": 2.20,
            "c_obs_used": 30.0,
            "max_risk_scaling": 5.0,
            "final_risk": 0,
            "max_risk_estimate": -6.0,
            "mahalanobis_distance": 15.0,
        }])
        scores = extract_positive_class_probabilities(self.models[MODEL_UNWEIGHTED_KEY], shuffled_df)
        self.assertEqual(len(scores), 1)
        self.assertTrue(0.0 <= scores[0] <= 1.0)

    def test_threshold_classification(self):
        scores = np.array([0.05, 0.10, 0.15, 0.25])
        classes, labels = classify_scores(scores, threshold=0.10)
        self.assertEqual(list(classes), [0, 1, 1, 1])
        self.assertEqual(labels, ["Low risk", "High risk", "High risk", "High risk"])

    def test_dual_model_evaluation(self):
        batch_df = pd.DataFrame([
            {"time_to_tca": 2.10, "max_risk_estimate": -7.0, "max_risk_scaling": 2.0, "miss_distance": 8000.0, "mahalanobis_distance": 40.0, "c_obs_used": 15.0, "c_object_type": "DEBRIS"},
            {"time_to_tca": 2.80, "max_risk_estimate": -3.5, "max_risk_scaling": 25.0, "miss_distance": 150.0, "mahalanobis_distance": 4.0, "c_obs_used": 60.0, "c_object_type": "PAYLOAD"},
        ])
        results = evaluate_dual_models(self.models, batch_df, DEFAULT_THRESHOLDS)
        self.assertEqual(len(results[f"{MODEL_UNWEIGHTED_KEY}_scores"]), 2)
        self.assertEqual(len(results[f"{MODEL_SMOTENC_KEY}_scores"]), 2)
        self.assertEqual(len(results["agreement_labels"]), 2)


if __name__ == "__main__":
    unittest.main()
