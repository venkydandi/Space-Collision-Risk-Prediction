"""Screening service integration and pipeline tests."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_THRESHOLDS,
    MODEL_SMOTENC_KEY,
    MODEL_UNWEIGHTED_KEY,
    MODELS_DIR,
    OUTPUT_COLS,
)
from core.model_loader import load_model_bundle
from services.screening_service import run_screening_pipeline


class ScreeningServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models, cls.metadata = load_model_bundle(MODELS_DIR)

    def setUp(self):
        self.sample_rows = [
            {"event_id": "EVT-1", "time_to_tca": 2.40, "max_risk_estimate": -4.2, "max_risk_scaling": 12.0, "miss_distance": 500.0, "mahalanobis_distance": 8.0, "c_obs_used": 40.0, "c_object_type": "DEBRIS", "extra_meta": "alpha"},
            {"event_id": "EVT-2_REJECTED", "time_to_tca": 1.50, "max_risk_estimate": -5.0, "max_risk_scaling": 10.0, "miss_distance": 1200.0, "mahalanobis_distance": 12.0, "c_obs_used": 30.0, "c_object_type": "PAYLOAD", "extra_meta": "beta"},
            {"event_id": "EVT-3", "time_to_tca": 2.80, "max_risk_estimate": -7.0, "max_risk_scaling": 3.0, "miss_distance": 6000.0, "mahalanobis_distance": 35.0, "c_obs_used": 16.0, "c_object_type": "ROCKET BODY", "extra_meta": "gamma"},
        ]
        self.df = pd.DataFrame(self.sample_rows)

    def test_pipeline_execution_and_row_preservation(self):
        res = run_screening_pipeline(
            data=self.df,
            models=self.models,
            primary_model_key=MODEL_UNWEIGHTED_KEY,
            thresholds=DEFAULT_THRESHOLDS,
        )

        full = res.full_table
        self.assertEqual(len(full), 3)

        # Verify row 1 (Accepted)
        self.assertEqual(full.iloc[0][OUTPUT_COLS["validation_status"]], "Accepted")
        self.assertTrue(pd.notna(full.iloc[0][OUTPUT_COLS["unweighted_score"]]))
        self.assertTrue(pd.notna(full.iloc[0][OUTPUT_COLS["smotenc_score"]]))
        self.assertTrue(pd.notna(full.iloc[0][OUTPUT_COLS["model_agreement"]]))

        # Verify row 2 (Rejected) - MUST NOT have fake predictions
        self.assertEqual(full.iloc[1][OUTPUT_COLS["validation_status"]], "Rejected")
        self.assertTrue(pd.isna(full.iloc[1][OUTPUT_COLS["unweighted_score"]]))
        self.assertTrue(pd.isna(full.iloc[1][OUTPUT_COLS["smotenc_score"]]))
        self.assertTrue(pd.isna(full.iloc[1][OUTPUT_COLS["model_agreement"]]))
        self.assertIn("screening window", full.iloc[1][OUTPUT_COLS["rejection_reason"]])

        # Verify extra meta columns are preserved
        self.assertEqual(list(full["extra_meta"]), ["alpha", "beta", "gamma"])

    def test_summary_metrics_consistency(self):
        res = run_screening_pipeline(
            data=self.df,
            models=self.models,
            primary_model_key=MODEL_UNWEIGHTED_KEY,
            thresholds=DEFAULT_THRESHOLDS,
        )
        s = res.summary
        self.assertEqual(s.total_records, 3)
        self.assertEqual(s.accepted_records, 2)
        self.assertEqual(s.rejected_records, 1)
        self.assertEqual(s.accepted_records + s.rejected_records, s.total_records)
        self.assertEqual(s.high_risk_count + s.low_risk_count, s.accepted_records)
        self.assertEqual(s.agreement_count + s.disagreement_count, s.accepted_records)

    def test_independent_threshold_impact(self):
        # With threshold set very low (0.01), both accepted events should be flagged high risk
        low_res = run_screening_pipeline(
            data=self.df,
            models=self.models,
            primary_model_key=MODEL_UNWEIGHTED_KEY,
            thresholds={MODEL_UNWEIGHTED_KEY: 0.01, MODEL_SMOTENC_KEY: 0.01},
        )
        # With threshold set very high (0.50), high risk count decreases
        high_res = run_screening_pipeline(
            data=self.df,
            models=self.models,
            primary_model_key=MODEL_UNWEIGHTED_KEY,
            thresholds={MODEL_UNWEIGHTED_KEY: 0.50, MODEL_SMOTENC_KEY: 0.50},
        )
        self.assertGreaterEqual(low_res.summary.high_risk_count, high_res.summary.high_risk_count)


if __name__ == "__main__":
    unittest.main()
