"""Unit tests for Export Service."""
from __future__ import annotations

import io
import unittest
import pandas as pd

from config.settings import OUTPUT_COLS
from services.export_service import (
    generate_cdm_template_csv,
    generate_executive_summary_text,
    generate_results_csv,
)
from services.screening_service import ScreeningBatchSummary


class ExportServiceTests(unittest.TestCase):
    def test_results_csv_generation(self):
        sample_df = pd.DataFrame([{
            OUTPUT_COLS["source_row"]: 1,
            OUTPUT_COLS["event_id"]: "EVT-1",
            OUTPUT_COLS["validation_status"]: "Accepted",
            OUTPUT_COLS["rejection_reason"]: "",
            OUTPUT_COLS["primary_score"]: 0.22,
            OUTPUT_COLS["primary_class"]: "High risk",
            "time_to_tca": 2.45,
            "miss_distance": 500.0,
        }])
        csv_bytes = generate_results_csv(sample_df)
        self.assertIsInstance(csv_bytes, bytes)
        self.assertTrue(len(csv_bytes) > 0)
        
        # Verify decoding back
        recovered = pd.read_csv(io.BytesIO(csv_bytes))
        self.assertEqual(len(recovered), 1)
        self.assertIn(OUTPUT_COLS["event_id"], recovered.columns)
        self.assertIn(OUTPUT_COLS["primary_score"], recovered.columns)

    def test_cdm_template_csv_generation(self):
        template_bytes = generate_cdm_template_csv()
        recovered = pd.read_csv(io.BytesIO(template_bytes))
        required = ["time_to_tca", "max_risk_estimate", "max_risk_scaling", "miss_distance", "mahalanobis_distance", "c_obs_used", "c_object_type"]
        for r in required:
            self.assertIn(r, recovered.columns)

    def test_executive_summary_text_generation(self):
        summary = ScreeningBatchSummary(
            total_records=10,
            accepted_records=8,
            rejected_records=2,
            primary_model_key="RF_unweighted",
            primary_model_name="Random Forest — Unweighted",
            primary_threshold=0.10,
            high_risk_count=3,
            low_risk_count=5,
            unweighted_high_risk=3,
            unweighted_low_risk=5,
            smotenc_high_risk=4,
            smotenc_low_risk=4,
            agreement_count=7,
            disagreement_count=1,
        )
        report_text = generate_executive_summary_text(summary)
        self.assertIn("Total CDM Records Processed : 10", report_text)
        self.assertIn("Accepted Records (Screened) : 8", report_text)
        self.assertIn("High Risk Events Identified : 3", report_text)
        self.assertIn("Model Disagreements         : 1", report_text)


if __name__ == "__main__":
    unittest.main()

