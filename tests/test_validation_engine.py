"""Unit tests for CDM validation engine."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from config.settings import TCA_MAX_DAYS, TCA_MIN_DAYS
from core.validation_engine import validate_cdm_dataset


class ValidationEngineTests(unittest.TestCase):
    def setUp(self):
        self.valid_record = {
            "event_id": "EVT-100",
            "time_to_tca": 2.45,
            "max_risk_estimate": -4.50,
            "max_risk_scaling": 10.0,
            "miss_distance": 1200.0,
            "mahalanobis_distance": 12.0,
            "c_obs_used": 30.0,
            "c_object_type": "DEBRIS",
            "mission_id": "SENTINEL-1",
        }

    def test_valid_record_accepted(self):
        df = pd.DataFrame([self.valid_record])
        result = validate_cdm_dataset(df)
        self.assertEqual(result.total_count, 1)
        self.assertEqual(result.accepted_count, 1)
        self.assertEqual(result.rejected_count, 0)
        self.assertTrue(result.valid_mask[0])
        self.assertEqual(result.statuses[0], "Accepted")
        self.assertEqual(result.rejection_reasons[0], "")

    def test_tca_boundary_exact_values(self):
        # Boundaries (inclusive) should be accepted
        r_min = dict(self.valid_record, time_to_tca=TCA_MIN_DAYS)
        r_max = dict(self.valid_record, time_to_tca=TCA_MAX_DAYS)
        # Outside boundaries should be rejected
        r_below = dict(self.valid_record, time_to_tca=TCA_MIN_DAYS - 0.01)
        r_above = dict(self.valid_record, time_to_tca=TCA_MAX_DAYS + 0.01)

        df = pd.DataFrame([r_min, r_max, r_below, r_above])
        result = validate_cdm_dataset(df)

        self.assertEqual(result.valid_mask, [True, True, False, False])
        self.assertIn("screening window", result.rejection_reasons[2])
        self.assertIn("screening window", result.rejection_reasons[3])

    def test_missing_numeric_field(self):
        bad_rec = dict(self.valid_record, miss_distance=None)
        df = pd.DataFrame([bad_rec])
        result = validate_cdm_dataset(df)
        self.assertFalse(result.valid_mask[0])
        self.assertIn("miss_distance", result.rejection_reasons[0])

    def test_infinite_numeric_field(self):
        bad_rec = dict(self.valid_record, mahalanobis_distance=np.inf)
        df = pd.DataFrame([bad_rec])
        result = validate_cdm_dataset(df)
        self.assertFalse(result.valid_mask[0])
        self.assertIn("must be finite", result.rejection_reasons[0])

    def test_non_numeric_string_value(self):
        bad_rec = dict(self.valid_record, max_risk_estimate="NOT_A_NUMBER")
        df = pd.DataFrame([bad_rec])
        result = validate_cdm_dataset(df)
        self.assertFalse(result.valid_mask[0])
        self.assertIn("max_risk_estimate", result.rejection_reasons[0])

    def test_empty_categorical_value(self):
        bad_rec1 = dict(self.valid_record, c_object_type="")
        bad_rec2 = dict(self.valid_record, c_object_type="   ")
        bad_rec3 = dict(self.valid_record, c_object_type="nan")
        bad_rec4 = dict(self.valid_record, c_object_type=None)

        df = pd.DataFrame([bad_rec1, bad_rec2, bad_rec3, bad_rec4])
        result = validate_cdm_dataset(df)
        self.assertEqual(result.valid_mask, [False, False, False, False])
        for r in result.rejection_reasons:
            self.assertIn("c_object_type", r)

    def test_missing_required_column(self):
        incomplete = {k: v for k, v in self.valid_record.items() if k != "miss_distance"}
        df = pd.DataFrame([incomplete])
        result = validate_cdm_dataset(df)
        self.assertFalse(result.valid_mask[0])
        self.assertIn("Missing required feature columns", result.rejection_reasons[0])

    def test_row_order_preservation(self):
        rec1 = dict(self.valid_record, event_id="FIRST")
        rec2 = dict(self.valid_record, event_id="SECOND_REJECTED", time_to_tca=1.5)
        rec3 = dict(self.valid_record, event_id="THIRD")

        df = pd.DataFrame([rec1, rec2, rec3])
        result = validate_cdm_dataset(df)

        self.assertEqual(result.source_rows, [1, 2, 3])
        self.assertEqual(result.display_event_ids, ["FIRST", "SECOND_REJECTED", "THIRD"])
        self.assertEqual(result.valid_mask, [True, False, True])


if __name__ == "__main__":
    unittest.main()

