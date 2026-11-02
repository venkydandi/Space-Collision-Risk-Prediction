"""Validation engine for Conjunction Data Messages (CDMs)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
import numpy as np
import pandas as pd

from config.settings import (
    FEATURES,
    NUMERIC_FEATURES,
    TCA_MAX_DAYS,
    TCA_MIN_DAYS,
)


@dataclass
class ValidationResult:
    """Structured result of CDM validation."""
    clean_features: pd.DataFrame
    valid_mask: List[bool]
    rejection_reasons: List[str]
    statuses: List[str]
    source_rows: List[int]
    display_event_ids: List[str]

    @property
    def total_count(self) -> int:
        return len(self.valid_mask)

    @property
    def accepted_count(self) -> int:
        return sum(self.valid_mask)

    @property
    def rejected_count(self) -> int:
        return self.total_count - self.accepted_count


def _is_empty_or_na(value: Any) -> bool:
    """Check if value is missing, NA, null, or empty."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "<na>"}


def validate_cdm_dataset(data: pd.DataFrame) -> ValidationResult:
    """
    Validate an input DataFrame against the 7-feature schema and TCA horizon.
    
    Preserves original row order and retains all rows.
    Rejected rows receive a specific explanation.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    raw = data.reset_index(drop=True)
    num_rows = len(raw)
    
    # Event ID handling: preserve if present, else use row reference
    display_event_ids: List[str] = []
    if "event_id" in raw.columns:
        for idx, val in enumerate(raw["event_id"]):
            if _is_empty_or_na(val):
                display_event_ids.append(f"CDM-{idx + 1:04d}")
            else:
                display_event_ids.append(str(val).strip())
    else:
        display_event_ids = [f"CDM-{idx + 1:04d}" for idx in range(num_rows)]

    source_rows = list(range(1, num_rows + 1))
    
    missing_columns = [col for col in FEATURES if col not in raw.columns]

    valid_mask: List[bool] = []
    rejection_reasons: List[str] = []
    statuses: List[str] = []
    clean_rows: List[dict[str, Any]] = []

    for row_idx in range(num_rows):
        row = raw.iloc[row_idx]
        row_errors: List[str] = []
        clean_row: dict[str, Any] = {}

        if missing_columns:
            row_errors.append(f"Missing required feature columns: {', '.join(missing_columns)}")
        else:
            # 1. Validate numeric features
            for feature in NUMERIC_FEATURES:
                raw_val = row[feature]
                if _is_empty_or_na(raw_val) or isinstance(raw_val, (bool, np.bool_)):
                    row_errors.append(f"'{feature}' is missing or non-numeric")
                    continue
                try:
                    num_val = float(pd.to_numeric(raw_val, errors="raise"))
                except (TypeError, ValueError, OverflowError):
                    row_errors.append(f"'{feature}' cannot be parsed as a float")
                    continue

                if not np.isfinite(num_val):
                    row_errors.append(f"'{feature}' must be finite (got {raw_val})")
                    continue

                clean_row[feature] = num_val

            # 2. Validate categorical feature: c_object_type
            cat_val = row["c_object_type"]
            if _is_empty_or_na(cat_val):
                row_errors.append("'c_object_type' is empty or missing")
            else:
                cat_str = str(cat_val).strip()
                # Accept raw value for pipeline preprocessing
                clean_row["c_object_type"] = cat_str

            # 3. Validate TCA screening window (2.0 <= time_to_tca <= 2.9)
            if "time_to_tca" in clean_row:
                tca = clean_row["time_to_tca"]
                if not (TCA_MIN_DAYS <= tca <= TCA_MAX_DAYS):
                    row_errors.append(
                        f"time_to_tca ({tca:.4f}) is outside valid screening window [{TCA_MIN_DAYS:.1f}, {TCA_MAX_DAYS:.1f}] days"
                    )

        is_valid = len(row_errors) == 0
        valid_mask.append(is_valid)
        if is_valid:
            statuses.append("Accepted")
            rejection_reasons.append("")
            clean_rows.append(clean_row)
        else:
            statuses.append("Rejected")
            rejection_reasons.append("; ".join(row_errors))
            clean_rows.append({})

    # Build clean features DataFrame strictly from accepted rows in canonical order
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    if valid_indices:
        clean_features = pd.DataFrame([clean_rows[i] for i in valid_indices], columns=FEATURES)
    else:
        clean_features = pd.DataFrame(columns=FEATURES)

    return ValidationResult(
        clean_features=clean_features,
        valid_mask=valid_mask,
        rejection_reasons=rejection_reasons,
        statuses=statuses,
        source_rows=source_rows,
        display_event_ids=display_event_ids,
    )

