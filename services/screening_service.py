"""Screening service orchestrating dataset validation, dual-model inference, and reporting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_THRESHOLDS,
    LABEL_HIGH_RISK,
    LABEL_LOW_RISK,
    MODEL_LABELS,
    MODEL_SMOTENC_KEY,
    MODEL_UNWEIGHTED_KEY,
    OUTPUT_COLS,
)
from core.prediction_engine import evaluate_dual_models
from core.validation_engine import ValidationResult, validate_cdm_dataset


@dataclass
class ScreeningBatchSummary:
    """Summary metrics of a screening batch."""
    total_records: int
    accepted_records: int
    rejected_records: int
    
    # Primary model metrics
    primary_model_key: str
    primary_model_name: str
    primary_threshold: float
    high_risk_count: int
    low_risk_count: int
    
    # Dual model metrics
    unweighted_high_risk: int
    unweighted_low_risk: int
    smotenc_high_risk: int
    smotenc_low_risk: int
    agreement_count: int
    disagreement_count: int
    
    @property
    def acceptance_rate(self) -> float:
        return (self.accepted_records / self.total_records) if self.total_records > 0 else 0.0

    @property
    def high_risk_rate(self) -> float:
        return (self.high_risk_count / self.accepted_records) if self.accepted_records > 0 else 0.0

    @property
    def disagreement_rate(self) -> float:
        return (self.disagreement_count / self.accepted_records) if self.accepted_records > 0 else 0.0


@dataclass
class ScreeningResult:
    """Complete output of the screening service."""
    full_table: pd.DataFrame
    accepted_table: pd.DataFrame
    rejected_table: pd.DataFrame
    summary: ScreeningBatchSummary
    validation: ValidationResult


def run_screening_pipeline(
    data: pd.DataFrame,
    models: Mapping[str, Any],
    primary_model_key: str = MODEL_UNWEIGHTED_KEY,
    thresholds: Optional[Mapping[str, float]] = None,
) -> ScreeningResult:
    """
    Run complete screening pipeline on an input DataFrame.
    
    Preserves original row order, includes both accepted and rejected rows,
    calculates dual model predictions, and records disagreement status.
    """
    if primary_model_key not in MODEL_LABELS:
        primary_model_key = MODEL_UNWEIGHTED_KEY

    current_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        current_thresholds.update(thresholds)

    # 1. Validation
    validation = validate_cdm_dataset(data)
    total_rows = validation.total_count
    valid_mask = validation.valid_mask

    # 2. Dual-Model Inference on accepted subset
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    num_valid = len(valid_indices)

    if num_valid > 0:
        inference_out = evaluate_dual_models(
            models=models,
            clean_features=validation.clean_features,
            thresholds=current_thresholds,
        )
    else:
        inference_out = {
            f"{MODEL_UNWEIGHTED_KEY}_scores": np.array([]),
            f"{MODEL_UNWEIGHTED_KEY}_classes": np.array([]),
            f"{MODEL_UNWEIGHTED_KEY}_labels": [],
            f"{MODEL_UNWEIGHTED_KEY}_threshold": current_thresholds[MODEL_UNWEIGHTED_KEY],
            f"{MODEL_SMOTENC_KEY}_scores": np.array([]),
            f"{MODEL_SMOTENC_KEY}_classes": np.array([]),
            f"{MODEL_SMOTENC_KEY}_labels": [],
            f"{MODEL_SMOTENC_KEY}_threshold": current_thresholds[MODEL_SMOTENC_KEY],
            "agreement_labels": [],
            "is_disagreement": [],
        }

    # 3. Assemble Full Output Table preserving original rows and order
    raw_copy = data.reset_index(drop=True).copy()
    
    # Initialize output columns
    unweighted_scores = [np.nan] * total_rows
    unweighted_classes = [pd.NA] * total_rows
    unweighted_labels = [pd.NA] * total_rows
    
    smotenc_scores = [np.nan] * total_rows
    smotenc_classes = [pd.NA] * total_rows
    smotenc_labels = [pd.NA] * total_rows
    
    agreement_statuses = [pd.NA] * total_rows
    
    primary_scores = [np.nan] * total_rows
    primary_classes = [pd.NA] * total_rows
    primary_labels = [pd.NA] * total_rows

    for local_idx, row_idx in enumerate(valid_indices):
        # Unweighted
        u_score = float(inference_out[f"{MODEL_UNWEIGHTED_KEY}_scores"][local_idx])
        u_cls = int(inference_out[f"{MODEL_UNWEIGHTED_KEY}_classes"][local_idx])
        u_lbl = inference_out[f"{MODEL_UNWEIGHTED_KEY}_labels"][local_idx]
        unweighted_scores[row_idx] = u_score
        unweighted_classes[row_idx] = u_cls
        unweighted_labels[row_idx] = u_lbl

        # SMOTENC
        s_score = float(inference_out[f"{MODEL_SMOTENC_KEY}_scores"][local_idx])
        s_cls = int(inference_out[f"{MODEL_SMOTENC_KEY}_classes"][local_idx])
        s_lbl = inference_out[f"{MODEL_SMOTENC_KEY}_labels"][local_idx]
        smotenc_scores[row_idx] = s_score
        smotenc_classes[row_idx] = s_cls
        smotenc_labels[row_idx] = s_lbl

        # Agreement
        agreement_statuses[row_idx] = inference_out["agreement_labels"][local_idx]

        # Primary
        if primary_model_key == MODEL_UNWEIGHTED_KEY:
            primary_scores[row_idx] = u_score
            primary_classes[row_idx] = u_cls
            primary_labels[row_idx] = u_lbl
        else:
            primary_scores[row_idx] = s_score
            primary_classes[row_idx] = s_cls
            primary_labels[row_idx] = s_lbl

    # Construct the master DataFrame with standard audit columns first, followed by original inputs
    audit_data = {
        OUTPUT_COLS["source_row"]: validation.source_rows,
        OUTPUT_COLS["event_id"]: validation.display_event_ids,
        OUTPUT_COLS["validation_status"]: validation.statuses,
        OUTPUT_COLS["rejection_reason"]: validation.rejection_reasons,
        OUTPUT_COLS["primary_model"]: [MODEL_LABELS[primary_model_key]] * total_rows,
        OUTPUT_COLS["primary_score"]: primary_scores,
        OUTPUT_COLS["primary_class"]: primary_labels,
        OUTPUT_COLS["unweighted_score"]: unweighted_scores,
        OUTPUT_COLS["unweighted_class"]: unweighted_labels,
        OUTPUT_COLS["unweighted_threshold"]: [current_thresholds[MODEL_UNWEIGHTED_KEY]] * total_rows,
        OUTPUT_COLS["smotenc_score"]: smotenc_scores,
        OUTPUT_COLS["smotenc_class"]: smotenc_labels,
        OUTPUT_COLS["smotenc_threshold"]: [current_thresholds[MODEL_SMOTENC_KEY]] * total_rows,
        OUTPUT_COLS["model_agreement"]: agreement_statuses,
    }
    
    audit_df = pd.DataFrame(audit_data)
    
    # Merge audit columns with raw data avoiding collision
    full_table = pd.concat([audit_df, raw_copy], axis=1)

    accepted_table = full_table[full_table[OUTPUT_COLS["validation_status"]] == "Accepted"].reset_index(drop=True)
    rejected_table = full_table[full_table[OUTPUT_COLS["validation_status"]] == "Rejected"].reset_index(drop=True)

    # Calculate summary metrics
    u_high = sum(1 for c in unweighted_classes if pd.notna(c) and c == 1)
    u_low = sum(1 for c in unweighted_classes if pd.notna(c) and c == 0)
    s_high = sum(1 for c in smotenc_classes if pd.notna(c) and c == 1)
    s_low = sum(1 for c in smotenc_classes if pd.notna(c) and c == 0)
    
    disagreements = sum(1 for a in agreement_statuses if pd.notna(a) and a == "Disagreement (Review Required)")
    agreements = sum(1 for a in agreement_statuses if pd.notna(a) and a == "Agreement")

    p_high = u_high if primary_model_key == MODEL_UNWEIGHTED_KEY else s_high
    p_low = u_low if primary_model_key == MODEL_UNWEIGHTED_KEY else s_low

    summary = ScreeningBatchSummary(
        total_records=total_rows,
        accepted_records=num_valid,
        rejected_records=total_rows - num_valid,
        primary_model_key=primary_model_key,
        primary_model_name=MODEL_LABELS[primary_model_key],
        primary_threshold=current_thresholds[primary_model_key],
        high_risk_count=p_high,
        low_risk_count=p_low,
        unweighted_high_risk=u_high,
        unweighted_low_risk=u_low,
        smotenc_high_risk=s_high,
        smotenc_low_risk=s_low,
        agreement_count=agreements,
        disagreement_count=disagreements,
    )

    return ScreeningResult(
        full_table=full_table,
        accepted_table=accepted_table,
        rejected_table=rejected_table,
        summary=summary,
        validation=validation,
    )
