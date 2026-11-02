"""Configuration constants for the Satellite Conjunction Risk Screening Dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Exact seven trained features in canonical order
FEATURES: List[str] = [
    "time_to_tca",
    "max_risk_estimate",
    "max_risk_scaling",
    "miss_distance",
    "mahalanobis_distance",
    "c_obs_used",
    "c_object_type",
]

# Numeric subset of features
NUMERIC_FEATURES: List[str] = FEATURES[:-1]

# Valid categorical values from the actual fitted pipeline
OBJECT_TYPE_CATEGORIES: List[str] = [
    "DEBRIS",
    "PAYLOAD",
    "ROCKET BODY",
    "TBA",
    "UNKNOWN",
]

# Model Identifiers and User-Friendly Labels
MODEL_UNWEIGHTED_KEY = "RF_unweighted"
MODEL_SMOTENC_KEY = "RF_SMOTENC_0.25"

MODEL_LABELS: Dict[str, str] = {
    MODEL_UNWEIGHTED_KEY: "Random Forest — Unweighted",
    MODEL_SMOTENC_KEY: "Random Forest + SMOTENC (0.25)",
}

# Default Decision Thresholds
DEFAULT_THRESHOLDS: Dict[str, float] = {
    MODEL_UNWEIGHTED_KEY: 0.10,
    MODEL_SMOTENC_KEY: 0.20,
}

# Threshold configuration range
THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.50
THRESHOLD_STEP = 0.01

# Operational Time to Closest Approach (TCA) screening window in days
TCA_MIN_DAYS: float = 2.0
TCA_MAX_DAYS: float = 2.9

# Fields forbidden from being used as model features
EXCLUDED_MODEL_FIELDS: List[str] = [
    "event_id",
    "mission_id",
    "risk",
    "final_risk",
    "final_risk_class",
]

# Class Labels
LABEL_LOW_RISK = "Low risk"
LABEL_HIGH_RISK = "High risk"

LABEL_MAPPING: Dict[str, str] = {
    "0": LABEL_LOW_RISK,
    "1": LABEL_HIGH_RISK,
}

# Standard Output Column Names
OUTPUT_COLS = {
    "source_row": "Source Row",
    "event_id": "Event ID",
    "validation_status": "Validation Status",
    "rejection_reason": "Rejection Reason",
    "unweighted_score": "RF Unweighted Score",
    "unweighted_class": "RF Unweighted Class",
    "unweighted_threshold": "RF Unweighted Threshold",
    "smotenc_score": "RF SMOTENC Score",
    "smotenc_class": "RF SMOTENC Class",
    "smotenc_threshold": "RF SMOTENC Threshold",
    "model_agreement": "Model Agreement",
    "primary_score": "Primary Score",
    "primary_class": "Primary Classification",
    "primary_model": "Primary Model",
}

