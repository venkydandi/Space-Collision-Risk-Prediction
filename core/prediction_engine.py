"""Inference and model prediction engine."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple
import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_THRESHOLDS,
    FEATURES,
    LABEL_LOW_RISK,
    LABEL_MAPPING,
    MODEL_SMOTENC_KEY,
    MODEL_UNWEIGHTED_KEY,
)


def extract_positive_class_probabilities(model: Any, features_df: pd.DataFrame) -> np.ndarray:
    """
    Execute pipeline predict_proba and extract the positive class (class 1) probabilities.
    
    The pipeline handles its own preprocessing (imputation, scaling, encoding).
    """
    if features_df.empty:
        return np.array([], dtype=float)

    # Strictly slice the exact 7 features in order
    X = features_df[FEATURES]

    raw_probs = model.predict_proba(X)
    probs = np.asarray(raw_probs)

    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = next((i for i, c in enumerate(classes) if str(c) == "1"), 1)

    if pos_idx >= probs.shape[1]:
        raise ValueError(f"Model predict_proba output has unexpected shape {probs.shape} for classes {classes}")

    scores = probs[:, pos_idx].astype(float)
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Model returned non-finite or out-of-bounds probability scores.")

    return scores


def classify_scores(scores: np.ndarray, threshold: float) -> Tuple[np.ndarray, List[str]]:
    """
    Classify probability scores against a decision threshold.
    
    Score >= threshold -> 1 (High risk)
    Score < threshold  -> 0 (Low risk)
    """
    classes = (scores >= float(threshold)).astype(int)
    labels = [LABEL_MAPPING.get(str(c), LABEL_LOW_RISK) for c in classes]
    return classes, labels


def evaluate_dual_models(
    models: Mapping[str, Any],
    clean_features: pd.DataFrame,
    thresholds: Mapping[str, float],
) -> Dict[str, Any]:
    """
    Execute inference for both models and calculate agreement.
    
    Returns a dictionary with scores, classes, labels, and agreement flags.
    """
    results: Dict[str, Any] = {}

    for model_key in [MODEL_UNWEIGHTED_KEY, MODEL_SMOTENC_KEY]:
        model = models[model_key]
        threshold = thresholds.get(model_key, DEFAULT_THRESHOLDS[model_key])
        
        scores = extract_positive_class_probabilities(model, clean_features)
        classes, labels = classify_scores(scores, threshold)
        
        results[f"{model_key}_scores"] = scores
        results[f"{model_key}_classes"] = classes
        results[f"{model_key}_labels"] = labels
        results[f"{model_key}_threshold"] = threshold

    # Compute agreement
    u_classes = results[f"{MODEL_UNWEIGHTED_KEY}_classes"]
    s_classes = results[f"{MODEL_SMOTENC_KEY}_classes"]
    
    agreements: List[str] = []
    is_disagreement: List[bool] = []
    
    for u, s in zip(u_classes, s_classes):
        if u == s:
            agreements.append("Agreement")
            is_disagreement.append(False)
        else:
            agreements.append("Disagreement (Review Required)")
            is_disagreement.append(True)

    results["agreement_labels"] = agreements
    results["is_disagreement"] = is_disagreement

    return results

