"""Model loading and compatibility management module."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

# Step 1: Ensure scikit-learn cross-version unpickling compatibility.
# In scikit-learn >= 1.7, the internal helper _RemainderColsList was refactored.
# For pipelines saved with scikit-learn 1.6.1, we provide a compatibility shim.
try:
    import sklearn.compose._column_transformer as _ct

    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Compatibility stub for unpickling scikit-learn <= 1.6 ColumnTransformers."""
            pass

        _ct._RemainderColsList = _RemainderColsList
except ImportError:
    pass

import joblib
from config.settings import (
    FEATURES,
    MODEL_LABELS,
    MODEL_SMOTENC_KEY,
    MODEL_UNWEIGHTED_KEY,
    MODELS_DIR,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


def compute_file_sha256(filepath: Path) -> str:
    """Calculate the SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest().upper()


def find_model_dir() -> Path:
    """Find the directory containing model artifacts."""
    candidates = [
        MODELS_DIR,
        PROJECT_ROOT / "models",
        PROJECT_ROOT,
        Path.cwd() / "models",
        Path.cwd(),
    ]
    for directory in candidates:
        if (directory / "metadata.json").is_file():
            return directory
    raise FileNotFoundError("Could not locate metadata.json in models directory or project root.")


def load_model_bundle(model_dir: Path | str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load both supplied fitted pipelines and metadata.
    
    Returns:
        tuple of (models_dict, metadata_dict)
    """
    resolved_dir = Path(model_dir) if model_dir is not None else find_model_dir()
    metadata_file = resolved_dir / "metadata.json"
    
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Missing metadata.json at: {metadata_file}")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    loaded_features = metadata.get("features", [])
    if loaded_features != FEATURES:
        raise ValueError(
            f"Metadata feature list mismatch. Expected {FEATURES}, got {loaded_features}"
        )
        
    model_paths = {
        MODEL_UNWEIGHTED_KEY: resolved_dir / metadata.get("models", {}).get(MODEL_UNWEIGHTED_KEY, "rf_unweighted.joblib"),
        MODEL_SMOTENC_KEY: resolved_dir / metadata.get("models", {}).get(MODEL_SMOTENC_KEY, "rf_smotenc_025.joblib"),
    }
    
    models: Dict[str, Any] = {}
    for key, path in model_paths.items():
        if not path.is_file():
            # Fallback to root or models directory if path is not found directly
            fallback = resolved_dir / path.name
            if fallback.is_file():
                path = fallback
            else:
                raise FileNotFoundError(f"Model artifact not found for {MODEL_LABELS.get(key, key)} at {path}")
        
        logger.info("Loading model artifact: %s (SHA256: %s)", path.name, compute_file_sha256(path))
        estimator = joblib.load(path)
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(f"Loaded object from {path.name} does not implement predict_proba()")
        models[key] = estimator

    return models, metadata


# Streamlit-cached loader
def get_cached_model_bundle(model_dir: Path | str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Cached loader for Streamlit applications."""
    try:
        import streamlit as st

        @st.cache_resource(show_spinner="Loading trained machine learning models...")
        def _cached_loader(m_dir: str):
            return load_model_bundle(m_dir)

        target_dir = str(model_dir if model_dir else find_model_dir())
        return _cached_loader(target_dir)
    except Exception:
        # Fallback to direct load when not running in Streamlit runtime
        return load_model_bundle(model_dir)

