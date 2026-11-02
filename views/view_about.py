"""About and Methodology View."""
from __future__ import annotations

from pathlib import Path
import streamlit as st

from components.ui_layout import render_header
from config.settings import (
    DEFAULT_THRESHOLDS,
    FEATURES,
    MODEL_LABELS,
    MODEL_SMOTENC_KEY,
    MODEL_UNWEIGHTED_KEY,
    MODELS_DIR,
    TCA_MAX_DAYS,
    TCA_MIN_DAYS,
)
from core.model_loader import compute_file_sha256


def render_about_view():
    """Render Documentation and Methodology View."""
    render_header(
        title="About and Scientific Methodology",
        subtitle="System architecture, conjunction assessment standards, ML model details, and operational constraints.",
        badge="Technical Documentation",
    )

    st.markdown(
        f"""
        ### 1. Operational Problem Formulation
        In modern low-Earth orbit (LEO) and geostationary orbit (GEO), thousands of Conjunction Data Messages (CDMs) 
        are generated daily by space surveillance networks (such as US Space Command and ESA's Space Debris Office). 
        Flight dynamics operators must filter out millions of benign encounters to identify genuine high-risk events 
        requiring collision avoidance maneuvers (CAM).

        This dashboard implements an **early-horizon conjunction screening system** focusing on the critical window 
        approximately **{TCA_MIN_DAYS:.1f} to {TCA_MAX_DAYS:.1f} days before Time of Closest Approach (TCA)**. 
        This window represents the operational threshold where orbit determination uncertainty is sufficiently reduced, 
        yet adequate lead time remains to plan, calculate, simulate, and uplink avoidance thruster burns.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### 2. The Seven Canonical Model Features
        Both machine learning models utilize strictly the following seven features in fixed order:

        | # | Feature Identifier | Units / Format | Physical Description |
        |---|---|---|---|
        | 1 | `time_to_tca` | Days (`float`) | Lead time remaining until closest orbital approach (valid: 2.0–2.9 days). |
        | 2 | `max_risk_estimate` | Log10 metric | Worst-case collision risk computed under worst-case covariance orientation. |
        | 3 | `max_risk_scaling` | Scale factor | Scaling parameter for the combined covariance matrix in maximum risk calculation. |
        | 4 | `miss_distance` | Meters | Euclidean separation distance between target and chaser at TCA. |
        | 5 | `mahalanobis_distance` | Normalized sigma | Statistical distance in sigma units weighted by the relative position covariance. |
        | 6 | `c_obs_used` | Integer count | Number of radar/optical tracking observations utilized in chaser orbit solution. |
        | 7 | `c_object_type` | Categorical | Object classification: `DEBRIS`, `PAYLOAD`, `ROCKET BODY`, `TBA`, `UNKNOWN`. |

        > **Excluded Fields:** Metadata and potential target leakage fields such as `event_id`, `mission_id`, `risk`, `final_risk`, and `final_risk_class` are strictly excluded from model inference and retained solely for audit reporting.
        """
    )

    st.markdown("---")

    # Display real SHA256 checksums from disk
    m1_path = MODELS_DIR / "rf_unweighted.joblib"
    m2_path = MODELS_DIR / "rf_smotenc_025.joblib"
    m1_hash = compute_file_sha256(m1_path) if m1_path.is_file() else "FILE_NOT_FOUND"
    m2_hash = compute_file_sha256(m2_path) if m2_path.is_file() else "FILE_NOT_FOUND"

    st.markdown(
        f"""
        ### 3. Machine Learning Model Artifacts & Integrity
        The dashboard utilizes two original trained Random Forest classification pipelines saved as `.joblib` artifacts:

        #### Model A: Random Forest (Unweighted)
        - **File:** `models/rf_unweighted.joblib`
        - **SHA-256 Checksum:** `{m1_hash}`
        - **Pipeline Architecture:** Robust scaling, log1p transformation, median imputation, ordinal & one-hot encoding, 300-tree Random Forest Classifier.
        - **Configured Threshold:** `{DEFAULT_THRESHOLDS[MODEL_UNWEIGHTED_KEY]:.2f}`
        - **Characteristic:** High precision on natural imbalanced distributions.

        #### Model B: Random Forest + SMOTENC (0.25)
        - **File:** `models/rf_smotenc_025.joblib`
        - **SHA-256 Checksum:** `{m2_hash}`
        - **Pipeline Architecture:** Combined preprocessing with Synthetic Minority Over-sampling Technique for Nominal and Continuous features (SMOTENC) applied during training.
        - **Configured Threshold:** `{DEFAULT_THRESHOLDS[MODEL_SMOTENC_KEY]:.2f}`
        - **Characteristic:** Heightened sensitivity / recall for rare, high-consequence encounters.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### 4. Regulatory & Operational Disclaimers
        - **Screening Aid Only:** This system is an automated triage tool. It does not replace flight dynamics engineers or certified astrodynamics software (such as ESA DRAMA, NASA CARA, or Orekit).
        - **Not Calibrated Probabilities:** The model scores represent classification ranking scores between 0.0 and 1.0; they are **not** calibrated physical collision probabilities (Pc).
        - **TCA Validation Gating:** Encounters with TCA outside the 2.0–2.9 day window are rejected by design because the model was calibrated specifically for early-horizon operations.
        """
    )
