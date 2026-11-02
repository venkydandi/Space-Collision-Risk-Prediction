"""Satellite Conjunction Risk Screening Dashboard — Streamlit Main Application."""
from __future__ import annotations

import logging
from pathlib import Path
import warnings
import pandas as pd
import streamlit as st

# Suppress benign library warning noise
warnings.filterwarnings("ignore")

# Configure page layout and metadata
st.set_page_config(
    page_title="Satellite Conjunction Risk Screening Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.sidebar import render_sidebar
from components.ui_layout import inject_custom_css
from core.model_loader import get_cached_model_bundle
from services.screening_service import run_screening_pipeline
from views import (
    render_about_view,
    render_analysis_view,
    render_comparison_view,
    render_manual_view,
    render_overview_view,
    render_reports_view,
    render_upload_view,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cdm_dashboard")


def main():
    # Inject enterprise styling
    inject_custom_css()

    # Load machine learning models (with Streamlit resource caching)
    try:
        models, metadata = get_cached_model_bundle()
    except Exception as error:
        st.error(f"Failed to load machine learning models: {error}")
        st.info(
            "Please ensure 'models/rf_unweighted.joblib', 'models/rf_smotenc_025.joblib', "
            "and 'models/metadata.json' exist and match compatible scikit-learn / joblib versions."
        )
        st.stop()

    # Render sidebar controls and retrieve user parameters
    selected_page, primary_model_key, thresholds = render_sidebar()

    # Pre-load standard demo dataset on initial session visit if no dataset exists
    if "raw_dataset" not in st.session_state:
        sample_path = Path("data/sample_cdm_valid.csv")
        if sample_path.is_file():
            st.session_state["raw_dataset"] = pd.read_csv(sample_path)
            st.session_state["dataset_name"] = "sample_cdm_valid.csv (Demo)"

    # Check if a sample dataset should be reloaded
    if st.session_state.get("use_sample_data", False):
        sample_path = Path("data/sample_cdm_valid.csv")
        if sample_path.is_file():
            st.session_state["raw_dataset"] = pd.read_csv(sample_path)
            st.session_state["dataset_name"] = "sample_cdm_valid.csv (Demo)"
        st.session_state["use_sample_data"] = False

    # Function to run screening pipeline on any DataFrame
    def execute_screening(df: pd.DataFrame):
        return run_screening_pipeline(
            data=df,
            models=models,
            primary_model_key=primary_model_key,
            thresholds=thresholds,
        )

    # Check if active dataset needs re-screening (e.g. threshold or primary model changed)
    current_screening_result = None
    if "raw_dataset" in st.session_state and isinstance(st.session_state["raw_dataset"], pd.DataFrame):
        active_df = st.session_state["raw_dataset"]
        current_signature = (
            primary_model_key,
            thresholds.get(primary_model_key, 0.1),
            tuple(sorted(thresholds.items())),
            len(active_df),
        )
        if (
            "current_screening_result" not in st.session_state
            or st.session_state.get("last_signature") != current_signature
        ):
            st.session_state["current_screening_result"] = execute_screening(active_df)
            st.session_state["last_signature"] = current_signature

        current_screening_result = st.session_state["current_screening_result"]

    # Router logic
    if selected_page == "Dashboard Overview":
        render_overview_view(current_screening_result)
    elif selected_page == "Upload CDM Dataset":
        render_upload_view(execute_screening, current_screening_result)
    elif selected_page == "Manual Screening":
        render_manual_view(models, thresholds)
    elif selected_page == "Model Comparison":
        render_comparison_view(current_screening_result)
    elif selected_page == "Risk Analysis":
        render_analysis_view(current_screening_result)
    elif selected_page == "Reports":
        render_reports_view(current_screening_result)
    elif selected_page == "About and Methodology":
        render_about_view()
    else:
        render_overview_view(current_screening_result)


if __name__ == "__main__":
    main()

