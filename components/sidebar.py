"""Sidebar navigation and model parameter controls."""
from __future__ import annotations

from typing import Dict, Tuple
import streamlit as st

from config.settings import (
    DEFAULT_THRESHOLDS,
    MODEL_LABELS,
    MODEL_SMOTENC_KEY,
    MODEL_UNWEIGHTED_KEY,
    TCA_MAX_DAYS,
    TCA_MIN_DAYS,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
)

PAGES = [
    "Dashboard Overview",
    "Upload CDM Dataset",
    "Manual Screening",
    "Model Comparison",
    "Risk Analysis",
    "Reports",
    "About and Methodology",
]


def render_sidebar() -> Tuple[str, str, Dict[str, float]]:
    """
    Render enterprise sidebar and return:
    (selected_page, selected_primary_model, thresholds_dict)
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.1em; text-transform: uppercase;">
                    ORBITAL SAFETY CONSOLE
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; line-height: 1.3;">
                    Conjunction Risk Screening
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Navigation")
        selected_page = st.radio(
            "Select View",
            options=PAGES,
            index=0,
            label_visibility="collapsed",
            key="nav_page_selection",
        )

        st.divider()

        st.markdown("### Primary Model")
        primary_model = st.selectbox(
            "Active Screener",
            options=[MODEL_UNWEIGHTED_KEY, MODEL_SMOTENC_KEY],
            format_func=lambda k: MODEL_LABELS[k],
            index=0,
            key="primary_model_select",
            help="Choose the model used for primary screening classification.",
        )

        st.markdown("### Decision Thresholds")
        st.caption("Configured independently for each model.")

        thresh_unweighted = st.slider(
            "RF Unweighted Threshold",
            min_value=THRESHOLD_MIN,
            max_value=THRESHOLD_MAX,
            value=float(DEFAULT_THRESHOLDS[MODEL_UNWEIGHTED_KEY]),
            step=THRESHOLD_STEP,
            format="%.2f",
            key="thresh_unweighted_slider",
            help=f"Default: {DEFAULT_THRESHOLDS[MODEL_UNWEIGHTED_KEY]:.2f}. Score >= threshold is classified as High Risk (1).",
        )

        thresh_smotenc = st.slider(
            "RF SMOTENC (0.25) Threshold",
            min_value=THRESHOLD_MIN,
            max_value=THRESHOLD_MAX,
            value=float(DEFAULT_THRESHOLDS[MODEL_SMOTENC_KEY]),
            step=THRESHOLD_STEP,
            format="%.2f",
            key="thresh_smotenc_slider",
            help=f"Default: {DEFAULT_THRESHOLDS[MODEL_SMOTENC_KEY]:.2f}. Adjusted for synthetic oversampling distribution.",
        )

        thresholds = {
            MODEL_UNWEIGHTED_KEY: thresh_unweighted,
            MODEL_SMOTENC_KEY: thresh_smotenc,
        }

        st.divider()

        st.markdown("### Operational Window")
        st.markdown(
            f"""
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.4;">
                <strong>TCA Range:</strong> {TCA_MIN_DAYS:.1f} to {TCA_MAX_DAYS:.1f} days<br>
                <strong>Required Features:</strong> 7 fields<br>
                <strong>Status:</strong> <span style="color: #4ADE80; font-weight: 600;">System Online</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page, primary_model, thresholds
