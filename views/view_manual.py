"""Manual Screening View."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.ui_layout import render_header
from config.settings import (
    MODEL_SMOTENC_KEY,
    MODEL_UNWEIGHTED_KEY,
    OBJECT_TYPE_CATEGORIES,
    TCA_MAX_DAYS,
    TCA_MIN_DAYS,
)
from core.prediction_engine import evaluate_dual_models
from core.validation_engine import validate_cdm_dataset


def render_manual_view(models, thresholds):
    """Render Manual CDM Event Screening Form."""
    render_header(
        title="Manual Conjunction Event Screening",
        subtitle="Individual CDM parameter entry with instant 7-feature validation and dual-model evaluation.",
        badge="Single Event Screening",
    )

    col_form, col_context = st.columns([2, 1])

    with col_context:
        st.markdown("#### Guidance & Horizons")
        st.markdown(
            f"""
            <div style="font-size: 0.85rem; line-height: 1.5; color: #475569; background: #FFFFFF; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 6px;">
                <strong>Screening Parameters:</strong><br>
                • <code>time_to_tca</code>: Must be in [{TCA_MIN_DAYS:.1f}, {TCA_MAX_DAYS:.1f}] days.<br>
                • <code>max_risk_estimate</code>: Log10 collision risk metric.<br>
                • <code>max_risk_scaling</code>: Risk scale factor.<br>
                • <code>miss_distance</code>: Miss distance in meters.<br>
                • <code>mahalanobis_distance</code>: Covariance-scaled distance.<br>
                • <code>c_obs_used</code>: Number of chaser observations.<br>
                • <code>c_object_type</code>: Actual chaser classification.<br><br>
                <em>All 7 features are required by the ML pipeline.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_form:
        st.markdown("#### Enter Conjunction Parameters")
        with st.form("manual_screening_form", clear_on_submit=False):
            event_id = st.text_input("Event Identifier (Optional)", value="CDM-MANUAL-001")

            c1, c2 = st.columns(2)
            with c1:
                time_to_tca = st.number_input(
                    f"Time to TCA (days) * [{TCA_MIN_DAYS:.1f}–{TCA_MAX_DAYS:.1f}]",
                    min_value=0.0,
                    max_value=30.0,
                    value=2.45,
                    step=0.01,
                    format="%.4f",
                    help="Permitted screening window is 2.0 to 2.9 days.",
                )
                max_risk_estimate = st.number_input(
                    "Max Risk Estimate (log10) *",
                    value=-4.50,
                    step=0.1,
                    format="%.6f",
                    help="Maximum risk estimate value.",
                )
                max_risk_scaling = st.number_input(
                    "Max Risk Scaling *",
                    value=12.0,
                    step=0.5,
                    format="%.4f",
                )
                c_object_type = st.selectbox(
                    "Chaser Object Type (c_object_type) *",
                    options=OBJECT_TYPE_CATEGORIES,
                    index=0,
                    help="Trained categorical categories from pipeline.",
                )

            with c2:
                miss_distance = st.number_input(
                    "Miss Distance (meters) *",
                    min_value=0.0,
                    value=850.0,
                    step=50.0,
                    format="%.2f",
                )
                mahalanobis_distance = st.number_input(
                    "Mahalanobis Distance *",
                    min_value=0.0,
                    value=11.2,
                    step=0.5,
                    format="%.4f",
                )
                c_obs_used = st.number_input(
                    "Observations Used (c_obs_used) *",
                    min_value=1.0,
                    value=32.0,
                    step=1.0,
                    format="%.0f",
                )

            submit_btn = st.form_submit_button("Screen Conjunction Event", type="primary", use_container_width=True)

    if submit_btn:
        row_dict = {
            "event_id": event_id.strip() or "CDM-MANUAL-001",
            "time_to_tca": time_to_tca,
            "max_risk_estimate": max_risk_estimate,
            "max_risk_scaling": max_risk_scaling,
            "miss_distance": miss_distance,
            "mahalanobis_distance": mahalanobis_distance,
            "c_obs_used": c_obs_used,
            "c_object_type": c_object_type,
        }

        input_df = pd.DataFrame([row_dict])
        validation = validate_cdm_dataset(input_df)

        st.markdown("---")
        st.markdown("### Screening Evaluation Result")

        if not validation.valid_mask[0]:
            st.error(f"Event Rejected: {validation.rejection_reasons[0]}")
            return

        # Execute dual-model inference
        eval_result = evaluate_dual_models(
            models=models,
            clean_features=validation.clean_features,
            thresholds=thresholds,
        )
        st.session_state["manual_eval_data"] = eval_result

    if "manual_eval_data" in st.session_state:
        eval_result = st.session_state["manual_eval_data"]

        u_score = eval_result[f"{MODEL_UNWEIGHTED_KEY}_scores"][0]
        u_label = eval_result[f"{MODEL_UNWEIGHTED_KEY}_labels"][0]
        u_thresh = eval_result[f"{MODEL_UNWEIGHTED_KEY}_threshold"]

        s_score = eval_result[f"{MODEL_SMOTENC_KEY}_scores"][0]
        s_label = eval_result[f"{MODEL_SMOTENC_KEY}_labels"][0]
        s_thresh = eval_result[f"{MODEL_SMOTENC_KEY}_threshold"]

        agreement = eval_result["agreement_labels"][0]

        display_mode = st.radio(
            "Display Mode:",
            ["Dual-Model Comparison", "Unweighted RF Only", "SMOTENC RF (0.25) Only"],
            horizontal=True,
            key="manual_display_mode_choice",
        )

        u_color = "#DC2626" if u_label == "High risk" else "#16A34A"
        u_card_html = f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 4px solid {u_color}; border-radius: 6px; padding: 1.25rem;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">
                Random Forest — Unweighted
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {u_color}; margin: 0.4rem 0;">
                {u_label}
            </div>
            <div style="font-size: 0.9rem; color: #334155; line-height: 1.6;">
                <strong>Score:</strong> <code>{u_score:.4f}</code><br>
                <strong>Decision Threshold:</strong> <code>{u_thresh:.2f}</code><br>
                <strong>Evaluation:</strong> {'Score >= Threshold (High Risk)' if u_score >= u_thresh else 'Score < Threshold (Low Risk)'}
            </div>
        </div>
        """

        s_color = "#DC2626" if s_label == "High risk" else "#16A34A"
        s_card_html = f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 4px solid {s_color}; border-radius: 6px; padding: 1.25rem;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">
                Random Forest + SMOTENC (0.25)
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {s_color}; margin: 0.4rem 0;">
                {s_label}
            </div>
            <div style="font-size: 0.9rem; color: #334155; line-height: 1.6;">
                <strong>Score:</strong> <code>{s_score:.4f}</code><br>
                <strong>Decision Threshold:</strong> <code>{s_thresh:.2f}</code><br>
                <strong>Evaluation:</strong> {'Score >= Threshold (High Risk)' if s_score >= s_thresh else 'Score < Threshold (Low Risk)'}
            </div>
        </div>
        """

        if display_mode == "Dual-Model Comparison":
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(u_card_html, unsafe_allow_html=True)
            with res_col2:
                st.markdown(s_card_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if agreement == "Agreement":
                st.success(f"Concordant Prediction: Both models classify this event as '{u_label}'.")
            else:
                st.warning("Prediction Disagreement: Unweighted RF and SMOTENC RF produced conflicting risk classes. Operational review recommended.")

        elif display_mode == "Unweighted RF Only":
            st.markdown(u_card_html, unsafe_allow_html=True)

        elif display_mode == "SMOTENC RF (0.25) Only":
            st.markdown(s_card_html, unsafe_allow_html=True)

