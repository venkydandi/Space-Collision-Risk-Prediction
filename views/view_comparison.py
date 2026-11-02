"""Model Comparison View."""
from __future__ import annotations

import streamlit as st

from components.charts import plot_model_comparison_bar
from components.ui_layout import render_header, render_kpi_card
from config.settings import OUTPUT_COLS


def render_comparison_view(screening_result):
    """Render Dual Model Comparison and Disagreement Analysis View."""
    render_header(
        title="Comparative Machine Learning Model Analysis",
        subtitle="Side-by-side evaluation of Unweighted Random Forest vs SMOTENC (0.25) Random Forest.",
        badge="Dual Model Verification",
    )

    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 1.25rem; border-radius: 6px; margin-bottom: 1.5rem;">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; color: #0F172A;">Model Architectural Distinctions</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; font-size: 0.88rem; color: #334155; line-height: 1.5;">
                <div>
                    <strong style="color: #1D4ED8;">1. Random Forest — Unweighted (Default Threshold: 0.10)</strong><br>
                    Trained on raw historical CDM distributions without synthetic balancing. 
                    Maintains high precision and low false-alarm rate, prioritizing reliable high-confidence flags.
                </div>
                <div>
                    <strong style="color: #1D4ED8;">2. Random Forest + SMOTENC 0.25 (Default Threshold: 0.20)</strong><br>
                    Trained with Synthetic Minority Over-sampling Technique for Nominal and Continuous features (SMOTENC) at a 0.25 ratio. 
                    Heightened sensitivity / recall towards rare high-risk conjunction events.
                </div>
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.82rem; color: #64748B;">
                <em>Important Note: Scores are evaluated independently using their respective decision thresholds. Scores are never averaged or combined into unverified hybrid figures.</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if screening_result is None or screening_result.summary.accepted_records == 0:
        st.info("No screened dataset available for comparison. Please upload a dataset or execute a screening batch first.")
        return

    summary = screening_result.summary
    full_table = screening_result.full_table

    c1, c2, c3, c4 = st.columns(4)
    render_kpi_card("Unweighted RF Flagged", f"{summary.unweighted_high_risk:,}", f"Threshold {summary.unweighted_high_risk / summary.accepted_records:.1%}", c1)
    render_kpi_card("SMOTENC RF Flagged", f"{summary.smotenc_high_risk:,}", f"Threshold {summary.smotenc_high_risk / summary.accepted_records:.1%}", c2)
    render_kpi_card("Concordant Agreements", f"{summary.agreement_count:,}", f"Both models match ({summary.agreement_count / summary.accepted_records:.1%})", c3)
    render_kpi_card("Model Disagreements", f"{summary.disagreement_count:,}", f"Actionable review cases ({summary.disagreement_rate:.1%})", c4)

    st.markdown("---")

    chart_col, desc_col = st.columns([1.5, 1])
    with chart_col:
        st.plotly_chart(plot_model_comparison_bar(summary), use_container_width=True)
    with desc_col:
        st.markdown("#### Disagreement Resolution Protocol")
        st.markdown(
            """
            When models disagree on an event:
            1. **Examine TCA:** Verify observation count (`c_obs_used`) and miss distance.
            2. **Conservative Default:** If SMOTENC flags High Risk while Unweighted flags Low Risk, the event warrants heightened orbital tracking.
            3. **Operational Rule:** Disagreements never automatically confirm nor dismiss collision danger—they flag marginal events near decision boundaries.
            """
        )

    st.markdown("---")
    st.markdown("### Model Agreement & Disagreement Ledger")

    # Filter selector specifically for agreement states
    agree_filter = st.radio(
        "Show events by agreement status:",
        ["All Screened Events", "Disagreements Only (Review Required)", "Concordant Agreements Only"],
        horizontal=True,
    )

    comparison_df = screening_result.accepted_table.copy()
    if agree_filter == "Disagreements Only (Review Required)":
        comparison_df = comparison_df[comparison_df[OUTPUT_COLS["model_agreement"]] == "Disagreement (Review Required)"]
    elif agree_filter == "Concordant Agreements Only":
        comparison_df = comparison_df[comparison_df[OUTPUT_COLS["model_agreement"]] == "Agreement"]

    st.caption(f"Displaying {len(comparison_df)} records")

    cols_to_show = [
        OUTPUT_COLS["source_row"],
        OUTPUT_COLS["event_id"],
        OUTPUT_COLS["unweighted_score"],
        OUTPUT_COLS["unweighted_class"],
        OUTPUT_COLS["unweighted_threshold"],
        OUTPUT_COLS["smotenc_score"],
        OUTPUT_COLS["smotenc_class"],
        OUTPUT_COLS["smotenc_threshold"],
        OUTPUT_COLS["model_agreement"],
    ]
    for feat in ["time_to_tca", "miss_distance", "mahalanobis_distance", "c_object_type"]:
        if feat in comparison_df.columns:
            cols_to_show.append(feat)

    st.dataframe(comparison_df[cols_to_show], use_container_width=True, hide_index=True)

