"""Dashboard Overview View."""
from __future__ import annotations

import streamlit as st

from components.charts import (
    plot_model_comparison_bar,
    plot_risk_distribution_donut,
    plot_score_histogram,
)
from components.data_tables import render_interactive_results_table
from components.ui_layout import (
    render_header,
    render_kpi_card,
    render_operational_notice,
)
from config.settings import OUTPUT_COLS


def render_overview_view(screening_result):
    """Render Executive Dashboard Overview."""
    render_header(
        title="Orbital Conjunction Risk Screening Console",
        subtitle="Real-time screening of satellite close-approach events 2.0–2.9 days before TCA using trained Random Forest pipelines.",
        badge="Enterprise Flight Operations",
    )

    render_operational_notice()

    if screening_result is None:
        st.info("No CDM data currently loaded in the session. Upload a dataset or load the sample screening batch below.")
        if st.button("Load Standard Sample Dataset", type="primary"):
            st.session_state["use_sample_data"] = True
            st.rerun()
        return

    summary = screening_result.summary
    full_table = screening_result.full_table

    # Row 1: Primary KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    render_kpi_card("Total Processed", f"{summary.total_records:,}", "All records retained", col1)
    render_kpi_card("Accepted (Screened)", f"{summary.accepted_records:,}", f"{summary.acceptance_rate:.1%} valid", col2)
    render_kpi_card("Rejected", f"{summary.rejected_records:,}", "Input validation errors", col3)
    render_kpi_card("High Risk Flagged", f"{summary.high_risk_count:,}", f"{summary.high_risk_rate:.1%} flagged", col4)
    render_kpi_card("Low Risk Flagged", f"{summary.low_risk_count:,}", "Under threshold", col5)
    render_kpi_card("Model Disagreements", f"{summary.disagreement_count:,}", "Requires review", col6)

    st.markdown("---")

    # Row 2: Analytics & Visualizations
    st.markdown("### Operational Risk Distribution & Model Concordance")
    chart_col1, chart_col2, chart_col3 = st.columns([1, 1, 1])

    with chart_col1:
        st.plotly_chart(
            plot_risk_distribution_donut(summary.high_risk_count, summary.low_risk_count),
            use_container_width=True,
        )

    with chart_col2:
        st.plotly_chart(
            plot_model_comparison_bar(summary),
            use_container_width=True,
        )

    with chart_col3:
        scores = full_table[OUTPUT_COLS["primary_score"]].dropna().tolist()
        st.plotly_chart(
            plot_score_histogram(scores, summary.primary_threshold, summary.primary_model_name),
            use_container_width=True,
        )

    st.markdown("---")

    # Row 3: Conjunction Events Ledger
    st.markdown("### Conjunction Events Ledger")
    render_interactive_results_table(full_table, key_prefix="overview_ledger")
