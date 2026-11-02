"""Risk Analysis View."""
from __future__ import annotations

import streamlit as st

from components.charts import (
    plot_miss_distance_scatter,
    plot_object_type_distribution,
    plot_risk_distribution_donut,
    plot_score_histogram,
)
from components.ui_layout import render_header
from config.settings import OUTPUT_COLS


def render_analysis_view(screening_result):
    """Render Analytical Visualizations and Spatial Risk Analysis."""
    render_header(
        title="Orbital Risk Diagnostics & Spatial Analytics",
        subtitle="Multivariate exploration of conjunction geometry, spatial proximity, and chaser characteristics.",
        badge="Advanced Diagnostics",
    )

    if screening_result is None or screening_result.summary.accepted_records == 0:
        st.info("No screened dataset available for risk analysis. Please upload or screen a dataset first.")
        return

    accepted_df = screening_result.accepted_table
    summary = screening_result.summary

    # Row 1: High level breakdown
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(
            plot_risk_distribution_donut(summary.high_risk_count, summary.low_risk_count),
            use_container_width=True,
        )
    with col2:
        scores = accepted_df[OUTPUT_COLS["primary_score"]].dropna().tolist()
        st.plotly_chart(
            plot_score_histogram(scores, summary.primary_threshold, summary.primary_model_name),
            use_container_width=True,
        )

    st.markdown("---")

    # Row 2: Spatial Proximity & Covariance
    st.markdown("### Spatial Proximity vs Statistical Mahalanobis Distance")
    st.caption("Low miss distance combined with low Mahalanobis distance indicates high conjunction probability density.")
    st.plotly_chart(plot_miss_distance_scatter(accepted_df), use_container_width=True)

    st.markdown("---")

    # Row 3: Chaser Object Class Breakdown
    st.markdown("### Chaser Object Type Vulnerability Analysis")
    st.plotly_chart(plot_object_type_distribution(accepted_df), use_container_width=True)

