"""Table display and interactive filter components."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import OUTPUT_COLS


def render_interactive_results_table(results_df: pd.DataFrame, key_prefix: str = "main"):
    """Render a searchable, filterable data table with clean visual presentation."""
    if results_df.empty:
        st.info("No records to display.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        status_filter = st.selectbox(
            "Filter records",
            options=["All Records", "High Risk Only", "Low Risk Only", "Rejected Only", "Model Disagreements"],
            key=f"{key_prefix}_filter",
        )
    with col2:
        search_query = st.text_input(
            "Search by Event ID",
            placeholder="Type event ID or identifier...",
            key=f"{key_prefix}_search",
        )

    filtered = results_df.copy()

    # Apply category filter
    if status_filter == "High Risk Only":
        filtered = filtered[filtered[OUTPUT_COLS["primary_class"]] == "High risk"]
    elif status_filter == "Low Risk Only":
        filtered = filtered[filtered[OUTPUT_COLS["primary_class"]] == "Low risk"]
    elif status_filter == "Rejected Only":
        filtered = filtered[filtered[OUTPUT_COLS["validation_status"]] == "Rejected"]
    elif status_filter == "Model Disagreements":
        filtered = filtered[filtered[OUTPUT_COLS["model_agreement"]] == "Disagreement (Review Required)"]

    # Apply search filter
    if search_query.strip():
        q = search_query.strip().lower()
        mask = filtered[OUTPUT_COLS["event_id"]].astype(str).str.lower().str.contains(q, na=False)
        filtered = filtered[mask]

    st.caption(f"Showing {len(filtered):,} of {len(results_df):,} total records")

    # Select standard view columns
    view_cols = [
        OUTPUT_COLS["source_row"],
        OUTPUT_COLS["event_id"],
        OUTPUT_COLS["validation_status"],
        OUTPUT_COLS["primary_score"],
        OUTPUT_COLS["primary_class"],
        OUTPUT_COLS["unweighted_score"],
        OUTPUT_COLS["smotenc_score"],
        OUTPUT_COLS["model_agreement"],
        OUTPUT_COLS["rejection_reason"],
    ]

    # Include original features if available in the table
    extra_features = ["time_to_tca", "miss_distance", "mahalanobis_distance", "c_object_type"]
    for feat in extra_features:
        if feat in filtered.columns and feat not in view_cols:
            view_cols.append(feat)

    # Only include columns present in filtered
    final_cols = [c for c in view_cols if c in filtered.columns]

    st.dataframe(
        filtered[final_cols],
        use_container_width=True,
        hide_index=True,
    )

