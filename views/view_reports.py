"""Reports and Audit Export View."""
from __future__ import annotations

import streamlit as st

from components.ui_layout import render_header, render_kpi_card
from config.settings import OUTPUT_COLS
from services.export_service import (
    generate_executive_summary_text,
    generate_results_csv,
)


def render_reports_view(screening_result):
    """Render Reports and Audit Download Center."""
    render_header(
        title="Screening Reports & Compliance Export Center",
        subtitle="Export full audit trails retaining all rows in canonical source sequence with dual-model classifications.",
        badge="Audit & Reporting",
    )

    if screening_result is None:
        st.info("No active screening results available to export. Please ingest and screen a dataset first.")
        return

    summary = screening_result.summary
    full_table = screening_result.full_table

    c1, c2, c3 = st.columns(3)
    render_kpi_card("Total Exportable Rows", f"{summary.total_records:,}", "100% order preserved", c1)
    render_kpi_card("Screened Events", f"{summary.accepted_records:,}", "Valid inference generated", c2)
    render_kpi_card("Audit Logged Errors", f"{summary.rejected_records:,}", "Documented with reasons", c3)

    st.markdown("---")

    col_csv, col_txt = st.columns([1, 1])

    with col_csv:
        st.markdown("#### Complete CSV Audit Trail")
        st.markdown(
            """
            Includes:
            - Original row number (Source Row)
            - Event ID (display)
            - Validation status & specific rejection reason
            - Unweighted RF score, threshold, and classification
            - SMOTENC RF score, threshold, and classification
            - Model disagreement status
            - All original input columns (unaltered)
            """
        )
        csv_bytes = generate_results_csv(full_table)
        st.download_button(
            label="Download Complete Screening Results (CSV)",
            data=csv_bytes,
            file_name="satellite_conjunction_screening_audit.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    with col_txt:
        st.markdown("#### Executive Summary Text Report")
        st.markdown(
            """
            Summary file suitable for operational flight logbooks, mission handover, 
            or space situational awareness shift reports.
            """
        )
        exec_text = generate_executive_summary_text(summary)
        st.download_button(
            label="Download Executive Summary Report (TXT)",
            data=exec_text.encode("utf-8"),
            file_name="conjunction_screening_executive_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Executive Summary Preview")
    st.text(exec_text)

    st.markdown("---")
    st.markdown("### Complete Audit Table Preview")
    st.caption("Verifying 100% row order preservation and error transparency.")
    st.dataframe(full_table, use_container_width=True, hide_index=True)
