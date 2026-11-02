"""Upload CDM Dataset View."""
from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import streamlit as st

from components.data_tables import render_interactive_results_table
from components.ui_layout import render_header, render_kpi_card
from config.settings import FEATURES, OUTPUT_COLS
from services.export_service import generate_cdm_template_csv


def parse_file(uploaded_file) -> pd.DataFrame:
    """Parse uploaded CSV or XLSX file securely."""
    suffix = Path(uploaded_file.name).suffix.lower()
    content = uploaded_file.getvalue()

    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=True)
    elif suffix == ".xlsx":
        df = pd.read_excel(io.BytesIO(content), dtype=object)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Please upload a CSV (.csv) or Excel (.xlsx) file.")

    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names detected: {', '.join(map(str, dups))}")

    if df.empty:
        raise ValueError("The uploaded file contains no data rows.")

    return df


def render_upload_view(run_screening_fn, current_screening_result):
    """Render the Dataset Upload and Batch Screening page."""
    render_header(
        title="Upload Conjunction Data Message (CDM) Dataset",
        subtitle="Batch ingestion of CDMs in CSV or Excel format with 7-feature validation and 2.0–2.9 day TCA gating.",
        badge="Batch Ingestion Pipeline",
    )

    col_upload, col_template = st.columns([2, 1])

    with col_template:
        st.markdown("#### CDM Ingestion Schema")
        st.markdown(
            f"""
            <div style="font-size: 0.85rem; line-height: 1.5; color: #475569; background: #FFFFFF; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 6px;">
                <strong>Required Fields (7):</strong><br>
                1. <code>time_to_tca</code> (2.0 to 2.9 days)<br>
                2. <code>max_risk_estimate</code> (float)<br>
                3. <code>max_risk_scaling</code> (float)<br>
                4. <code>miss_distance</code> (meters)<br>
                5. <code>mahalanobis_distance</code> (float)<br>
                6. <code>c_obs_used</code> (float / count)<br>
                7. <code>c_object_type</code> (DEBRIS, PAYLOAD, etc.)<br>
                <em>Additional columns are preserved for audit.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )
        template_bytes = generate_cdm_template_csv()
        st.download_button(
            label="Download CSV Template",
            data=template_bytes,
            file_name="cdm_screening_template.csv",
            mime="text/csv",
            help="Download a clean template with the canonical 7-feature schema.",
        )

    with col_upload:
        st.markdown("#### Upload Dataset File")
        uploaded_file = st.file_uploader(
            "Choose a CSV or XLSX file",
            type=["csv", "xlsx"],
            help="Files up to 50MB supported.",
            key="cdm_file_uploader",
        )

        sample_btn = st.button("Load Pre-Packaged Demo Dataset (Mixed Valid/Invalid)")
        if sample_btn:
            demo_path = Path("data/sample_cdm_mixed.csv")
            if demo_path.is_file():
                raw_df = pd.read_csv(demo_path)
                st.session_state["raw_dataset"] = raw_df
                st.session_state["dataset_name"] = "sample_cdm_mixed.csv"
                st.success(f"Loaded demo dataset with {len(raw_df)} records.")

    if uploaded_file is not None:
        try:
            raw_df = parse_file(uploaded_file)
            st.session_state["raw_dataset"] = raw_df
            st.session_state["dataset_name"] = uploaded_file.name
            st.success(f"Successfully loaded '{uploaded_file.name}' ({len(raw_df):,} rows, {len(raw_df.columns)} columns).")
        except Exception as e:
            st.error(f"File ingestion error: {e}")
            return

    if "raw_dataset" in st.session_state:
        df_to_screen = st.session_state["raw_dataset"]

        st.markdown("---")
        st.markdown(f"### Ingested Data Preview (`{st.session_state.get('dataset_name', 'Uploaded File')}`)")
        st.caption(f"Showing first 5 rows of {len(df_to_screen)} total records. Select 'Execute Screening Pipeline' to validate and classify.")
        st.dataframe(df_to_screen.head(5), use_container_width=True, hide_index=True)

        if st.button("Execute Screening Pipeline", type="primary", use_container_width=True):
            with st.spinner("Executing 7-feature validation and dual-model inference..."):
                screening_result = run_screening_fn(df_to_screen)
                st.session_state["current_screening_result"] = screening_result
                st.success("Screening pipeline completed successfully!")

        if current_screening_result is not None:
            st.markdown("---")
            st.markdown("### Batch Screening Results")
            summary = current_screening_result.summary

            c1, c2, c3, c4 = st.columns(4)
            render_kpi_card("Total Ingested", f"{summary.total_records:,}", "100% retained", c1)
            render_kpi_card("Accepted for Inference", f"{summary.accepted_records:,}", f"{summary.acceptance_rate:.1%} valid", c2)
            render_kpi_card("Rejected (Invalid)", f"{summary.rejected_records:,}", "Specific errors logged", c3)
            render_kpi_card("High Risk Identified", f"{summary.high_risk_count:,}", f"Under threshold {summary.primary_threshold:.2f}", c4)

            render_interactive_results_table(current_screening_result.full_table, key_prefix="upload_results")
