"""Reusable UI layout components following corporate design guidelines."""
from __future__ import annotations

from pathlib import Path
import streamlit as st

from config.settings import ASSETS_DIR


def inject_custom_css():
    """Inject corporate CSS stylesheet into the Streamlit app."""
    css_path = ASSETS_DIR / "style.css"
    if css_path.is_file():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge: str = "Space Situational Awareness"):
    """Render a clean enterprise page header banner."""
    st.markdown(
        f"""
        <div class="corp-header">
            <div class="corp-header-badge">{badge}</div>
            <h1 class="corp-header-title">{title}</h1>
            <p class="corp-header-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: str | int, detail: str, col=None):
    """Render a standardized KPI card."""
    card_html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-detail">{detail}</div>
    </div>
    """
    if col:
        with col:
            st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.markdown(card_html, unsafe_allow_html=True)


def render_operational_notice():
    """Render mandatory operational disclaimer."""
    st.markdown(
        """
        <div class="notice-box">
            <strong>Operational Screening Notice:</strong> This dashboard is a machine-learning-assisted 
            screening aid designed to prioritize high-risk conjunction events 2.0–2.9 days before TCA. 
            Predicted scores represent statistical ranking scores, not calibrated physical collision 
            probabilities. Expert orbital mechanics review remains mandatory for all operational maneuver decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )
