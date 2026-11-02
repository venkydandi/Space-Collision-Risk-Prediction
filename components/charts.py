"""Enterprise Plotly visualization components."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import OUTPUT_COLS


def plot_risk_distribution_donut(high_count: int, low_count: int) -> go.Figure:
    """Create a clean donut chart displaying high vs low risk proportion."""
    labels = ["High Risk", "Low Risk"]
    values = [high_count, low_count]
    colors = ["#DC2626", "#16A34A"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.65,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            )
        ]
    )

    fig.update_layout(
        title=dict(text="Risk Classification Breakdown", font=dict(size=14, color="#0F172A")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_score_histogram(scores: list[float], threshold: float, model_name: str) -> go.Figure:
    """Plot score distribution histogram with decision threshold vertical line."""
    clean_scores = [s for s in scores if pd.notna(s)]
    if not clean_scores:
        fig = go.Figure()
        fig.add_annotation(text="No score data available", showarrow=False)
        fig.update_layout(height=280)
        return fig

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=clean_scores,
            nbinsx=25,
            marker=dict(color="#1D4ED8", line=dict(color="#FFFFFF", width=1)),
            name="Score Frequency",
        )
    )

    fig.add_vline(
        x=threshold,
        line_width=2.5,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="top right",
        annotation_font=dict(color="#DC2626", size=11),
    )

    fig.update_layout(
        title=dict(text=f"Score Distribution — {model_name}", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Model Positive Class Score", range=[0, 1], gridcolor="#F1F5F9"),
        yaxis=dict(title="Event Count", gridcolor="#F1F5F9"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_miss_distance_scatter(accepted_df: pd.DataFrame) -> go.Figure:
    """Plot Miss Distance vs Mahalanobis Distance colored by classification."""
    if accepted_df.empty or "miss_distance" not in accepted_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No accepted data for spatial analysis", showarrow=False)
        fig.update_layout(height=340)
        return fig

    color_map = {
        "High risk": "#DC2626",
        "Low risk": "#16A34A",
    }

    fig = px.scatter(
        accepted_df,
        x="miss_distance",
        y="mahalanobis_distance",
        color=OUTPUT_COLS["primary_class"],
        color_discrete_map=color_map,
        hover_data=[OUTPUT_COLS["event_id"], "time_to_tca", OUTPUT_COLS["primary_score"]],
        labels={
            "miss_distance": "Miss Distance (m)",
            "mahalanobis_distance": "Mahalanobis Distance",
            OUTPUT_COLS["primary_class"]: "Risk Class",
        },
    )

    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=1, color="#FFFFFF")))
    fig.update_layout(
        title=dict(text="Spatial Proximity vs Statistical Mahalanobis Distance", font=dict(size=14, color="#0F172A")),
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=40),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_model_comparison_bar(summary) -> go.Figure:
    """Compare predictions between Unweighted RF and SMOTENC RF."""
    models = ["Unweighted RF", "SMOTENC RF (0.25)"]
    high_risks = [summary.unweighted_high_risk, summary.smotenc_high_risk]
    low_risks = [summary.unweighted_low_risk, summary.smotenc_low_risk]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="High Risk", x=models, y=high_risks, marker_color="#DC2626"))
    fig.add_trace(go.Bar(name="Low Risk", x=models, y=low_risks, marker_color="#16A34A"))

    fig.update_layout(
        barmode="group",
        title=dict(text="Model Prediction Comparison", font=dict(size=14, color="#0F172A")),
        yaxis=dict(title="Classified Events", gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_object_type_distribution(accepted_df: pd.DataFrame) -> go.Figure:
    """Plot risk counts partitioned by chaser object type."""
    if accepted_df.empty or "c_object_type" not in accepted_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data available for object type distribution", showarrow=False)
        fig.update_layout(height=300)
        return fig

    grouped = (
        accepted_df.groupby(["c_object_type", OUTPUT_COLS["primary_class"]])
        .size()
        .reset_index(name="count")
    )

    color_map = {
        "High risk": "#DC2626",
        "Low risk": "#16A34A",
    }

    fig = px.bar(
        grouped,
        x="c_object_type",
        y="count",
        color=OUTPUT_COLS["primary_class"],
        color_discrete_map=color_map,
        barmode="stack",
        labels={
            "c_object_type": "Secondary Object Type (c_object_type)",
            "count": "Number of Events",
            OUTPUT_COLS["primary_class"]: "Risk Class",
        },
    )

    fig.update_layout(
        title=dict(text="Risk Classification by Object Type", font=dict(size=14, color="#0F172A")),
        yaxis=dict(gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
