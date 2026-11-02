"""Views package."""
from views.view_about import render_about_view
from views.view_analysis import render_analysis_view
from views.view_comparison import render_comparison_view
from views.view_manual import render_manual_view
from views.view_overview import render_overview_view
from views.view_reports import render_reports_view
from views.view_upload import render_upload_view

__all__ = [
    "render_overview_view",
    "render_upload_view",
    "render_manual_view",
    "render_comparison_view",
    "render_analysis_view",
    "render_reports_view",
    "render_about_view",
]

