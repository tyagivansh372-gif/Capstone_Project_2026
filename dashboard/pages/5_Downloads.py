"""Downloads Page.

Provides download buttons for all project outputs.
"""

import streamlit as st
from pathlib import Path
from utils.path_setup import setup_project_paths

setup_project_paths()

st.title("Download Project Outputs")

# =============================================================================
# SECTION 1: FORECAST DATA
# =============================================================================
with st.expander("Forecast Data", expanded=True):
    st.markdown("### Live Forecasts and Volatility")
    
    files_forecast = [
        ('outputs/forecasts/live_forecasts_may2026.csv', 'Live Forecasts (May 14-15)'),
        ('outputs/forecasts/garch_volatility_may2026.csv', 'GARCH Volatility Forecasts'),
    ]
    
    for path, label in files_forecast:
        file_path = Path(path)
        if file_path.exists():
            with open(file_path, 'rb') as f:
                st.download_button(
                    label=f"Download {label}",
                    data=f.read(),
                    file_name=file_path.name,
                    mime='text/csv',
                    key=f"dl_{file_path.name}"
                )
        else:
            st.warning(f"{label} not yet generated")

# =============================================================================
# SECTION 2: EVALUATION & METRICS
# =============================================================================
with st.expander("Evaluation & Metrics"):
    st.markdown("### Model Comparison and Performance Reports")
    
    files_metrics = [
        ('outputs/metrics/model_comparison.csv', 'Model Comparison Metrics'),
        ('outputs/reports/live_vs_predicted.csv', 'Live vs Predicted Comparison'),
        ('outputs/reports/portfolio_performance.csv', 'Portfolio Performance'),
        ('outputs/reports/model_comparison_report.md', 'Model Comparison Report (Markdown)'),
    ]
    
    for path, label in files_metrics:
        file_path = Path(path)
        if file_path.exists():
            mime = 'text/markdown' if path.endswith('.md') else 'text/csv'
            with open(file_path, 'rb') as f:
                st.download_button(
                    label=f"Download {label}",
                    data=f.read(),
                    file_name=file_path.name,
                    mime=mime,
                    key=f"dl_{file_path.name}"
                )
        else:
            st.warning(f"{label} not yet generated")

# =============================================================================
# SECTION 3: PORTFOLIO
# =============================================================================
with st.expander("Portfolio"):
    st.markdown("### Portfolio Allocation")
    
    files_portfolio = [
        ('outputs/reports/portfolio_allocation.csv', 'Portfolio Allocation'),
    ]
    
    for path, label in files_portfolio:
        file_path = Path(path)
        if file_path.exists():
            with open(file_path, 'rb') as f:
                st.download_button(
                    label=f"Download {label}",
                    data=f.read(),
                    file_name=file_path.name,
                    mime='text/csv',
                    key=f"dl_{file_path.name}"
                )
        else:
            st.warning(f"{label} not yet generated")

st.markdown("---")
st.info("""
**Note:** Some files require actual prices to be entered on the **Live Results** page
before they will contain complete data. Forecast data is always available.
""")
