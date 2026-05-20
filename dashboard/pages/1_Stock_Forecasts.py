"""Stock Forecasts Page.

Displays price forecasts with historical context and model comparisons.
"""

import logging
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_setup import setup_project_paths

logger = logging.getLogger(__name__)

setup_project_paths()

from config.config import STOCKS
from utils.data_loader import (
    load_live_forecasts,
    load_model_comparison,
    load_raw_prices
)
from utils.chart_builder import (
    build_forecast_chart,
    build_model_comparison_table
)

def render_empty_state(message: str) -> None:
    """Render a user-facing empty-state info message.

    Logs the message at INFO level for operational visibility and displays
    it to the user via st.info.

    Args:
        message: Human-readable description of why data is unavailable.

    Returns:
        None
    """
    logger.info(message)
    st.info(message)

st.title("Stock Forecasts")

# Stock selector
selected_stock = st.selectbox("Select Stock", STOCKS, index=0)

# Load data
with st.spinner("Loading forecast data..."):
    live_df = load_live_forecasts()
with st.spinner("Loading model comparison..."):
    metrics_df = load_model_comparison()
with st.spinner("Loading price data..."):
    raw_df = load_raw_prices(selected_stock)

# Full-width forecast chart
if not live_df.empty and selected_stock:
    st.subheader(f"Price Forecast - {selected_stock}")
    fig = build_forecast_chart(selected_stock, raw_df, None, live_df)
    st.plotly_chart(fig, use_container_width=True)
else:
    logger.warning("No forecast data available for selected stock.")
    st.warning("No forecast data available for selected stock.")

# Two columns below chart
col1, col2 = st.columns(2)

with col1:
    # Model comparison table
    if not metrics_df.empty:
        st.subheader("Model Comparison")
        table_fig = build_model_comparison_table(metrics_df, selected_stock)
        st.plotly_chart(table_fig, use_container_width=True)

with col2:
    # Metrics summary cards
    st.subheader("Metrics Summary")
    
    if not metrics_df.empty:
        stock_metrics = metrics_df[metrics_df['stock'] == selected_stock]
        
        if not stock_metrics.empty:
            best_row = stock_metrics.loc[stock_metrics['mape'].idxmin()]
            
            st.metric("Best Model MAPE", f"{best_row['mape']:.2f}%")
            
            # Ensemble MAPE
            ensemble_row = stock_metrics[stock_metrics['model'] == 'ensemble']
            if not ensemble_row.empty:
                st.metric("Ensemble MAPE", f"{ensemble_row.iloc[0]['mape']:.2f}%")
            
            st.metric("Best Model DA", f"{best_row['directional_accuracy']:.1f}%")
            
            # MARUTI warning
            if selected_stock == 'MARUTI.NS':
                st.warning("""
                ⚠️ **Documented High MAPE**
                
                MARUTI shows elevated MAPE (14-22%) across all models.
                This is a known limitation — price-only models cannot
                capture fuel price policy and regulatory sensitivity.
                
                Mitigation: Reduced Strategy A weight in portfolio.
                """)
        else:
            render_empty_state("No metrics available for this stock.")
    else:
        render_empty_state("Model comparison data not available.")
