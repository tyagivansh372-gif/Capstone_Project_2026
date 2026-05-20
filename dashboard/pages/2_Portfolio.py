"""Portfolio Page.

Displays portfolio allocation, sector breakdown, and correlation analysis.
"""

import logging

import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_setup import setup_project_paths

logger = logging.getLogger(__name__)

setup_project_paths()

from config.config import STOCKS, UI_CONFIG
from utils.data_loader import load_portfolio_allocation
from utils.chart_builder import (
    build_portfolio_pie,
    build_sector_bar,
    build_correlation_heatmap
)

st.title("Portfolio Allocation")

# Load data
with st.spinner("Loading portfolio allocation..."):
    allocation_df = load_portfolio_allocation()

if allocation_df.empty:
    st.error("Portfolio allocation data not found. Run Phase 6 first.")
    st.stop()

# Validate required columns
required_cols = ['ticker', 'weight_pct', 'allocated_INR', 'shares_to_buy', 'last_price']
missing_cols = [col for col in required_cols if col not in allocation_df.columns]
if missing_cols:
    st.error(f"Portfolio allocation CSV is missing required columns: {missing_cols}")
    st.stop()

# Validate for null values in critical columns
critical_cols = ['ticker', 'weight_pct', 'allocated_INR']
if allocation_df[critical_cols].isnull().any().any():
    st.error("Portfolio allocation CSV contains null values in critical columns. Data may be corrupted.")
    st.stop()

# Row 1: Pie and Bar charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Allocation by Stock")
    pie_fig = build_portfolio_pie(allocation_df)
    st.plotly_chart(pie_fig, use_container_width=True)

with col2:
    st.subheader("Allocation by Sector")
    bar_fig = build_sector_bar(allocation_df)
    st.plotly_chart(bar_fig, use_container_width=True)

# Row 2: Allocation table
st.subheader("Detailed Allocation")

# Format for display
display_df = allocation_df.copy()
display_df.columns = [
    'Stock', 'Weight %', 'Allocated INR', 'Shares', 'Last Price',
    'Predicted Return %', 'Avg Volatility', 'Strategy Rationale'
]

# Pagination
page_size: int = UI_CONFIG["portfolio_page_size"]
total_rows: int = len(display_df)
total_pages: int = max(1, -(-total_rows // page_size))  # ceiling division

page_num: int = st.number_input(
    "Page",
    min_value=1,
    max_value=total_pages,
    value=1,
    step=1,
    key="portfolio_page",
)
start_idx: int = (page_num - 1) * page_size
end_idx: int = min(start_idx + page_size, total_rows)

st.caption(f"Showing rows {start_idx + 1}–{end_idx} of {total_rows}")

st.dataframe(
    display_df.iloc[start_idx:end_idx],
    use_container_width=True,
    column_config={
        "Weight %": st.column_config.NumberColumn(format="%.2f%%"),
        "Allocated INR": st.column_config.NumberColumn(format="₹%.0f"),
        "Last Price": st.column_config.NumberColumn(format="₹%.2f"),
        "Predicted Return %": st.column_config.NumberColumn(format="%.3f%%"),
        "Avg Volatility": st.column_config.NumberColumn(format="%.4f"),
    },
)

# Row 3: Correlation heatmap
st.subheader("Return Correlation Matrix")
heatmap_fig = build_correlation_heatmap(STOCKS)
st.plotly_chart(heatmap_fig, use_container_width=True)

# Strategy notes expander
with st.expander("Portfolio Strategy Notes"):
    st.markdown("""
    ### Strategy A + B Combination
    
    The portfolio uses a hybrid allocation strategy:
    
    **Strategy A (Forecast-Guided, 60% weight)**
    - Allocates based on predicted 5-day returns
    - Higher predicted return = higher allocation
    - Stocks ranked and weighted proportionally
    
    **Strategy B (Volatility-Aware, 40% weight)**
    - Allocates based on inverse volatility
    - Lower volatility = higher allocation
    - Risk-adjusted position sizing
    
    **Final Allocation = 0.6 × Strategy A + 0.4 × Strategy B**
    
    ---
    
    ### Key Adjustments
    
    **MARUTI.NS Modification**
    Due to documented high MAPE (14-22%) across all models, MARUTI's Strategy A
    weight was reduced from 60% to 40%, with Strategy B increased to 60%.
    This gives more weight to volatility-based allocation rather than
    forecast-based allocation for this stock.
    
    **TATASTEEL.NS & BAJFINANCE.NS Note**
    Initial ARIMA models selected degenerate (0,1,0) orders producing
    flatline forecasts with 0% directional accuracy. This was detected
    during audit and fixed by adding min_p=1, min_q=1 constraints.
    Both models now show healthy variance and >40% directional accuracy.
    
    ---
    
    *All position sizes capped at 20% maximum per stock to ensure diversification.*
    """)
