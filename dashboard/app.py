"""Streamlit Dashboard Entry Point for TSA Capstone 2026.

Minimal entry point for multipage Streamlit architecture.
All page logic is in dashboard/pages/ directory.
"""

import logging
import streamlit as st
from utils.path_setup import setup_project_paths

logger = logging.getLogger(__name__)

setup_project_paths()

from config.config import STOCKS, DATE_RANGES

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="TSA Capstone 2026",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title('TSA Capstone 2026')
st.sidebar.caption('NSE Stock Forecasting & Portfolio Analysis')
st.sidebar.markdown('---')
st.sidebar.markdown(f"Trading Window: {DATE_RANGES['live_forecast_start']} – {DATE_RANGES['live_forecast_end']}")
st.sidebar.markdown('Portfolio: ₹10,00,000 virtual capital')
st.sidebar.markdown('Stocks: 8 NSE large-caps')
st.sidebar.markdown('---')

# =============================================================================
# HOME PAGE CONTENT
# =============================================================================
from utils.data_loader import load_portfolio_allocation, load_model_comparison

st.title('Time Series Analysis — Capstone 2026')
st.markdown('IIT Guwahati | Consulting & Analytics Club')

# Summary metric cards
allocation = load_portfolio_allocation()
metrics = load_model_comparison()

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_capital = allocation['allocated_INR'].sum() if not allocation.empty else 1000000
    st.metric('Total Capital', f'₹{total_capital:,.0f}')

with col2:
    num_stocks = len(STOCKS)
    st.metric('No. of Stocks', num_stocks)

with col3:
    if not metrics.empty and 'mape' in metrics.columns:
        best_model = metrics.groupby('model')['mape'].mean().idxmin()
        st.metric('Best Model', best_model.upper())
    else:
        st.metric('Best Model', 'N/A')

with col4:
    st.metric('Trading Day', 'May 14-15')

st.markdown('---')

st.markdown("""
### 📊 Dashboard Pages

Navigate using the sidebar to explore:

- **Stock Forecasts** — View price predictions and model comparisons
- **Portfolio** — Allocation breakdown by stock and sector
- **Volatility** — GARCH forecasts and rolling volatility analysis
- **Live Results** — Enter actual prices from StockGro and evaluate predictions
- **Downloads** — Export all project data and reports

### 🎯 Project Overview

This capstone project implements a complete time series forecasting pipeline
for 8 NSE large-cap stocks across 7 sectors. The system uses 6+ statistical
and ML models, ensembles predictions, and optimizes a ₹10L virtual portfolio.

**Models Implemented:**
- ARIMA & SARIMA (statistical)
- Holt-Winters (exponential smoothing)
- Prophet (Facebook)
- LSTM & GRU (deep learning)
- GARCH (volatility)
- Ensemble (weighted combination)

**Key Features:**
- Real-time forecast visualization
- Portfolio allocation by sector
- Volatility risk assessment
- Live performance tracking

---
*Built with Streamlit • Data from yfinance • Trading on StockGro*
""")
