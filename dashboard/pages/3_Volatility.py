"""Volatility Page.

Displays GARCH volatility forecasts and rolling volatility analysis.
"""

import logging

import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_setup import setup_project_paths

logger = logging.getLogger(__name__)

setup_project_paths()

from config.config import STOCKS, RISK_THRESHOLDS
from utils.data_loader import load_garch_volatility
from utils.chart_builder import (
    build_rolling_volatility,
    build_garch_bar
)

st.title("Volatility Analysis")

# Load data
with st.spinner("Loading GARCH volatility data..."):
    garch_df = load_garch_volatility()

if garch_df.empty:
    st.error("GARCH volatility data not found. Run Phase 4C first.")
    st.stop()

# Row 1: Stock multiselect
selected_stocks = st.multiselect(
    "Select Stocks for Analysis",
    STOCKS,
    default=STOCKS
)

# Row 2: Rolling volatility chart
st.subheader("Rolling 30-Day Volatility (Historical)")
vol_fig = build_rolling_volatility(selected_stocks)
st.plotly_chart(vol_fig, use_container_width=True)

# Row 3: Two columns - GARCH bar and ranking table
col1, col2 = st.columns(2)

with col1:
    st.subheader("GARCH Forecasted Volatility")
    
    # Date selector
    available_dates = garch_df['date'].dt.strftime('%Y-%m-%d').unique()
    selected_date_str = st.selectbox("Select Date", available_dates)
    selected_date = pd.to_datetime(selected_date_str)
    
    # GARCH bar chart
    garch_fig = build_garch_bar(garch_df, selected_date_str)
    st.plotly_chart(garch_fig, use_container_width=True)

with col2:
    st.subheader("Volatility Ranking")
    
    # Build ranking table
    pivot_df = garch_df.pivot(index='stock', columns='date', values='forecasted_volatility')
    pivot_df.columns = [f"May {d.day}" for d in pivot_df.columns]
    
    # Calculate trend and risk level
    if len(pivot_df.columns) >= 2:
        col1_name = pivot_df.columns[0]
        col2_name = pivot_df.columns[1]
        
        pivot_df['Trend'] = pivot_df.apply(
            lambda row: '↑' if row[col2_name] > row[col1_name] else 
                       '↓' if row[col2_name] < row[col1_name] else '→',
            axis=1
        )
    else:
        pivot_df['Trend'] = '→'
    
    # Risk level based on latest date
    latest_col = pivot_df.columns[0]
    pivot_df['Risk Level'] = pivot_df[latest_col].apply(
        lambda x: 'High' if x > RISK_THRESHOLDS['high'] else 
                  'Med' if x > RISK_THRESHOLDS['medium'] else 'Low'
    )
    
    # Format percentages
    for col in pivot_df.columns:
        if col not in ['Trend', 'Risk Level']:
            pivot_df[col] = (pivot_df[col] * 100).round(2)
    
    # Sort by volatility
    pivot_df = pivot_df.sort_values(pivot_df.columns[0], ascending=False)
    
    # Color code Risk Level
    def color_risk(val):
        if val == 'High':
            return 'background-color: #E74C3C; color: white'
        elif val == 'Med':
            return 'background-color: #F39C12; color: white'
        else:
            return 'background-color: #27AE60; color: white'
    
    styled_df = pivot_df.style.applymap(color_risk, subset=['Risk Level'])
    st.dataframe(styled_df, use_container_width=True)

st.markdown("---")
st.caption(f"""
**Risk Level Thresholds:**
- **High Risk** (>{RISK_THRESHOLDS['high']*100:.1f}%): Elevated volatility expected
- **Medium Risk** ({RISK_THRESHOLDS['medium']*100:.1f}%-{RISK_THRESHOLDS['high']*100:.1f}%): Moderate volatility expected  
- **Low Risk** (<{RISK_THRESHOLDS['medium']*100:.1f}%): Stable conditions expected
""")
