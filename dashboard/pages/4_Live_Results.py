"""Live Results Page.

Enter actual prices from StockGro and view prediction accuracy.
"""

import logging

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.path_setup import setup_project_paths

logger = logging.getLogger(__name__)

setup_project_paths()

from config.config import STOCKS, PATHS, PRICE_BOUNDS, CHART_COLORS
from utils.data_loader import (
    load_live_actuals,
    load_live_vs_predicted,
    load_portfolio_performance,
    load_raw_prices,
)
from utils.chart_builder import build_live_comparison_table

st.title("Live Results")

# =============================================================================
# SECTION A: INPUT FORM
# =============================================================================
st.subheader("Enter StockGro Actual Closing Prices")
st.caption("Fill after market close on May 15, 2026 (3:25 PM IST)")

# Build form
with st.form(key="actuals_form"):
    inputs = {}
    
    st.markdown("**May 14, 2026**")
    cols1 = st.columns(4)
    for i, stock in enumerate(STOCKS):
        with cols1[i % 4]:
            bounds = PRICE_BOUNDS.get(stock, {"min": 0.0, "max": 100000.0})
            inputs[f"{stock}_2026-05-14"] = st.number_input(
                label=stock,
                min_value=0.0,
                max_value=bounds["max"],
                step=0.5,
                format="%.2f",
                value=0.0,
                key=f"may14_{stock}"
            )
            
    st.markdown("---")
    st.markdown("**May 15, 2026**")
    cols2 = st.columns(4)
    for i, stock in enumerate(STOCKS):
        with cols2[i % 4]:
            bounds = PRICE_BOUNDS.get(stock, {"min": 0.0, "max": 100000.0})
            inputs[f"{stock}_2026-05-15"] = st.number_input(
                label=stock,
                min_value=0.0,
                max_value=bounds["max"],
                step=0.5,
                format="%.2f",
                value=0.0,
                key=f"may15_{stock}"
            )
    
    submitted = st.form_submit_button("Save Actual Prices", type="primary")

if submitted:
    try:
        # Read current actuals
        with st.spinner("Loading current actuals..."):
            actuals_df = load_live_actuals()
        
        if actuals_df.empty:
            # Create structure
            dates = pd.to_datetime(["2026-05-14", "2026-05-15"])
            rows = []
            for stock in STOCKS:
                for date in dates:
                    rows.append({
                        "stock": stock,
                        "date": date,
                        "actual_close_price": np.nan,
                        "actual_return_pct": np.nan
                    })
            actuals_df = pd.DataFrame(rows)
        
        # Get last known prices from training data
        last_prices = {}
        for stock in STOCKS:
            with st.spinner(f"Loading price data for {stock}..."):
                raw_df = load_raw_prices(stock)
            if raw_df is not None and not raw_df.empty:
                last_prices[stock] = float(raw_df["Close"].iloc[-1])
            else:
                last_prices[stock] = None

        # Update with entered values
        updated = False
        for key, price in inputs.items():
            if price > 0:
                stock, date_str = key.rsplit("_", 1)
                date = pd.to_datetime(date_str)

                mask = (actuals_df["stock"] == stock) & (actuals_df["date"] == date)
                if mask.any():
                    actuals_df.loc[mask, "actual_close_price"] = price

                    # Calculate return
                    if last_prices.get(stock):
                        ret = (price / last_prices[stock] - 1) * 100
                        actuals_df.loc[mask, "actual_return_pct"] = ret

                    updated = True
        
        if updated:
            # Save back to CSV
            from pathlib import Path
            actuals_df.to_csv(Path(PATHS["data_external"]) / "live_actuals_may2026.csv", index=False)
            
            # Trigger re-evaluation
            from evaluation.evaluator import PortfolioEvaluator
            ev = PortfolioEvaluator()
            ev.compute_all_live_metrics()
            ev.compute_portfolio_performance()
            
            st.success("Prices saved. Results updated below.")
            st.rerun()
        else:
            logger.info("No prices entered (all values were 0).")
            st.info("No prices entered (all values were 0).")
    except (IOError, OSError) as e:
        logger.error(f"File write failed: {e}")
        st.error(f"Failed to save data: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error in form submission: {e}")
        st.error("An unexpected error occurred. Please try again.")

# =============================================================================
# SECTION B: RESULTS TABLE
# =============================================================================
st.markdown("---")
st.subheader("Predicted vs Actual — Live Window")

# Stock selector for detailed view
selected_stock = st.selectbox("Select Stock for Details", STOCKS)

# Load live vs predicted
with st.spinner("Loading live comparison data..."):
    live_vs_df = load_live_vs_predicted()

if not live_vs_df.empty:
    # Build and display table
    table_fig = build_live_comparison_table(live_vs_df, selected_stock)
    st.plotly_chart(table_fig, use_container_width=True)
    
    # Calculate directional accuracy if data available
    stock_data = live_vs_df[live_vs_df["stock"] == selected_stock]
    complete_data = stock_data[stock_data["status"] == "complete"]

    if not complete_data.empty and "direction_correct" in complete_data.columns:
        da = complete_data['direction_correct'].mean() * 100
        st.metric("Directional Accuracy", f"{da:.1f}%")
    else:
        logger.info("Awaiting actual data for directional accuracy calculation.")
        st.info("Awaiting actual data for directional accuracy calculation.")
else:
    logger.info("Live comparison data not yet generated.")
    st.info("Live comparison data not yet generated. Save actual prices above.")

# =============================================================================
# SECTION C: PORTFOLIO PERFORMANCE
# =============================================================================
st.markdown("---")
st.subheader("Portfolio Performance")

with st.spinner("Loading portfolio performance..."):
    perf_df = load_portfolio_performance()

if perf_df.empty or (perf_df["status"] == "pending").all():
    logger.info("Portfolio performance will appear after actual prices are entered.")
    st.info("Portfolio performance will appear after actual prices are entered above.")
else:
    # Check if any data available
    available = perf_df[perf_df["status"] != "pending"]
    
    if not available.empty:
        # Metric cards
        col1, col2, col3 = st.columns(3)
        
        # Total return
        if "actual_return_pct" in available.columns:
            total_ret = (available["weight_pct"] / 100 * available["actual_return_pct"]).sum()
            with col1:
                st.metric("Total Return", f"{total_ret:.2f}%")

        # Best performer
        best = available.loc[available["actual_return_pct"].idxmax()]
        with col2:
            st.metric("Best Performer", f"{best['stock']}", f"{best['actual_return_pct']:.2f}%")

        # Worst performer
        worst = available.loc[available["actual_return_pct"].idxmin()]
        with col3:
            st.metric("Worst Performer", f"{worst['stock']}", f"{worst['actual_return_pct']:.2f}%")
        
        # Bar chart of returns
        st.subheader("Stock Returns (Actual)")
        
        return_fig = {
            "data": [{
                "x": available["stock"].tolist(),
                "y": available["actual_return_pct"].tolist(),
                "type": "bar",
                "marker": {
                    "color": [CHART_COLORS["positive"] if r > 0 else CHART_COLORS["negative"] for r in available["actual_return_pct"]]
                }
            }],
            "layout": {
                "title": "Actual Return % by Stock",
                "yaxis": {"title": "Return %"},
                "template": "plotly_dark",
                "paper_bgcolor": CHART_COLORS["background"],
                "plot_bgcolor": CHART_COLORS["background"]
            }
        }
        st.plotly_chart(return_fig, use_container_width=True)
        
        # Written summary
        st.markdown(f"""
        ### Portfolio Performance Summary
        
        **Overall Return:** {total_ret:.2f}%
        
        **Best Performing Stock:** {best['stock']} (+{best['actual_return_pct']:.2f}%)
        
        **Worst Performing Stock:** {worst['stock']} ({worst['actual_return_pct']:.2f}%)
        
        **Strategy Validation:**
        The forecast-guided strategy (Strategy A) recommended higher allocations to
        stocks with predicted upside. Actual results will validate whether this
        approach outperformed a volatility-only (Strategy B) approach.
        
        *Note: Full evaluation requires all actual prices to be entered.*
        """)
    else:
        logger.info("No portfolio performance data available yet.")
        st.info("No portfolio performance data available yet.")
