"""Chart building utilities for Streamlit dashboard.

All Plotly figure construction happens here. Import these functions
from page files — never build figures inline in pages.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.config import CHART_COLORS

logger = logging.getLogger(__name__)


def build_forecast_chart(
    stock: str,
    raw_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    live_df: pd.DataFrame
) -> go.Figure:
    """Build three-layer forecast chart.
    
    Layers:
    1. Historical close prices (white, solid, 1.5px)
    2. Backtest predictions Jul-Dec 2025 (#F5A623, dashed)
    3. Live forecast May 14-15 (#4F8BF9, solid with markers) + confidence interval
    
    Args:
        stock: Stock ticker
        raw_df: Historical price data with DatetimeIndex
        backtest_df: Backtest predictions
        live_df: Live forecasts with confidence intervals
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    # Layer 1: Historical prices (last 90 days for context)
    if raw_df is not None and not raw_df.empty:
        hist_data = raw_df.tail(90) if len(raw_df) > 90 else raw_df
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data["Close"],
            mode="lines",
            name="Historical Close",
            line=dict(color="white", width=1.5),
            hovertemplate="%{x}<br>₹%{y:.2f}<extra>Historical</extra>"
        ))
    
    # Layer 2: Backtest predictions
    if backtest_df is not None and not backtest_df.empty:
        price_col = "predicted_price" if "predicted_price" in backtest_df.columns else "forecast"
        fig.add_trace(go.Scatter(
            x=backtest_df["date"],
            y=backtest_df[price_col],
            mode="lines",
            name="Backtest Forecast (Jul-Dec 2025)",
            line=dict(color=CHART_COLORS["backtest"], width=2, dash="dash"),
            hovertemplate="%{x}<br>₹%{y:.2f}<extra>Backtest</extra>"
        ))
    
    # Layer 3: Live forecast with confidence interval
    if live_df is not None and not live_df.empty:
        stock_live = live_df[live_df["stock"] == stock] if "stock" in live_df.columns else live_df
        
        if not stock_live.empty:
            # Confidence interval shading
            if "confidence_interval_low" in stock_live.columns and "confidence_interval_high" in stock_live.columns:
                fig.add_trace(go.Scatter(
                    x=stock_live["date"],
                    y=stock_live["confidence_interval_high"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(
                    x=stock_live["date"],
                    y=stock_live["confidence_interval_low"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor=CHART_COLORS["ci_fill"],
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))
            
            # Live forecast line
            fig.add_trace(go.Scatter(
                x=stock_live["date"],
                y=stock_live["predicted_price"],
                mode="lines+markers",
                name="Live Forecast (May 14-15)",
                line=dict(color=CHART_COLORS["live_forecast"], width=2.5),
                marker=dict(size=8, color=CHART_COLORS["live_forecast"]),
                hovertemplate="%{x}<br>₹%{y:.2f}<extra>Live Forecast</extra>"
            ))
    
    fig.update_layout(
        title=f"{stock} — Price Forecast",
        title_font_size=18,
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["background"],
        plot_bgcolor=CHART_COLORS["background"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=True,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    return fig


def build_portfolio_pie(allocation_df: pd.DataFrame) -> go.Figure:
    """Build pie chart of portfolio allocation.
    
    Args:
        allocation_df: Portfolio allocation DataFrame
        
    Returns:
        Plotly Figure
    """
    if allocation_df.empty or "weight_pct" not in allocation_df.columns:
        return go.Figure()

    labels = allocation_df["ticker"].tolist()
    values = allocation_df["weight_pct"].tolist()
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        textinfo="label+percent",
        textposition="outside",
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Vivid),
        hovertemplate="%{label}<br>%{percent}<br>₹%{value:.0f}<extra></extra>"
    )])
    
    fig.update_layout(
        title="Portfolio Allocation by Stock",
        title_font_size=16,
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["background"],
        plot_bgcolor=CHART_COLORS["background"],
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def build_sector_bar(allocation_df: pd.DataFrame) -> go.Figure:
    """Build horizontal bar chart of capital allocation by sector.
    
    Args:
        allocation_df: Portfolio allocation DataFrame
        
    Returns:
        Plotly Figure
    """
    from config.config import STOCK_INFO
    
    if allocation_df.empty:
        return go.Figure()
    
    # Sector mapping from config
    sector_map = {k: v["sector"] for k, v in STOCK_INFO.items()}

    allocation_df = allocation_df.copy()
    allocation_df["sector"] = allocation_df["ticker"].map(sector_map)

    # Aggregate by sector
    sector_df = allocation_df.groupby("sector")["allocated_INR"].sum().reset_index()
    sector_df = sector_df.sort_values("allocated_INR", ascending=True)
    
    fig = go.Figure(data=[go.Bar(
        x=sector_df["allocated_INR"],
        y=sector_df["sector"],
        orientation="h",
        marker=dict(
            color=sector_df["allocated_INR"],
            colorscale="Viridis"
        ),
        hovertemplate="%{y}<br>₹%{x:,.0f}<extra></extra>"
    )])
    
    fig.update_layout(
        title="Capital Allocation by Sector",
        title_font_size=16,
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["background"],
        plot_bgcolor=CHART_COLORS["background"],
        xaxis_title="Allocated Capital (₹)",
        yaxis_title=None,
        margin=dict(l=150, r=40, t=60, b=60),
        xaxis=dict(showgrid=True, gridcolor=CHART_COLORS["grid"]),
        yaxis=dict(showgrid=False)
    )
    
    return fig


@st.cache_data(ttl=300)
def build_correlation_heatmap(stocks: List[str]) -> go.Figure:
    """Build correlation heatmap of stock returns.
    
    Args:
        stocks: List of stock tickers
        
    Returns:
        Plotly Figure
    """
    # Load price data and compute returns
    returns_data = {}
    
    for stock in stocks:
        path = f"data/processed/{stock.replace('.', '_')}_train.csv"
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            df = df.set_index("Date")
            if "Close" in df.columns:
                # Get last 90 days
                recent = df["Close"].tail(90)
                returns = recent.pct_change().dropna()
                returns_data[stock] = returns
        except (pd.errors.EmptyDataError, FileNotFoundError, KeyError) as e:
            logger.error(f"Failed to load data for {stock}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading data for {stock}: {e}")
    
    if not returns_data:
        return go.Figure()
    
    # Create returns DataFrame and compute correlation
    returns_df = pd.DataFrame(returns_data)
    corr_matrix = returns_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title="Return Correlation Matrix (90-day)",
        title_font_size=16,
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["background"],
        plot_bgcolor=CHART_COLORS["background"],
        margin=dict(l=100, r=40, t=80, b=80),
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def build_rolling_volatility(stocks: List[str]) -> go.Figure:
    """Build rolling 30-day volatility chart.
    
    Args:
        stocks: List of stock tickers
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Vivid
    
    for i, stock in enumerate(stocks):
        path = f"data/processed/{stock.replace('.', '_')}_train.csv"
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            df = df.set_index("Date")
            if "Close" in df.columns:
                # Compute log returns and rolling volatility
                log_returns = np.log(df["Close"] / df["Close"].shift(1))
                rolling_vol = log_returns.rolling(window=30).std() * np.sqrt(252) * 100

                fig.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    mode="lines",
                    name=stock,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate="%{x}<br>Vol: %{y:.2f}%<extra>" + stock + "</extra>"
                ))
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, ValueError) as e:
            logger.warning("Could not load volatility data for %s: %s", stock, e)
            continue
    
    fig.update_layout(
        title="Rolling 30-Day Volatility",
        title_font_size=16,
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["background"],
        plot_bgcolor=CHART_COLORS["background"],
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=100, b=60),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=CHART_COLORS["grid"])
    )
    
    return fig


def build_garch_bar(garch_df: pd.DataFrame, selected_date: str) -> go.Figure:
    """Build GARCH volatility bar chart for selected date.
    
    Args:
        garch_df: GARCH volatility DataFrame
        selected_date: Date string (YYYY-MM-DD)
        
    Returns:
        Plotly Figure
    """
    from config.config import RISK_THRESHOLDS
    
    if garch_df.empty or "forecasted_volatility" not in garch_df.columns:
        return go.Figure()

    # Filter for date
    date_data = garch_df[garch_df["date"] == selected_date].copy()

    if date_data.empty:
        return go.Figure()

    # Sort by volatility
    date_data = date_data.sort_values("forecasted_volatility", ascending=True)

    # Color by risk level
    colors = []
    for vol in date_data["forecasted_volatility"]:
        if vol > RISK_THRESHOLDS["high"]:
            colors.append(CHART_COLORS["negative"])  # Red - High
        elif vol > RISK_THRESHOLDS["medium"]:
            colors.append(CHART_COLORS["neutral"])  # Orange - Medium
        else:
            colors.append(CHART_COLORS["positive"])  # Green - Low

    fig = go.Figure(data=[go.Bar(
        x=date_data["forecasted_volatility"] * 100,  # Convert to percentage
        y=date_data["stock"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>Vol: %{x:.2f}%<extra></extra>"
    )])

    fig.update_layout(
        title=f"GARCH Forecasted Volatility — {selected_date}",
        title_font_size=16,
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["background"],
        plot_bgcolor=CHART_COLORS["background"],
        xaxis_title="Forecasted Volatility (%)",
        yaxis_title=None,
        margin=dict(l=100, r=40, t=60, b=60),
        xaxis=dict(showgrid=True, gridcolor=CHART_COLORS["grid"]),
        yaxis=dict(showgrid=False)
    )

    return fig


def build_model_comparison_table(
    metrics_df: pd.DataFrame,
    selected_stock: str
) -> go.Figure:
    """Build model comparison table for selected stock.
    
    Args:
        metrics_df: Model comparison DataFrame
        selected_stock: Stock ticker to filter by
        
    Returns:
        Plotly Figure (Table)
    """
    if metrics_df.empty:
        return go.Figure()
    
    # Filter for stock
    stock_df = metrics_df[metrics_df["stock"] == selected_stock].copy()

    if stock_df.empty:
        return go.Figure()

    # Sort by MAPE
    stock_df = stock_df.sort_values("mape", ascending=True)

    # Prepare data for table
    best_idx = stock_df.index[0]
    worst_idx = stock_df.index[-1]

    fill_colors = []
    for idx in stock_df.index:
        if idx == best_idx:
            fill_colors.append(CHART_COLORS["table_cell_best"])
        elif idx == worst_idx:
            fill_colors.append(CHART_COLORS["table_cell_worst"])
        else:
            fill_colors.append(CHART_COLORS["table_cell_default"])

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Model", "MAPE (%)", "RMSE", "MAE", "Directional Accuracy (%)"],
            fill_color=CHART_COLORS["table_header"],
            font=dict(color="white", size=12),
            align="left"
        ),
        cells=dict(
            values=[
                stock_df["model"].str.upper(),
                stock_df["mape"].round(2),
                stock_df["rmse"].round(2),
                stock_df["mae"].round(2),
                stock_df["directional_accuracy"].round(1)
            ],
            fill_color=[fill_colors],
            font=dict(color="white", size=11),
            align="left",
            height=30
        )
    )])

    fig.update_layout(
        title=f"Model Comparison — {selected_stock}",
        title_font_size=16,
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["background"],
        plot_bgcolor=CHART_COLORS["background"],
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def build_live_comparison_table(
    live_df: pd.DataFrame,
    stock: str
) -> go.Figure:
    """Build live prediction vs actual table for selected stock.
    
    Args:
        live_df: Live vs predicted DataFrame
        stock: Stock ticker to filter by
        
    Returns:
        Plotly Figure (Table)
    """
    if live_df.empty:
        return go.Figure()
    
    # Filter for stock
    stock_df = live_df[live_df["stock"] == stock].copy()

    if stock_df.empty:
        return go.Figure()

    # Prepare display columns
    dates = []
    predicted = []
    actual = []
    error = []
    direction = []
    status = []

    for _, row in stock_df.iterrows():
        date_str = pd.to_datetime(row["date"]).strftime("%b %d")
        dates.append(date_str)
        predicted.append(f"₹{row['predicted_price']:.2f}")

        if pd.isna(row.get("actual_close_price")):
            actual.append("Awaiting data")
            error.append("-")
            direction.append("-")
            status.append("⏳ Pending")
        else:
            actual.append(f"₹{row['actual_close_price']:.2f}")

            if pd.notna(row.get("prediction_error_pct")):
                error.append(f"{row['prediction_error_pct']:.2f}%")
            else:
                error.append("-")

            if row.get("direction_correct"):
                direction.append("✓")
            else:
                direction.append("✗")

            status.append("✅ Complete")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Date", "Predicted", "Actual", "Error %", "Dir", "Status"],
            fill_color=CHART_COLORS["table_header"],
            font=dict(color="white", size=12),
            align="left"
        ),
        cells=dict(
            values=[dates, predicted, actual, error, direction, status],
            fill_color=CHART_COLORS["table_cell_default"],
            font=dict(
                color=["white", "white", "white", "white",
                       [CHART_COLORS["positive"] if d == "✓" else CHART_COLORS["negative"] if d == "✗" else "white" for d in direction],
                       "white"],
                size=11
            ),
            align="left",
            height=35
        )
    )])

    fig.update_layout(
        title=f"Live Prediction vs Actual — {stock}",
        title_font_size=16,
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["background"],
        plot_bgcolor=CHART_COLORS["background"],
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig
