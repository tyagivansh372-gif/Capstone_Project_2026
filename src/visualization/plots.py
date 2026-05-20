"""Visualization module for forecasts and analysis.

Generates plots using plotly with professional styling.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.config import CHART_COLORS, PATHS, STOCK_INFO, STOCKS

logger = logging.getLogger(__name__)


class ForecastVisualizer:
    """Creates forecast visualizations and charts.

    Attributes:
        plots_path: Directory to save plots
    """

    def __init__(self) -> None:
        """Initialize visualizer."""
        self.plots_path: Path = Path(PATHS["outputs_plots"])
        self.plots_path.mkdir(parents=True, exist_ok=True)

        # Use centralized color palette from config
        self.colors = {
            "background": "#0E1117",
            "secondary_bg": CHART_COLORS["table_cell_default"],
            "accent": CHART_COLORS["primary"],
            "text": "#FFFFFF",
            "grid": CHART_COLORS["grid"],
        }

    def plot_stock_forecast(
        self,
        ticker: str,
        actual: pd.Series,
        forecast: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        confidence_lower: Optional[np.ndarray] = None,
        confidence_upper: Optional[np.ndarray] = None,
        live_forecast: Optional[pd.Series] = None,
        live_dates: Optional[pd.DatetimeIndex] = None,
    ) -> go.Figure:
        """Create forecast vs actual plot for a single stock.
        
        Args:
            ticker: Stock ticker
            actual: Actual price series
            forecast: Forecasted prices (backtest period)
            forecast_dates: Dates for forecasts
            confidence_lower: Lower confidence bound
            confidence_upper: Upper confidence bound
            live_forecast: Live forecast for May 2026
            live_dates: Dates for live forecast
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            name="Actual",
            line=dict(color="white", width=2),
        ))
        
        # Backtest forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            name="Forecast (Backtest)",
            line=dict(color=self.colors["accent"], width=2, dash="dash"),
        ))
        
        # Confidence interval
        if confidence_lower is not None and confidence_upper is not None:
            fig.add_trace(go.Scatter(
                x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                y=np.concatenate([confidence_upper, confidence_lower[::-1]]),
                fill="toself",
                fillcolor=CHART_COLORS["ci_fill"],
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Interval",
            ))
        
        # Live forecast
        if live_forecast is not None and live_dates is not None:
            fig.add_trace(go.Scatter(
                x=live_dates,
                y=live_forecast,
                name="Live Forecast (May 2026)",
                line=dict(color=CHART_COLORS["primary"], width=3),
                mode="lines+markers",
            ))
        
        # Layout
        stock_name = STOCK_INFO.get(ticker, {}).get("name", ticker)
        fig.update_layout(
            title=f"{stock_name} ({ticker}) - Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            plot_bgcolor=self.colors["background"],
            paper_bgcolor=self.colors["background"],
            font=dict(color=self.colors["text"]),
            legend=dict(
                bgcolor=self.colors["secondary_bg"],
                bordercolor=self.colors["grid"],
                borderwidth=1,
            ),
            xaxis=dict(
                gridcolor=self.colors["grid"],
                showgrid=True,
            ),
            yaxis=dict(
                gridcolor=self.colors["grid"],
                showgrid=True,
            ),
        )
        
        return fig
    
    def save_stock_forecast_plot(self, ticker: str, fig: go.Figure) -> None:
        """Save stock forecast plot to PNG."""
        filepath = self.plots_path / f"{ticker.replace('.', '_')}_forecast.png"
        fig.write_image(filepath, width=1200, height=600, scale=2)
        logger.info(f"Saved forecast plot for {ticker}")
    
    def plot_correlation_heatmap(self, returns_df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap of stock returns.
        
        Args:
            returns_df: DataFrame of returns with stocks as columns
            
        Returns:
            Plotly figure object
        """
        corr_matrix = returns_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
        )
        
        fig.update_layout(
            title="Stock Returns Correlation Matrix",
            plot_bgcolor=self.colors["background"],
            paper_bgcolor=self.colors["background"],
            font=dict(color=self.colors["text"]),
        )
        
        return fig
    
    def plot_volatility(self, volatility_data: Dict[str, pd.Series]) -> go.Figure:
        """Plot rolling volatility for all stocks.
        
        Args:
            volatility_data: Dictionary of volatility series per stock
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Plotly
        
        for i, (ticker, vol_series) in enumerate(volatility_data.items()):
            fig.add_trace(go.Scatter(
                x=vol_series.index,
                y=vol_series.values * 100,  # Convert to percentage
                name=ticker,
                line=dict(color=colors[i % len(colors)]),
            ))
        
        fig.update_layout(
            title="30-Day Rolling Volatility (Annualized %)",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            plot_bgcolor=self.colors["background"],
            paper_bgcolor=self.colors["background"],
            font=dict(color=self.colors["text"]),
            legend=dict(
                bgcolor=self.colors["secondary_bg"],
                bordercolor=self.colors["grid"],
            ),
            xaxis=dict(gridcolor=self.colors["grid"]),
            yaxis=dict(gridcolor=self.colors["grid"]),
        )
        
        return fig
    
    def plot_portfolio_allocation(self, allocation_df: pd.DataFrame) -> go.Figure:
        """Create portfolio allocation pie chart.
        
        Args:
            allocation_df: DataFrame with ticker and weight_pct columns
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("Portfolio Allocation", "Investment Amount (INR)")
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=allocation_df["ticker"],
                values=allocation_df["weight_pct"],
                hole=0.4,
                marker=dict(
                    colors=px.colors.qualitative.Plotly,
                    line=dict(color=self.colors["background"], width=2)
                ),
                textinfo="label+percent",
                textfont=dict(color="white"),
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=allocation_df["ticker"],
                y=allocation_df["allocated_INR"],
                marker_color=px.colors.qualitative.Plotly,
                text=allocation_df["allocated_INR"].apply(lambda x: f"₹{x:,.0f}"),
                textposition="outside",
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Portfolio Allocation Summary",
            plot_bgcolor=self.colors["background"],
            paper_bgcolor=self.colors["background"],
            font=dict(color=self.colors["text"]),
            showlegend=False,
            height=500,
        )
        
        fig.update_xaxes(gridcolor=self.colors["grid"], row=1, col=2)
        fig.update_yaxes(gridcolor=self.colors["grid"], row=1, col=2)
        
        return fig
    
    def save_all_charts(self) -> None:
        """Save all standard charts to outputs/plots/."""
        # Load data
        from preprocessing.preprocessor import DataPreprocessor
        from portfolio.optimizer import PortfolioOptimizer
        from volatility.analyzer import VolatilityAnalyzer
        
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        
        # Correlation heatmap
        returns_dict = {}
        for ticker in STOCKS:
            if ticker in preprocessor.train_data:
                prices = preprocessor.train_data[ticker]["Close"].squeeze()
                returns_dict[ticker] = prices.pct_change()
        
        returns_df = pd.DataFrame(returns_dict).dropna()
        if not returns_df.empty:
            corr_fig = self.plot_correlation_heatmap(returns_df)
            corr_fig.write_image(self.plots_path / "correlation_heatmap.png", width=800, height=700, scale=2)
        
        # Volatility plot
        analyzer = VolatilityAnalyzer()
        analyzer.compute_rolling_volatility()
        vol_data = {t: v["vol_30d"] for t, v in analyzer.volatility_data.items() if "vol_30d" in v}
        if vol_data:
            vol_fig = self.plot_volatility(vol_data)
            vol_fig.write_image(self.plots_path / "volatility_rolling.png", width=1200, height=600, scale=2)
        
        # Portfolio allocation
        optimizer = PortfolioOptimizer()
        optimizer.load_forecasts()
        optimizer.strategy_a_forecast_guided()
        optimizer.strategy_b_volatility_aware()
        optimizer.combine_strategies()
        
        if not optimizer.allocation.empty:
            alloc_fig = self.plot_portfolio_allocation(optimizer.allocation)
            alloc_fig.write_image(self.plots_path / "portfolio_allocation_pie.png", width=1200, height=500, scale=2)
        
        logger.info("All charts saved to outputs/plots/")
