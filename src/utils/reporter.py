"""Report generation module.

Generates final summary markdown and exports charts.
"""

import logging
from datetime import datetime
from pathlib import Path


from config.config import (
    DATE_RANGES,
    PATHS,
    PROJECT_NAME,
    PROJECT_VERSION,
    STOCK_INFO,
    STOCKS,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates final project reports and summaries.
    
    Attributes:
        reports_path: Directory to save reports
    """
    
    def __init__(self) -> None:
        """Initialize reporter."""
        self.reports_path: Path = Path(PATHS["outputs_reports"])
        self.plots_path: Path = Path(PATHS["outputs_plots"])
        self.metrics_path: Path = Path(PATHS["outputs_metrics"])
        self.forecasts_path: Path = Path(PATHS["outputs_forecasts"])
    
    def generate_final_summary(self) -> str:
        """Generate final summary markdown report.
        
        Returns:
            Generated markdown content
        """
        lines = []
        
        # Header
        lines.append(f"# {PROJECT_NAME} - Final Summary Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Version:** {PROJECT_VERSION}\n")
        
        lines.append("---\n")
        
        # Task 1: Stock Selection
        lines.append("## Task 1: Stock Selection\n")
        lines.append("**Selected Stocks (8 from NSE):**\n")
        
        for ticker in STOCKS:
            info = STOCK_INFO.get(ticker, {})
            lines.append(f"- **{info.get('name', ticker)}** ({ticker}): {info.get('sector', 'N/A')}")

        selection_file = self.reports_path / "stock_selection.csv"
        if selection_file.exists():
            lines.append(f"\n*Detailed selection metrics: `{selection_file}`*")

        lines.append("\n---\n")
        
        # Task 2: Data Pipeline
        lines.append("## Task 2: Data Pipeline\n")
        lines.append(f"- **Train Period:** {DATE_RANGES['train_start']} to {DATE_RANGES['train_end']}")
        lines.append(f"- **Backtest Period:** {DATE_RANGES['backtest_start']} to {DATE_RANGES['backtest_end']}")
        lines.append(f"- **Live Forecast:** {DATE_RANGES['live_forecast_start']} to {DATE_RANGES['live_forecast_end']}")
        lines.append("- **Data Source:** Yahoo Finance (yfinance)")

        lines.append("\n---\n")
        
        # Task 3: Models Implemented
        lines.append("## Task 3: Forecasting Models\n")
        models = [
            "**Statistical Models (Phase 4A):** ARIMA, SARIMA, Holt-Winters",
            "**ML/DL Models (Phase 4B):** Prophet, LSTM, GRU",
            "**Volatility Model (Phase 4C):** GARCH(1,1)",
            "**Ensemble (Phase 4D):** Weighted average of ARIMA + Prophet + LSTM",
        ]
        for m in models:
            lines.append(f"- {m}")

        # Model comparison
        comparison_file = self.metrics_path / "model_comparison.csv"
        if comparison_file.exists():
            lines.append(f"\n*Model comparison: `{comparison_file}`*")

        lines.append("\n---\n")
        
        # Task 4: Evaluation Metrics
        lines.append("## Task 4: Evaluation Metrics\n")
        lines.append("All models evaluated on:")
        lines.append("- RMSE (Root Mean Squared Error)")
        lines.append("- MAE (Mean Absolute Error)")
        lines.append("- MAPE (Mean Absolute Percentage Error)")
        lines.append("- Directional Accuracy (%)")
        
        lines.append("\n---\n")
        
        # Task 5: Portfolio Optimization
        lines.append("## Task 5: Portfolio Optimization\n")
        lines.append("**Strategies Implemented:**")
        lines.append("- **Strategy A:** Forecast-guided (rank by predicted return)")
        lines.append("- **Strategy B:** Volatility-aware (inverse-volatility weights)")
        lines.append("- **Strategy C:** Correlation-based (diversification penalty)")
        lines.append("- **Strategy D:** Sector momentum rotation")
        lines.append("\n**Combined:** A(60%) + B(40%) | Total Capital: ₹10,00,000")
        
        allocation_file = self.reports_path / "portfolio_allocation.csv"
        if allocation_file.exists():
            lines.append(f"\n*Allocation details: `{allocation_file}`*")
        
        lines.append("\n---\n")
        
        # Task 6: Backtest Results
        lines.append("## Task 6: Backtest Results (Jul-Dec 2025)\n")
        
        perf_file = self.reports_path / "portfolio_summary.csv"
        if perf_file.exists():
            lines.append(f"*Performance summary: `{perf_file}`*")
        
        lines.append("\n---\n")
        
        # Task 7: Live Forecasts
        lines.append("## Task 7: Live Forecasts (May 11-15, 2026)\n")
        
        live_file = self.forecasts_path / "live_forecasts_may2026.csv"
        if live_file.exists():
            lines.append(f"*Live forecasts: `{live_file}`*")
        
        vol_file = self.forecasts_path / "garch_volatility_may2026.csv"
        if vol_file.exists():
            lines.append(f"*Volatility forecasts: `{vol_file}`*")
        
        lines.append("\n---\n")
        
        # Task 8: Outputs
        lines.append("## Task 8: Generated Outputs\n")
        lines.append("### Forecasts")
        lines.append("- `live_forecasts_may2026.csv` - 5-day price forecasts")
        lines.append("- `garch_volatility_may2026.csv` - Conditional volatility")
        lines.append("\n### Metrics")
        lines.append("- `model_comparison.csv` - All models vs all metrics")
        lines.append("- `ensemble_weights.csv` - Inverse-MAPE weights")
        lines.append("\n### Reports")
        lines.append("- `stock_selection.csv` - Selection justification")
        lines.append("- `volatility_trend_summary.csv` - Trend analysis")
        lines.append("- `portfolio_allocation.csv` - Final allocation")
        lines.append("- `portfolio_performance.csv` - Backtest P&L")
        lines.append("- `live_vs_predicted.csv` - Post-trading comparison")
        
        lines.append("\n---\n")
        
        # Footer
        lines.append("## Appendix: File Structure\n")
        lines.append("```")
        lines.append("tsa_capstone/")
        lines.append("├── data/raw/           # yfinance downloads")
        lines.append("├── data/processed/     # Scaled, differenced datasets")
        lines.append("├── models/             # Serialized models (.pkl, .keras)")
        lines.append("├── outputs/forecasts/  # All forecasts (CSV)")
        lines.append("├── outputs/metrics/    # Evaluation metrics (CSV)")
        lines.append("├── outputs/plots/      # Visualization images (PNG)")
        lines.append("├── outputs/reports/    # Summary tables (CSV, MD)")
        lines.append("├── dashboard/          # Streamlit application")
        lines.append("└── notebooks/          # Submission notebook")
        lines.append("```")
        
        # Write to file
        content = "\n".join(lines)
        out_path = self.reports_path / "final_summary.md"
        out_path.write_text(content, encoding="utf-8")
        
        logger.info(f"Final summary saved to {out_path}")
        return content
    
    def export_charts(self) -> None:
        """Trigger chart generation via visualizer."""
        from visualization.plots import ForecastVisualizer
        
        visualizer = ForecastVisualizer()
        visualizer.save_all_charts()
        
        logger.info("All charts exported to outputs/plots/")
