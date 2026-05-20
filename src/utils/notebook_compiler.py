"""Notebook compiler module.

Programmatically generates the submission notebook using nbformat.
Stitches together markdown cells, code cells from src modules, and outputs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from config.config import PATHS, PROJECT_NAME

logger = logging.getLogger(__name__)


class NotebookCompiler:
    """Compiles the final submission notebook.
    
    Attributes:
        cells: List of notebook cells
    """
    
    def __init__(self) -> None:
        """Initialize compiler."""
        self.cells: List[Dict[str, Any]] = []
        self.src_path: Path = Path("src")
        self.notebooks_path: Path = Path(PATHS["notebooks"])
        self.notebooks_path.mkdir(parents=True, exist_ok=True)
    
    def _add_header(self) -> None:
        """Add notebook header and title."""
        title = f"""# {PROJECT_NAME}
## NSE Time Series Forecasting & Portfolio Optimization

**Submitted to:** Consulting & Analytics Club, IIT Guwahati  
**Date:** May 2026

---

### Project Overview

This notebook presents a complete time series forecasting and portfolio optimization pipeline covering:

1. **Stock Selection** - 8 NSE stocks across 7 sectors
2. **Data Pipeline** - Historical data (2021-2025) + Extended (2021-May 2026)
3. **Forecasting Models** - ARIMA, SARIMA, Holt-Winters, Prophet, LSTM, GRU, GARCH
4. **Ensemble Methods** - Weighted combination for final predictions
5. **Portfolio Optimization** - Multi-strategy allocation framework
6. **Backtesting & Evaluation** - Jul-Dec 2025 validation
7. **Live Forecasts** - May 11-15, 2026 predictions for StockGro trading

---
"""
        self.cells.append(new_markdown_cell(title))
    
    def _add_section(self, title: str, description: str) -> None:
        """Add a section header."""
        cell = f"""## {title}

{description}
"""
        self.cells.append(new_markdown_cell(cell))
    
    def _add_code_from_file(self, filepath: Path, description: str = "") -> None:
        """Add code cell from file content."""
        if filepath.exists():
            code = filepath.read_text(encoding="utf-8")
            
            if description:
                self.cells.append(new_markdown_cell(f"**{description}**"))
            
            self.cells.append(new_code_cell(code))
        else:
            self.cells.append(new_code_cell(f"# File not found: {filepath}"))
    
    def _add_task_1_stock_selection(self) -> None:
        """Add Task 1: Stock Selection."""
        self._add_section(
            "Task 1: Stock Selection",
            """Selected 8 stocks from NSE for sector diversification:

| Stock | Ticker | Sector |
|-------|--------|--------|
| Reliance Industries | RELIANCE.NS | Energy |
| HDFC Bank | HDFCBANK.NS | Banking |
| Infosys | INFY.NS | IT |
| Sun Pharma | SUNPHARMA.NS | Pharma |
| Maruti Suzuki | MARUTI.NS | Auto |
| ITC | ITC.NS | FMCG |
| Tata Steel | TATASTEEL.NS | Metals |
| Bajaj Finance | BAJFINANCE.NS | NBFC |
"""
        )
        self._add_code_from_file(
            self.src_path / "data_fetching" / "fetcher.py",
            "Data fetching module:"
        )
    
    def _add_task_2_data_pipeline(self) -> None:
        """Add Task 2: Data Pipeline."""
        self._add_section(
            "Task 2: Data Pipeline",
            """Historical data fetched via yfinance:
- Train: Jan 2021 - Jun 2025
- Backtest: Jul 2025 - Dec 2025
- Extended: Jan 2021 - May 10, 2026 (for live forecasts)
"""
        )
        self._add_code_from_file(
            self.src_path / "preprocessing" / "preprocessor.py",
            "Preprocessing module (missing values, scaling, stationarity tests):"
        )
    
    def _add_task_3_models(self) -> None:
        """Add Task 3: Forecasting Models."""
        self._add_section(
            "Task 3: Forecasting Models",
            """Implemented models:

**Statistical Models:**
- ARIMA (pmdarima auto-tuned)
- SARIMA (weekly seasonality m=5)
- Holt-Winters Exponential Smoothing

**ML/DL Models:**
- Prophet (with Indian holidays)
- LSTM (2 layers, dropout 0.2)
- GRU (mirrors LSTM)

**Volatility:**
- GARCH(1,1)
"""
        )
        
        models = [
            ("forecasting/arima.py", "ARIMA implementation"),
            ("forecasting/sarima.py", "SARIMA implementation"),
            ("forecasting/holt_winters.py", "Holt-Winters implementation"),
            ("forecasting/prophet_model.py", "Prophet implementation"),
            ("forecasting/lstm.py", "LSTM implementation"),
            ("forecasting/gru.py", "GRU implementation"),
            ("forecasting/garch.py", "GARCH implementation"),
        ]
        
        for filepath, desc in models:
            self._add_code_from_file(self.src_path / filepath, desc)
    
    def _add_task_4_ensemble(self) -> None:
        """Add Task 4: Ensemble."""
        self._add_section(
            "Task 4: Ensemble Model",
            """Weighted ensemble combining ARIMA + Prophet + LSTM.
Weights derived from inverse-MAPE on backtest set."""
        )
        self._add_code_from_file(
            self.src_path / "forecasting" / "ensemble.py",
            "Ensemble implementation:"
        )
    
    def _add_task_5_portfolio(self) -> None:
        """Add Task 5: Portfolio Optimization."""
        self._add_section(
            "Task 5: Portfolio Optimization",
            """Four strategies implemented:
- A: Forecast-guided (60% weight)
- B: Volatility-aware (40% weight)
- C: Correlation-based
- D: Sector momentum

Final: Combined A + B = ₹10,00,000 allocation"""
        )
        self._add_code_from_file(
            self.src_path / "portfolio" / "optimizer.py",
            "Portfolio optimizer:"
        )
    
    def _add_task_6_evaluation(self) -> None:
        """Add Task 6: Evaluation."""
        self._add_section(
            "Task 6: Evaluation Framework",
            """Metrics: RMSE, MAE, MAPE, Directional Accuracy  
Backtest: Jul-Dec 2025 portfolio P&L simulation"""
        )
        self._add_code_from_file(
            self.src_path / "evaluation" / "evaluator.py",
            "Evaluator implementation:"
        )
    
    def _add_task_7_volatility(self) -> None:
        """Add Task 7: Volatility Analysis."""
        self._add_section(
            "Task 7: Volatility & Trend Analysis",
            """Rolling 30/90-day volatility + STL decomposition + trend classification"""
        )
        self._add_code_from_file(
            self.src_path / "volatility" / "analyzer.py",
            "Volatility analyzer:"
        )
    
    def _add_task_8_outputs(self) -> None:
        """Add Task 8: Outputs."""
        self._add_section(
            "Task 8: Outputs & Results",
            """### Generated Files:

**Forecasts:**
- `live_forecasts_may2026.csv` - 5-day price forecasts (May 11-15)
- `garch_volatility_may2026.csv` - Conditional volatility

**Metrics:**
- `model_comparison.csv` - All models evaluation
- `ensemble_weights.csv` - Inverse-MAPE weights

**Reports:**
- `portfolio_allocation.csv` - Final ₹10L allocation
- `portfolio_performance.csv` - Backtest results
- `volatility_trend_summary.csv` - Trend analysis

**Dashboard:**
- Run: `streamlit run dashboard/app.py`
"""
        )
    
    def _add_conclusion(self) -> None:
        """Add conclusion cell."""
        conclusion = """## Conclusion

This capstone project demonstrates a production-quality time series forecasting pipeline:

1. **Complete Data Pipeline:** From raw yfinance data to model-ready datasets
2. **Multiple Models:** Statistical, ML, DL, and ensemble approaches
3. **Robust Evaluation:** Backtesting on 6 months of holdout data
4. **Portfolio Optimization:** Multi-strategy allocation with risk management
5. **Professional Outputs:** Forecasts, visualizations, and interactive dashboard

**Live Trading:** Use forecasts in `live_forecasts_may2026.csv` for StockGro May 11-15, 2026 trading window.

---

*End of Notebook*
"""
        self.cells.append(new_markdown_cell(conclusion))
    
    def compile_notebook(self) -> Path:
        """Compile all cells into a notebook and save.
        
        Returns:
            Path to saved notebook
        """
        logger.info("Starting notebook compilation...")
        
        # Build notebook structure
        self._add_header()
        self._add_task_1_stock_selection()
        self._add_task_2_data_pipeline()
        self._add_task_3_models()
        self._add_task_4_ensemble()
        self._add_task_5_portfolio()
        self._add_task_6_evaluation()
        self._add_task_7_volatility()
        self._add_task_8_outputs()
        self._add_conclusion()
        
        # Create notebook
        nb = new_notebook(cells=self.cells)
        
        # Add metadata
        nb.metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        }
        
        # Save
        output_path = self.notebooks_path / "capstone_submission.ipynb"
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        
        logger.info(f"Notebook saved to {output_path}")
        return output_path
    
    def run(self) -> None:
        """Execute full compilation."""
        self.compile_notebook()


if __name__ == "__main__":
    compiler = NotebookCompiler()
    compiler.run()
