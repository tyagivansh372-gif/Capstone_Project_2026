"""Main orchestration entry point for TSA Capstone 2026.

Coordinates the full pipeline from data fetching to final report generation.
Run: python main.py
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.config import LOGGING_CONFIG


def setup_logging() -> logging.Logger:
    """Configure logging for the pipeline.
    
    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["level"]),
        format=LOGGING_CONFIG["format"],
        datefmt=LOGGING_CONFIG["date_format"],
    )
    return logging.getLogger(__name__)


def run_phase_1() -> None:
    """Phase 1: Foundation - Environment validation."""
    logger = logging.getLogger(__name__)
    logger.info("Phase 1: Environment validation")
    # Import validation handled by check_env.py separately


def run_phase_2() -> None:
    """Phase 2: Data Pipeline - Fetch and validate stock data."""
    logger = logging.getLogger(__name__)
    logger.info("Phase 2: Data fetching and validation")
    from data_fetching.fetcher import DataFetcher
    
    fetcher = DataFetcher()
    fetcher.fetch_all()
    fetcher.validate_data()
    fetcher.generate_selection_report()


def run_phase_3() -> None:
    """Phase 3: Preprocessing - Clean, scale, and split data."""
    logger = logging.getLogger(__name__)
    logger.info("Phase 3: Data preprocessing")
    from preprocessing.preprocessor import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    preprocessor.load_raw_data()
    preprocessor.handle_missing_values()
    preprocessor.test_stationarity()
    preprocessor.apply_scaling()
    preprocessor.save_processed_data()


def run_phase_4a() -> None:
    """Phase 4A: Statistical Forecasting (ARIMA, SARIMA, Holt-Winters)."""
    logger = logging.getLogger(__name__)
    logger.info("Phase 4A: Statistical models")
    from forecasting.arima import ArimaModel
    from forecasting.sarima import SarimaModel
    from forecasting.holt_winters import HoltWintersModel
    
    # ARIMA
    arima = ArimaModel()
    arima.train_all()
    arima.backtest_all()
    arima.save_metrics()
    
    # SARIMA
    sarima = SarimaModel()
    sarima.train_all()
    sarima.backtest_all()
    sarima.save_metrics()
    
    # Holt-Winters
    hw = HoltWintersModel()
    hw.train_all()
    hw.backtest_all()
    hw.save_metrics()


def run_phase_4b() -> None:
    """Phase 4B: ML/DL Forecasting (Prophet, LSTM, GRU)."""
    logger = logging.getLogger(__name__)
    logger.info("Phase 4B: ML/DL models")
    from forecasting.prophet_model import ProphetModel
    from forecasting.lstm import LSTMModel
    from forecasting.gru import GRUModel
    
    # Prophet
    prophet = ProphetModel()
    prophet.train_all()
    prophet.backtest_all()
    prophet.save_metrics()
    
    # LSTM
    lstm = LSTMModel()
    lstm.prepare_sequences()
    lstm.train_all()
    lstm.backtest_all()
    lstm.save_metrics()
    
    # GRU
    gru = GRUModel()
    gru.prepare_sequences()
    gru.train_all()
    gru.backtest_all()
    gru.save_metrics()


def run_phase_4c() -> None:
    """Phase 4C: GARCH Volatility Model."""
    logger = logging.getLogger(__name__)
    logger.info("Phase 4C: GARCH volatility")
    from forecasting.garch import GarchModel
    
    garch = GarchModel()
    garch.compute_log_returns()
    garch.fit_all()
    garch.forecast_volatility()
    garch.save_results()


def run_phase_4d() -> None:
    """Phase 4D: Ensemble + Live Forecasts."""
    logger = logging.getLogger(__name__)
    logger.info("Phase 4D: Ensemble and live forecasts")
    from forecasting.ensemble import EnsembleModel
    from data_fetching.fetcher import DataFetcher
    
    # Fetch extended data
    fetcher = DataFetcher()
    fetcher.fetch_extended_data()
    
    # Build ensemble
    ensemble = EnsembleModel()
    ensemble.compute_weights()
    ensemble.retrain_all_models()
    ensemble.generate_live_forecasts()
    ensemble.save_results()


def run_phase_5() -> None:
    """Phase 5: Volatility & Trend Analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Phase 5: Volatility and trend analysis")
    from volatility.analyzer import VolatilityAnalyzer
    
    analyzer = VolatilityAnalyzer()
    analyzer.compute_rolling_volatility()
    analyzer.stl_decomposition()
    analyzer.classify_trends()
    analyzer.save_summary()


def run_phase_6() -> None:
    """Phase 6: Portfolio Optimization."""
    logger = logging.getLogger(__name__)
    logger.info("Phase 6: Portfolio optimization")
    from portfolio.optimizer import PortfolioOptimizer
    
    optimizer = PortfolioOptimizer()
    optimizer.load_forecasts()
    optimizer.strategy_a_forecast_guided()
    optimizer.strategy_b_volatility_aware()
    optimizer.strategy_c_correlation_based()
    optimizer.strategy_d_sector_momentum()
    optimizer.combine_strategies()
    optimizer.save_allocation()


def run_phase_7() -> None:
    """Phase 7: Evaluation Framework."""
    logger = logging.getLogger(__name__)
    logger.info("Phase 7: Evaluation framework")
    from evaluation.evaluator import ModelEvaluator
    
    evaluator = ModelEvaluator()
    evaluator.load_predictions()
    evaluator.compute_metrics()
    evaluator.portfolio_backtest()
    evaluator.create_live_actuals_placeholder()
    evaluator.save_results()


def run_phase_8() -> None:
    """Phase 8: Streamlit Dashboard.
    
    Note: Dashboard is run separately via `streamlit run dashboard/app.py`
    """
    logger = logging.getLogger(__name__)
    logger.info("Phase 8: Dashboard available at `streamlit run dashboard/app.py`")


def run_phase_9() -> None:
    """Phase 9: Reporting & Export."""
    logger = logging.getLogger(__name__)
    logger.info("Phase 9: Final reporting")
    from utils.reporter import ReportGenerator
    from utils.notebook_compiler import NotebookCompiler
    
    # Generate markdown summary
    reporter = ReportGenerator()
    reporter.generate_final_summary()
    reporter.export_charts()
    
    # Compile submission notebook
    compiler = NotebookCompiler()
    compiler.compile_notebook()


def main(phase: Optional[str] = None) -> None:
    """Main entry point for the capstone pipeline.
    
    Args:
        phase: Optional phase to run (1-9, 4a, 4b, 4c, 4d). 
               If None, runs all phases.
    """
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TSA CAPSTONE 2026 - STARTING PIPELINE")
    logger.info("=" * 60)
    
    phase_runners = {
        "1": run_phase_1,
        "2": run_phase_2,
        "3": run_phase_3,
        "4a": run_phase_4a,
        "4b": run_phase_4b,
        "4c": run_phase_4c,
        "4d": run_phase_4d,
        "5": run_phase_5,
        "6": run_phase_6,
        "7": run_phase_7,
        "8": run_phase_8,
        "9": run_phase_9,
    }
    
    if phase:
        if phase.lower() in phase_runners:
            logger.info(f"Running Phase {phase}")
            phase_runners[phase.lower()]()
        else:
            logger.error(f"Unknown phase: {phase}")
            sys.exit(1)
    else:
        # Run all phases
        for p in ["2", "3", "4a", "4b", "4c", "4d", "5", "6", "7", "9"]:
            phase_runners[p]()
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TSA Capstone 2026 Pipeline")
    parser.add_argument(
        "--phase", 
        type=str, 
        help="Phase to run (1, 2, 3, 4a, 4b, 4c, 4d, 5, 6, 7, 8, 9)"
    )
    args = parser.parse_args()
    
    main(phase=args.phase)
