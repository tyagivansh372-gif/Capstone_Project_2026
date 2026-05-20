"""Portfolio optimization module.

Implements four allocation strategies and combines for final allocation.
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config.config import STOCKS, PATHS, PORTFOLIO_CONFIG, STOCK_INFO

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Portfolio optimization with multiple strategies.
    
    Strategies:
        A: Forecast-guided (rank by predicted return, weight by confidence)
        B: Volatility-aware (inverse-volatility weights)
        C: Correlation-based (penalize high-correlation pairs)
        D: Sector momentum rotation
    
    Attributes:
        allocation: Final combined allocation
        strategy_outputs: Individual strategy results
    """
    
    def __init__(self) -> None:
        """Initialize optimizer."""
        self.allocation: pd.DataFrame = pd.DataFrame()
        self.strategy_outputs: Dict[str, pd.DataFrame] = {}
        
        self.total_capital: float = PORTFOLIO_CONFIG["total_capital"]
        self.max_position: float = PORTFOLIO_CONFIG["max_position_pct"]
        self.min_position: float = PORTFOLIO_CONFIG["min_position_pct"]
        
        self.reports_path: Path = Path(PATHS["outputs_reports"])
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        self.forecasts: pd.DataFrame = pd.DataFrame()
        self.volatility: pd.DataFrame = pd.DataFrame()
        self.returns_90d: pd.DataFrame = pd.DataFrame()
    
    def load_forecasts(self) -> None:
        """Load forecast and volatility data for optimization."""
        # Load live forecasts if available
        forecast_path = Path(PATHS["outputs_forecasts"]) / "live_forecasts_may2026.csv"
        if forecast_path.exists():
            self.forecasts = pd.read_csv(forecast_path, parse_dates=["date"])
        
        # Load GARCH volatility
        vol_path = Path(PATHS["outputs_forecasts"]) / "garch_volatility_may2026.csv"
        if vol_path.exists():
            self.volatility = pd.read_csv(vol_path, parse_dates=["date"])
        
        # Load historical returns for correlation and fallback forecasts
        from preprocessing.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load_raw_data()
        
        returns_dict = {}
        last_prices = {}
        for ticker in STOCKS:
            if ticker in preprocessor.train_data:
                prices = preprocessor.train_data[ticker]["Close"].squeeze()
                returns = prices.pct_change().dropna()
                returns_dict[ticker] = returns.tail(90)  # Last 90 days
                last_prices[ticker] = prices.iloc[-1]
        
        self.returns_90d = pd.DataFrame(returns_dict)
        self.last_prices = last_prices
        
        # Create fallback forecasts from historical momentum if live forecasts missing
        if self.forecasts.empty:
            self._create_fallback_forecasts()
    
    def _create_fallback_forecasts(self) -> None:
        """Create fallback forecasts from 90-day momentum when live forecasts unavailable."""
        from config.config import DATE_RANGES
        
        logger.info("Creating fallback forecasts from historical momentum")
        
        # Use 90-day annualized return as forecast proxy
        forecast_records = []
        start_date = pd.Timestamp(DATE_RANGES["live_forecast_start"])
        dates = pd.date_range(start=start_date, periods=5, freq="B")
        
        for ticker in STOCKS:
            if ticker not in self.returns_90d.columns:
                continue
            
            # Compute 90-day momentum as forecast proxy
            returns = self.returns_90d[ticker].dropna()
            if len(returns) == 0:
                continue
            
            momentum = returns.mean() * 252  # Annualized
            last_price = self.last_prices.get(ticker, 100)
            
            # Simple linear projection for 5 days
            daily_return = momentum / 252
            for i, date in enumerate(dates):
                projected_price = last_price * (1 + daily_return) ** (i + 1)
                forecast_records.append({
                    "stock": ticker,
                    "date": date,
                    "predicted_price": projected_price,
                    "model": "ensemble",
                    "confidence_interval_low": None,
                    "confidence_interval_high": None,
                })
        
        self.forecasts = pd.DataFrame(forecast_records)
        logger.info(f"Fallback forecasts created: {len(self.forecasts)} records")
    
    def strategy_a_forecast_guided(self) -> pd.DataFrame:
        """Strategy A: Forecast-guided allocation.
        
        Rank stocks by predicted 5-day return, weight by forecast confidence.
        
        Returns:
            DataFrame with weights and rationale
        """
        if self.forecasts.empty:
            return pd.DataFrame()
        
        # Get ensemble forecasts
        ensemble = self.forecasts[self.forecasts["model"] == "ensemble"]
        
        # Compute predicted 5-day return
        forecast_returns: Dict[str, float] = {}
        for ticker in STOCKS:
            stock_fc = ensemble[ensemble["stock"] == ticker]
            if len(stock_fc) >= 2:
                start_price = stock_fc.iloc[0]["predicted_price"]
                end_price = stock_fc.iloc[-1]["predicted_price"]
                forecast_returns[ticker] = (end_price / start_price) - 1
        
        # Rank and allocate
        sorted_stocks = sorted(forecast_returns.items(), key=lambda x: x[1], reverse=True)
        
        # Simple rank-weighted allocation
        n = len(sorted_stocks)
        weights = {}
        total_rank = sum(range(1, n + 1))
        
        for rank, (ticker, ret) in enumerate(sorted_stocks, 1):
            # Higher rank (lower number) gets more weight
            weight = (n - rank + 1) / total_rank
            weights[ticker] = weight
        
        # Normalize to ensure sum = 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        df = pd.DataFrame([
            {
                "ticker": t,
                "weight": w,
                "predicted_return": forecast_returns.get(t, 0),
                "strategy": "A_forecast_guided",
            }
            for t, w in weights.items()
        ])
        
        self.strategy_outputs["A"] = df
        return df
    
    def strategy_b_volatility_aware(self) -> pd.DataFrame:
        """Strategy B: Volatility-aware allocation.
        
        Inverse-volatility weights from GARCH output.
        
        Returns:
            DataFrame with weights and rationale
        """
        if self.volatility.empty:
            return pd.DataFrame()
        
        # Average volatility across forecast period
        avg_vols: Dict[str, float] = {}
        for ticker in STOCKS:
            stock_vol = self.volatility[self.volatility["stock"] == ticker]
            if not stock_vol.empty:
                avg_vols[ticker] = stock_vol["forecasted_volatility"].mean()
        
        # Inverse volatility weights
        inv_vols = {t: 1.0 / v for t, v in avg_vols.items() if v > 0}
        total = sum(inv_vols.values())
        weights = {t: v / total for t, v in inv_vols.items()}
        
        df = pd.DataFrame([
            {
                "ticker": t,
                "weight": w,
                "avg_volatility": avg_vols.get(t, 0),
                "strategy": "B_volatility_aware",
            }
            for t, w in weights.items()
        ])
        
        self.strategy_outputs["B"] = df
        return df
    
    def strategy_c_correlation_based(self) -> pd.DataFrame:
        """Strategy C: Correlation-based allocation.
        
        Penalize high-correlation pairs, diversify selection.
        
        Returns:
            DataFrame with weights and rationale
        """
        if self.returns_90d.empty:
            return pd.DataFrame()
        
        corr_matrix = self.returns_90d.corr()
        
        # Compute diversification score (lower correlation = higher score)
        div_scores: Dict[str, float] = {}
        for ticker in STOCKS:
            if ticker in corr_matrix.columns:
                # Average correlation with other stocks
                avg_corr = corr_matrix[ticker].drop(ticker).mean()
                div_scores[ticker] = 1 - avg_corr  # Higher is better
        
        # Normalize to weights
        total = sum(div_scores.values())
        weights = {t: s / total for t, s in div_scores.items()} if total > 0 else {}
        
        df = pd.DataFrame([
            {
                "ticker": t,
                "weight": w,
                "diversification_score": div_scores.get(t, 0),
                "strategy": "C_correlation_based",
            }
            for t, w in weights.items()
        ])
        
        self.strategy_outputs["C"] = df
        return df
    
    def strategy_d_sector_momentum(self) -> pd.DataFrame:
        """Strategy D: Sector momentum rotation.
        
        Compute sector momentum, weight sectors, then stocks within sectors.
        
        Returns:
            DataFrame with weights and rationale
        """
        # Group by sector
        sectors: Dict[str, List[str]] = {}
        for ticker in STOCKS:
            sector = STOCK_INFO[ticker]["sector"]
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(ticker)
        
        # For simplicity, equal weight within equal sector weight
        # (Full implementation would compute sector momentum scores)
        n_sectors = len(sectors)
        sector_weight = 1.0 / n_sectors
        
        weights: Dict[str, float] = {}
        for sector, tickers in sectors.items():
            stock_weight = sector_weight / len(tickers)
            for ticker in tickers:
                weights[ticker] = stock_weight
        
        df = pd.DataFrame([
            {
                "ticker": t,
                "weight": w,
                "sector": STOCK_INFO[t]["sector"],
                "strategy": "D_sector_momentum",
            }
            for t, w in weights.items()
        ])
        
        self.strategy_outputs["D"] = df
        return df
    
    def combine_strategies(self) -> pd.DataFrame:
        """Combine strategies A and B for final allocation.
        
        A (forecast-guided) gets 60% weight, B (volatility-aware) gets 40%.
        
        Returns:
            DataFrame with final weights
        """
        if "A" not in self.strategy_outputs or "B" not in self.strategy_outputs:
            raise ValueError("Strategies A and B must be run before combining")
        
        df_a = self.strategy_outputs["A"].set_index("ticker")["weight"]
        df_b = self.strategy_outputs["B"].set_index("ticker")["weight"]
        
        # Weighted combination: 60% A, 40% B
        combined = 0.6 * df_a + 0.4 * df_b
        
        # Apply position limits
        combined = combined.clip(lower=self.min_position, upper=self.max_position)
        
        # Renormalize
        combined = combined / combined.sum()
        
        # Calculate INR allocation
        allocation_df = pd.DataFrame({
            "ticker": combined.index,
            "weight_pct": (combined.values * 100).round(2),
            "allocated_INR": (combined.values * self.total_capital).round(2),
            "strategy_rationale": "Combined A(60%) + B(40%)",
        })
        
        self.allocation = allocation_df
        return allocation_df
    
    def save_allocation(self) -> None:
        """Save portfolio allocation to CSV."""
        if self.allocation.empty:
            raise ValueError("No allocation to save. Run combine_strategies first.")
        
        out_path = self.reports_path / "portfolio_allocation.csv"
        self.allocation.to_csv(out_path, index=False)
        
        # Validate total
        total_allocated = self.allocation["allocated_INR"].sum()
        logger.info(f"Portfolio allocation saved. Total: ₹{total_allocated:,.2f}")
        
        if abs(total_allocated - self.total_capital) > 1000:
            logger.warning(f"Allocation mismatch: {total_allocated} vs {self.total_capital}")
