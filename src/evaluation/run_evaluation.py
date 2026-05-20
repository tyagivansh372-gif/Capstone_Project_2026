#!/usr/bin/env python3
"""Phase 7 Evaluation Runner.

Executes full evaluation pipeline:
1. Backtest metrics computation
2. Live metrics computation (with pending status)
3. Portfolio performance evaluation
4. Model comparison report generation
"""

import sys
sys.path.insert(0, 'src')

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from evaluation.evaluator import PortfolioEvaluator


def main():
    """Run complete Phase 7 evaluation pipeline."""
    logger.info("="*70)
    logger.info("PHASE 7: EVALUATION FRAMEWORK")
    logger.info("="*70)
    
    # Initialize evaluator
    evaluator = PortfolioEvaluator()
    
    # Step 1: Compute all backtest metrics
    logger.info("\n[1] Computing backtest metrics for all stock-model pairs...")
    backtest_df = evaluator.compute_all_backtest_metrics()
    backtest_pairs = len(backtest_df) if not backtest_df.empty else 0
    logger.info(f"    Computed: {backtest_pairs} stock-model pairs")
    
    # Step 2: Compute live metrics (may have pending status)
    logger.info("\n[2] Computing live metrics for May 14-15, 2026...")
    live_df = evaluator.compute_all_live_metrics()
    if not live_df.empty:
        complete_count = len(live_df[live_df['status'] == 'complete'])
        pending_count = len(live_df[live_df['status'] == 'pending'])
        logger.info(f"    Complete: {complete_count} rows")
        logger.info(f"    Pending: {pending_count} rows (awaiting actuals)")
    else:
        complete_count = 0
        pending_count = 0
        logger.info("    WARNING: No live metrics computed")
    
    # Step 3: Compute portfolio performance
    logger.info("\n[3] Computing portfolio performance...")
    portfolio_summary = evaluator.compute_portfolio_performance()
    
    portfolio_status = portfolio_summary.get('status', 'unknown')
    total_return = portfolio_summary.get('total_return_pct', None)
    
    if portfolio_status == 'pending':
        logger.info("    Status: PENDING (awaiting actuals)")
    elif portfolio_status == 'partial':
        stocks_eval = portfolio_summary.get('stocks_evaluated', 0)
        total_stocks = portfolio_summary.get('total_stocks', 8)
        logger.info(f"    Status: PARTIAL ({stocks_eval}/{total_stocks} stocks)")
        if total_return is not None:
            logger.info(f"    Return: {total_return:.2f}%")
    elif portfolio_status == 'complete':
        logger.info("    Status: COMPLETE")
        logger.info(f"    Return: {total_return:.2f}%")
        best = portfolio_summary.get('best_performing_stock', 'N/A')
        worst = portfolio_summary.get('worst_performing_stock', 'N/A')
        logger.info(f"    Best: {best}")
        logger.info(f"    Worst: {worst}")
    else:
        logger.info(f"    Status: {portfolio_status.upper()}")
    
    # Step 4: Generate model comparison report
    logger.info("\n[4] Generating model comparison report...")
    report = evaluator.generate_model_comparison_report()
    if report:
        logger.info("    Report saved to: outputs/reports/model_comparison_report.md")
    else:
        logger.info("    WARNING: Report generation failed")
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Backtest metrics: [{backtest_pairs}] model-stock pairs computed")
    logger.info(f"Live metrics: [{complete_count}] populated, [{pending_count}] pending")
    
    if portfolio_status == 'pending':
        logger.info("Portfolio performance: pending (awaiting May 15 close)")
    else:
        return_str = f"{total_return:.2f}%" if total_return else "N/A"
        logger.info(f"Portfolio performance: {return_str} return")
    
    logger.info(f"Model comparison report: {'saved' if report else 'FAILED'}")
    
    logger.info("\nAll outputs:")
    logger.info("  - outputs/metrics/model_comparison.csv")
    logger.info("  - outputs/reports/live_vs_predicted.csv")
    logger.info("  - outputs/reports/portfolio_performance.csv")
    logger.info("  - outputs/reports/model_comparison_report.md")
    
    logger.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
