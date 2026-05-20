"""Phase 9: Notebook Compiler and Final Report.

Runs the three pending Phase 9 modules in sequence:
  1. ForecastVisualizer  — saves all charts to outputs/plots/
  2. ReportGenerator     — writes final_summary.md to outputs/reports/
  3. NotebookCompiler    — stitches capstone_submission.ipynb into notebooks/

Usage:
    python run_phase_9.py
"""

import logging
import sys
from pathlib import Path

# Bootstrap: add src/ to sys.path so all local modules are importable
sys.path.insert(0, "src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEP = "=" * 60


def run_visualizer() -> None:
    """Save all standard charts to outputs/plots/."""
    from visualization.plots import ForecastVisualizer

    logger.info("Initializing ForecastVisualizer...")
    visualizer = ForecastVisualizer()
    visualizer.save_all_charts()
    logger.info("Charts saved to outputs/plots/")


def run_reporter() -> str:
    """Generate final_summary.md and return its content."""
    from utils.reporter import ReportGenerator

    logger.info("Initializing ReportGenerator...")
    generator = ReportGenerator()
    content = generator.generate_final_summary()
    logger.info("Final summary saved to outputs/reports/final_summary.md")
    return content


def run_notebook_compiler() -> Path:
    """Compile capstone_submission.ipynb and return its path."""
    from utils.notebook_compiler import NotebookCompiler

    logger.info("Initializing NotebookCompiler...")
    compiler = NotebookCompiler()
    path = compiler.compile_notebook()
    logger.info(f"Notebook saved to {path}")
    return path


def main() -> None:
    """Execute Phase 9 pipeline: visualizer → reporter → notebook compiler."""
    print(SEP)
    print("PHASE 9: NOTEBOOK COMPILER & FINAL REPORT")
    print(SEP)

    # Step 1: Charts
    print("\n[1/3] Generating charts...")
    try:
        run_visualizer()
        print("      ✓ Charts complete")
    except Exception as exc:
        logger.warning(
            "Chart generation failed (data may not be available yet): %s", exc
        )
        print(f"      ⚠  Charts skipped — {exc}")

    # Step 2: Final summary report
    print("\n[2/3] Generating final summary report...")
    try:
        run_reporter()
        print("      ✓ Report complete")
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        print(f"      ✗ Report failed — {exc}")

    # Step 3: Notebook
    print("\n[3/3] Compiling submission notebook...")
    try:
        nb_path = run_notebook_compiler()
        print(f"      ✓ Notebook saved → {nb_path}")
    except Exception as exc:
        logger.error("Notebook compilation failed: %s", exc)
        print(f"      ✗ Notebook failed — {exc}")

    print("\n" + SEP)
    print("PHASE 9 COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()
