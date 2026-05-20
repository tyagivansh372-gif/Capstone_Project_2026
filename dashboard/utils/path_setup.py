"""Path setup utility for dashboard pages.

Centralizes the repeated path manipulation logic used across all dashboard pages.
"""

import sys
from pathlib import Path


def setup_project_paths() -> None:
    """Configure Python sys.path for dashboard imports.
    
    Adds the dashboard directory, project root, and src directory to sys.path
    to enable imports from config, utils, and other project modules.
    
    This function should be called at the top of every dashboard page file
    before any other imports.
    """
    DASHBOARD_DIR = Path(__file__).parent.parent
    PROJECT_ROOT = DASHBOARD_DIR.parent
    
    sys.path.insert(0, str(DASHBOARD_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
