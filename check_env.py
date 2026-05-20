"""Environment validation script for TSA Capstone 2026.

Validates all required imports and prints OK/FAIL for each library.
Run this after setting up the virtual environment and installing requirements.
"""

import sys
from typing import List, Tuple


def check_imports() -> List[Tuple[str, bool, str]]:
    """Check all required imports and return status for each.
    
    Returns:
        List of tuples (library_name, success_status, message)
    """
    results: List[Tuple[str, bool, str]] = []
    
    # Core dependencies
    libraries = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("statsmodels", "statsmodels"),
        ("pmdarima", "pmdarima"),
        ("prophet", "prophet"),
        ("arch", "arch"),
        ("tensorflow", "tensorflow"),
        ("keras", "keras"),
        ("yfinance", "yfinance"),
        ("plotly", "plotly"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("streamlit", "streamlit"),
        ("nbformat", "nbformat"),
        ("yaml", "pyyaml"),
        ("holidays", "holidays"),
    ]
    
    for module_name, package_name in libraries:
        try:
            __import__(module_name)
            results.append((package_name, True, "OK"))
        except ImportError as e:
            results.append((package_name, False, f"FAIL: {str(e)}"))
        except Exception as e:
            results.append((package_name, False, f"ERROR: {str(e)}"))
    
    return results


def print_results(results: List[Tuple[str, bool, str]]) -> None:
    """Print formatted results table."""
    print("=" * 60)
    print("TSA CAPSTONE 2026 - ENVIRONMENT VALIDATION")
    print("=" * 60)
    print()
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, message in results:
        status = "✓ OK" if success else "✗ FAIL"
        print(f"{name:20} {status}")
        if not success:
            print(f"  → {message}")
    
    print()
    print("-" * 60)
    print(f"RESULT: {passed}/{total} libraries imported successfully")
    print("-" * 60)
    
    if passed == total:
        print("✓ Environment ready for Phase 2")
        sys.exit(0)
    else:
        print("✗ Environment issues detected. Fix imports before proceeding.")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    print("Validating imports...\n")
    results = check_imports()
    print_results(results)


if __name__ == "__main__":
    main()
