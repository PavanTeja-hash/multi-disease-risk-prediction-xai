"""
Run the whole pipeline end to end.

    python main.py            # every phase, in order
    python main.py --quick    # skip the slow phases (neural net, SHAP, insights)

Each phase is also runnable on its own - `python src/eda.py`, and so on - which
is the easier way to work on one piece at a time.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config  # noqa: E402


PHASES = [
    ("2", "Data collection", "data_loader", False),
    ("3", "Data cleaning", "data_cleaning", False),
    ("4", "Exploratory data analysis", "eda", False),
    ("5", "Feature engineering", "feature_engineering", False),
    ("6", "Machine learning models", "train_models", False),
    ("6b", "Neural network", "train_neural_network", True),
    ("7", "Model evaluation", "evaluate", False),
    ("8", "Explainable AI", "explainability", True),
    ("8b", "Plain-English reports", "report_generator", False),
    ("9", "Business insights", "insights", True),
    ("-", "Package for deployment", "deploy", False),
]


def run_phase(number: str, title: str, module_name: str) -> float:
    print("\n" + "=" * 70)
    print(f"PHASE {number}: {title.upper()}")
    print("=" * 70)

    start = time.time()
    module = __import__(module_name)
    module.main()
    elapsed = time.time() - start

    print(f"\n[phase {number} finished in {elapsed:.1f}s]")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="skip the slow phases (neural network, SHAP, insights)",
    )
    args = parser.parse_args()

    config.ensure_directories()

    print("=" * 70)
    print("MULTI-DISEASE RISK PREDICTION WITH EXPLAINABLE AI")
    print("=" * 70)
    if args.quick:
        print("Running in quick mode - slow phases will be skipped.")

    total = 0.0
    for number, title, module_name, is_slow in PHASES:
        if args.quick and is_slow:
            print(f"\n[skipping phase {number}: {title} (--quick)]")
            continue
        total += run_phase(number, title, module_name)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE in {total / 60:.1f} minutes")
    print("=" * 70)
    print(f"  figures  -> {config.FIGURES_DIR.relative_to(config.PROJECT_ROOT)}/")
    print(f"  models   -> {config.MODELS_DIR.relative_to(config.PROJECT_ROOT)}/")
    print(f"  reports  -> {config.REPORTS_DIR.relative_to(config.PROJECT_ROOT)}/")
    print("\n  Launch the interactive demo with:  streamlit run app.py")


if __name__ == "__main__":
    main()
