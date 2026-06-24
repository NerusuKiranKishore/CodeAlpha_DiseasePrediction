"""
main.py
-------
Entry point for the CodeAlpha Disease Prediction project.

Usage
-----
    python main.py                      # Run all 3 diseases
    python main.py --disease heart      # Heart disease only
    python main.py --disease diabetes
    python main.py --disease breast_cancer
    python main.py --no-plots           # Skip plot generation
"""

import argparse
import logging
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.helpers import setup_logging, ensure_dirs, print_summary_table
from src.data.data_loader import load_dataset
from src.features.feature_engineering import prepare_features
from src.models.trainer import train_and_evaluate
from src.visualization.plots import generate_all_plots

DISEASES = ["heart", "diabetes", "breast_cancer"]
OUTPUT_DIR = "outputs"


def run_pipeline(disease: str, generate_plots: bool = True):
    """Full pipeline for one disease: load → engineer → train → visualize."""
    logger = logging.getLogger(__name__)

    logger.info("\n%s\nRunning pipeline for: %s\n%s", "─" * 60, disease.upper(), "─" * 60)

    # 1. Load data
    df, target_col = load_dataset(disease, data_dir="data")

    # 2. Feature engineering
    X, y, preprocessor = prepare_features(disease, df, target_col)

    # 3. Train & evaluate
    disease_results = train_and_evaluate(
        disease=disease,
        X=X,
        y=y,
        preprocessor=preprocessor,
        output_dir=OUTPUT_DIR,
    )

    # 4. Visualize
    if generate_plots:
        generate_all_plots(disease_results, y, output_dir=OUTPUT_DIR)

    return disease_results


def main():
    parser = argparse.ArgumentParser(
        description="CodeAlpha Disease Prediction — ML Pipeline"
    )
    parser.add_argument(
        "--disease",
        choices=DISEASES + ["all"],
        default="all",
        help="Which disease to run (default: all)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (faster)",
    )
    args = parser.parse_args()

    setup_logging(
        level="INFO",
        log_file=os.path.join(OUTPUT_DIR, "run.log"),
    )
    ensure_dirs(
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "models"),
        os.path.join(OUTPUT_DIR, "plots"),
        os.path.join(OUTPUT_DIR, "reports"),
        "data",
    )

    diseases_to_run = DISEASES if args.disease == "all" else [args.disease]
    generate_plots = not args.no_plots

    all_results = []
    for disease in diseases_to_run:
        try:
            result = run_pipeline(disease, generate_plots=generate_plots)
            all_results.append(result)
        except Exception as e:
            logging.getLogger(__name__).error(
                "Pipeline failed for %s: %s", disease, e, exc_info=True
            )

    if all_results:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
