
"""
Entry point for the EEG seizure classification experimentation framework.

Usage:
    python main.py
    python main.py --config my_config.yaml
    python main.py --config config.yaml --only train
    python main.py --config config.yaml --only eval
    python main.py --config config.yaml --only plots
    python main.py --config config.yaml --only xai
    python main.py --experiment loso_fm   # LOSO seizure prediction with foundation models
"""

import argparse
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from experimentation import Experiment


def main():
    parser = argparse.ArgumentParser(
        description="EEG Seizure Classification Experimentation Framework",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        choices=["train", "eval", "plots", "xai"],
        help="Run only a specific pipeline phase",
    )
    parser.add_argument(
        "--experiment", type=str, default="classic", choices=["classic", "loso_fm"],
        help="classic = config.yaml pipeline; loso_fm = loso_foundation_model_prediction.py",
    )
    args = parser.parse_args()

    if args.experiment == "loso_fm":
        import loso_foundation_model_prediction
        loso_foundation_model_prediction.run()
        return

    exp = Experiment(config_path=args.config)

    if args.only is None:
        exp.run()
    elif args.only == "train":
        exp.load_or_extract_features()
        exp.select_features()
        exp.train()
    elif args.only == "eval":
        exp.load_or_extract_features()
        exp.select_features()
        exp.load_models()
        exp.evaluate_validation()
        exp.evaluate_test()
        exp.save_predictions()
    elif args.only == "plots":
        exp.load_or_extract_features()
        exp.select_features()
        exp.load_models()
        exp.evaluate_test()
        exp.generate_plots()
    elif args.only == "xai":
        exp.load_or_extract_features()
        exp.select_features()
        exp.load_models()
        exp.generate_xai()


if __name__ == "__main__":
    main()
