
"""
Entry point for the EEG seizure classification experimentation framework.

Usage:
    python main.py
    python main.py --config my_config.yaml
    python main.py --config config.yaml --only train
    python main.py --config config.yaml --only eval
    python main.py --config config.yaml --only plots
    python main.py --config config.yaml --only xai
    python main.py --experiment loso_fm       # LOSO seizure prediction with foundation models
    python main.py --experiment lora_tabicl   # LoRA fine-tuning of TabICL (best loso_fm variant)
    python main.py --experiment loso_fm --models tabicl --context 300 --ictal excluded
    python main.py --experiment lora_tabicl --context 300 --ictal excluded
    python main.py --experiment loso_fm --models tabicl --context 300 --ictal excluded --history change_index
    python main.py --experiment xai_tabicl --context 600 --ictal excluded --history trend10
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
        "--experiment", type=str, default="classic", choices=["classic", "loso_fm", "lora_tabicl", "xai_tabicl"],
        help="classic = config.yaml pipeline; loso_fm = loso_foundation_model_prediction.py; "
             "lora_tabicl = finetune_lora_tabicl.py; xai_tabicl = xai_tabicl.py (SHAP, LIME, topomap)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="loso_fm only: models to run (default: all in loso_foundation_model_prediction.MODELS)",
    )
    parser.add_argument(
        "--context", nargs="+", type=int, default=None,
        help="Seconds of EEG seen per prediction, e.g. 300 (lora_tabicl takes one value)",
    )
    parser.add_argument(
        "--ictal", nargs="+", default=None, choices=["excluded", "0", "1"],
        help="How ictal windows are labelled (lora_tabicl takes one value)",
    )
    parser.add_argument(
        "--history", default=None, choices=["stats", "change_index", "trend10"],
        help="How the EEG before each window is summarised (default: stats = change + std, 600 columns)",
    )
    parser.add_argument(
        "--patients", nargs="+", default=None,
        help="xai_tabicl only: explain just these test patients, e.g. chb01 chb05 (default: all folds)",
    )
    args = parser.parse_args()
    ictal = [i if i == "excluded" else int(i) for i in args.ictal] if args.ictal else None

    if args.experiment == "loso_fm":
        import loso_foundation_model_prediction
        loso_foundation_model_prediction.run(models=args.models, context_seconds=args.context,
                                             ictal_labels=ictal, history=args.history)
        return

    if args.experiment in ("lora_tabicl", "xai_tabicl"):
        if (args.context and len(args.context) > 1) or (ictal and len(ictal) > 1):
            parser.error(f"{args.experiment} takes a single --context and --ictal value")

    if args.experiment == "xai_tabicl":
        import xai_tabicl
        xai_tabicl.run(context_seconds=args.context[0] if args.context else None,
                       ictal_label=ictal[0] if ictal else None, history=args.history, patients=args.patients)
        return

    if args.experiment == "lora_tabicl":
        import finetune_lora_tabicl
        finetune_lora_tabicl.run(context_seconds=args.context[0] if args.context else None,
                                 ictal_label=ictal[0] if ictal else None, history=args.history)
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
