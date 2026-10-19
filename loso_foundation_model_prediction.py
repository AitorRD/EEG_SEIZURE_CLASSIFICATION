"""
LOSO seizure prediction on CHB-MIT with tabular foundation models (in-context learning).

Question: at the end of a 30 s window (t), does a seizure start within (t + SPH, t + SPH + HORIZON]?
Label 1 if so, 0 otherwise (no grey zone, no postictal exclusion). Windows crossing a clipping gap
and onsets inside the SPH are left out. Ictal windows are run in three variants: excluded, 0 or 1.

Run from main.py:  python main.py --experiment loso_fm
"""

import gc
import math
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# ===========================================================================
# Configuration
# ===========================================================================
MODELS = ["tabicl", "tabpfnv3", "tabfm", "mitra"]
HORIZON_SEC = 60.0
SPH_SEC = 0.0
CONTEXT_SECONDS = [30, 60, 120]
K_FEATURES = 200
MAX_CONTEXT_ROWS = 10000
PREDICT_CHUNK_ROWS = 2000   # TabFM materialises every ensemble member × all query rows in RAM at once
N_VAL_PATIENTS = 3
SEED = 42
ICTAL_LABELS = ["excluded", 0, 1]

FEATURES_DIR = Path("data/processed/tsfresh_features_chbmit_efficient")
OUTPUT_DIR = Path(f"images/results/loso_fm_prediction_chbmit_h{HORIZON_SEC:g}_sph{SPH_SEC:g}")

WINDOW_SEC = 30.0
STEP_SEC = 22.5


LOG_FILES = []


def log(*msg):
    line = " ".join([datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), *map(str, msg)])
    print(line, flush=True)
    for path in LOG_FILES:
        with open(path, "a") as f:
            f.write(line + "\n")


# ===========================================================================
# Data
# ===========================================================================
def load_patient(pid):
    windows = pd.read_parquet(FEATURES_DIR / "preictal_meta" / f"{pid}_ov0.25.parquet")
    features = pd.read_parquet(FEATURES_DIR / f"{pid}_ov0.25.parquet").set_index("window_id")

    X = features.drop(columns=["patient_id", "label"]).loc[windows["window_id"]].to_numpy(np.float32)
    X[np.logical_not(np.isfinite(X))] = np.nan

    time_to_onset = windows["tto"].to_numpy()
    seizure_soon = (time_to_onset > SPH_SEC) & (time_to_onset <= SPH_SEC + HORIZON_SEC)
    during_seizure = windows["ictal"].to_numpy()
    crosses_gap = windows["gap"].to_numpy()

    new_session = (windows["session"] != windows["session"].shift()).to_numpy()
    onset_id = windows["session"] + "@" + (windows["t_end"] + time_to_onset).round(1).astype(str)

    return {
        "X": X,
        "seizure_soon": seizure_soon,
        "ictal": during_seizure,
        "valid": np.logical_not(crosses_gap) & (time_to_onset > SPH_SEC),
        "segment": np.cumsum(new_session | crosses_gap),
        "tto": time_to_onset,
        "event": onset_id.to_numpy(),
    }


def with_ictal_label(patient, ictal_label):
    y = patient["seizure_soon"] | (patient["ictal"] & (ictal_label == 1))
    usable = patient["valid"] & np.logical_not(patient["ictal"] & (ictal_label == "excluded"))
    return {**patient, "y": y.astype(int), "usable": usable}


# ===========================================================================
# Features per fold
# ===========================================================================
def column_medians(X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nan_to_num(np.nanmedian(X, axis=0))


def fill_missing(X, medians):
    return np.where(np.isnan(X), medians, X)


def f_scores(X, y):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return np.nan_to_num(f_classif(X, y)[0], nan=-np.inf)


def select_columns(train, block_size=1000):
    y = np.concatenate([p["y"][p["usable"]] for p in train])
    n_columns = train[0]["X"].shape[1]
    scores = np.empty(n_columns)
    for start in range(0, n_columns, block_size):
        # float64: some tsfresh features reach ~1e19 and f_classif overflows in float32,
        # scoring constant columns as 3.4e38 so they win the selection.
        block = np.concatenate([p["X"][p["usable"], start:start + block_size] for p in train]).astype(np.float64)
        scores[start:start + block_size] = f_scores(fill_missing(block, column_medians(block)), y)
    return np.argsort(scores)[-K_FEATURES:]


def history_windows(context_seconds):
    return math.ceil((context_seconds - WINDOW_SEC) / STEP_SEC)


def add_history(X, segment, n_previous):
    if n_previous == 0:
        return X
    previous = pd.DataFrame(X).groupby(segment).shift(1)
    recent = previous.groupby(segment).rolling(n_previous, min_periods=1)
    recent_mean = recent.mean().droplevel(0).sort_index().to_numpy()
    recent_std = recent.std().droplevel(0).sort_index().to_numpy()
    change = np.nan_to_num(X - recent_mean)
    return np.hstack([X, change, np.nan_to_num(recent_std)])


def design(train, others, n_previous):
    columns = select_columns(train)
    medians = column_medians(np.concatenate([p["X"][:, columns][p["usable"]] for p in train]))

    def build(patient):
        X = fill_missing(patient["X"][:, columns], medians)
        return add_history(X, patient["segment"], n_previous)[patient["usable"]]

    X_train = np.concatenate([build(p) for p in train])
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), [scaler.transform(build(p)) for p in others]


def context(X, y, rng):
    if len(X) <= MAX_CONTEXT_ROWS:
        return X, y
    positives = np.flatnonzero(y == 1)
    negatives = rng.choice(np.flatnonzero(y == 0), MAX_CONTEXT_ROWS - len(positives), replace=False)
    rows = np.concatenate([positives, negatives])
    return X[rows], y[rows]


# ===========================================================================
# Models
# ===========================================================================
_tabfm_weights = None


def build_model(key, n_rows):
    global _tabfm_weights
    if key == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(random_state=SEED)
    if key in ("tabpfnv2", "tabpfnv3"):
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier.create_default_for_version(key[-2:], random_state=SEED,
                                                           ignore_pretraining_limits=True)
    if key == "tabfm":
        import torch
        from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch
        if _tabfm_weights is None:
            _tabfm_weights = tabfm_v1_0_0_pytorch.load()
        return TabFMClassifier(model=_tabfm_weights.to("cuda" if torch.cuda.is_available() else "cpu"),
                               random_state=SEED)
    if key == "mitra":
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
        model = MitraClassifier(fine_tune=False, seed=SEED, verbose=False)
        create_config = model._create_config

        def _create_config(*a, **kw):  # whole context as support; Mitra halves it itself on OOM
            cfg, cls = create_config(*a, **kw)
            cfg.hyperparams["max_samples_support"] = n_rows
            return cfg, cls
        model._create_config = _create_config
        return model
    raise ValueError(key)


def release_gpu():
    gc.collect()
    try:
        import torch
        if _tabfm_weights is not None:
            _tabfm_weights.to("cpu")
        torch.cuda.empty_cache()
    except ImportError:
        pass


# ===========================================================================
# Metrics
# ===========================================================================
def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    return float(thr[np.nanargmax(2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None))])


def evaluate(f):
    y, p, alarm = f["y"].to_numpy(), f["proba"].to_numpy(), (f["proba"] >= f["threshold"]).to_numpy()
    tp, fp = (alarm & (y == 1)).sum(), (alarm & (y == 0)).sum()
    fn, tn = (np.logical_not(alarm) & (y == 1)).sum(), (np.logical_not(alarm) & (y == 0)).sum()
    preictal = (y == 1) & np.logical_not(f["ictal"].to_numpy())
    per_seizure = f[preictal].assign(alarm=alarm[preictal]).groupby("event")
    warning = f[preictal & alarm].groupby("event")["tto"].max()
    f1_seizure = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_no_seizure = 2 * tn / max(2 * tn + fp + fn, 1)
    return {
        "ROC AUC": roc_auc_score(y, p), "PR AUC": average_precision_score(y, p),
        "Accuracy": (tp + tn) / len(y),
        "Recall": tp / max(tp + fn, 1), "Specificity": tn / max(tn + fp, 1),
        "Precision": tp / max(tp + fp, 1),
        "F1": f1_seizure, "F1 Macro": (f1_seizure + f1_no_seizure) / 2,
        "Seizure Sensitivity": per_seizure["alarm"].any().mean(),
        "Warning Time (s)": warning.mean(),
        "FA/h": fp / ((y == 0).sum() * STEP_SEC / 3600),   # alarm windows, no refractory period
        "Context rows": f["context_rows"].iloc[0],
        "Threshold": f["threshold"].iloc[0], "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ===========================================================================
# LOSO
# ===========================================================================
def context_rows_used(model, n_rows):
    if getattr(model, "trainers", None):
        return min(model.trainers[0].cfg.hyperparams["max_samples_support"], n_rows)
    return n_rows


def predict_positive(model, X):
    # Query rows do not attend to each other, so chunking only bounds memory.
    return np.concatenate([model.predict_proba(X[s:s + PREDICT_CHUNK_ROWS])[:, 1]
                           for s in range(0, len(X), PREDICT_CHUNK_ROWS)])


def run_loso(raw, ictal_label, context_seconds, out_dir):
    pids = list(raw)
    data = {pid: with_ictal_label(p, ictal_label) for pid, p in raw.items()}
    (out_dir / "fold_results").mkdir(parents=True, exist_ok=True)

    for i, test_pid in enumerate(pids):
        test = data[test_pid]
        todo = [k for k in MODELS if not (out_dir / "fold_results" / f"{k}__{test_pid}.parquet").exists()]
        if not todo or not test["y"][test["usable"]].any():
            continue
        rng = np.random.RandomState(SEED + i)
        others = [p for p in pids if p != test_pid]
        val_pids = sorted(rng.choice([p for p in others if data[p]["y"][data[p]["usable"]].any()],
                                     N_VAL_PATIENTS, replace=False))
        train = [data[p] for p in others if p not in val_pids]
        val = [data[p] for p in val_pids]

        X_tr, (*X_val, X_te) = design(train, val + [test], history_windows(context_seconds))
        X_ctx, y_ctx = context(X_tr, np.concatenate([p["y"][p["usable"]] for p in train]), rng)
        y_val = np.concatenate([p["y"][p["usable"]] for p in val])
        log(f"[EEG {context_seconds}s · ictal={ictal_label}] Fold {i + 1}/{len(pids)} {test_pid}: "
            f"{X_ctx.shape[0]} context rows ({int(y_ctx.sum())} positives) × {X_ctx.shape[1]} columns")

        for key in todo:
            try:
                model = build_model(key, len(X_ctx))
                model.fit(X_ctx, y_ctx)
                thr = best_f1_threshold(y_val, predict_positive(model, np.concatenate(X_val)))
                proba = predict_positive(model, X_te)
                rows_used = context_rows_used(model, len(X_ctx))
            except Exception as e:
                reason = "out of GPU memory" if "OutOfMemory" in type(e).__name__ else f"{type(e).__name__}: {e}"
                log(f"  {key}: skipped, EEG {context_seconds}s does not work ({reason})")
                continue
            finally:
                model = None
                release_gpu()
            u = test["usable"]
            fold = pd.DataFrame({"y": test["y"][u], "ictal": test["ictal"][u], "tto": test["tto"][u],
                                 "event": test["event"][u], "proba": proba, "threshold": thr,
                                 "context_rows": rows_used})
            fold.to_parquet(out_dir / "fold_results" / f"{key}__{test_pid}.parquet", index=False)
            log(f"  {key}: EEG {context_seconds}s with {rows_used} context rows, ROC={roc_auc_score(fold['y'], proba):.3f}")


# ===========================================================================
# Report
# ===========================================================================
def load_predictions(out_dir):
    predictions = {}
    for model in MODELS:
        files = sorted((out_dir / "fold_results").glob(f"{model}__*.parquet"))
        if files:
            predictions[model] = {f.stem.split("__")[1]: pd.read_parquet(f) for f in files}
    return predictions


def metrics_per_fold(predictions):
    rows = []
    for model, folds in predictions.items():
        for patient, fold in folds.items():
            rows.append({"model": model, "patient": patient, **evaluate(fold)})
    return pd.DataFrame(rows)


def write_markdown(metrics, ictal_label, context_seconds, out_dir):
    counts = ["TN", "FP", "FN", "TP"]
    summary = (metrics.drop(columns=["patient", "Threshold", *counts])
               .groupby("model", sort=False)
               .agg(["mean", "std"]))
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
    summary.insert(0, "folds", metrics.groupby("model", sort=False).size())
    confusion_totals = metrics.groupby("model", sort=False)[counts].sum()

    setup = (f"Onset within ({SPH_SEC:g}, {SPH_SEC + HORIZON_SEC:g}] s · "
             f"EEG seen per prediction {context_seconds} s · "
             f"k={K_FEATURES} · context {MAX_CONTEXT_ROWS} rows · "
             f"threshold max-F1 on {N_VAL_PATIENTS} val patients · "
             f"ictal windows: {ictal_label} "
             f"(seizure sensitivity and warning time use preictal windows only)")
    sections = [
        "# LOSO seizure prediction — tabular foundation models — CHB-MIT",
        setup,
        "## Mean / std across folds", summary.round(4).to_markdown(),
        "## Confusion totals", confusion_totals.to_markdown(),
        "## Per fold", metrics.round(4).to_markdown(index=False),
    ]
    (out_dir / "loso_report.md").write_text("\n\n".join(sections))


def plot_roc(ax, name, predictions):
    fpr, tpr, _ = roc_curve(predictions["y"], predictions["proba"])
    auc = roc_auc_score(predictions["y"], predictions["proba"])
    ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})")


def plot_precision_recall(ax, name, predictions):
    precision, recall, _ = precision_recall_curve(predictions["y"], predictions["proba"])
    ap = average_precision_score(predictions["y"], predictions["proba"])
    ax.plot(recall, precision, label=f"{name} ({ap:.3f})")


def plot_alarm_rate(ax, name, predictions, max_seconds=1800, bin_seconds=30):
    not_ictal = predictions[np.logical_not(predictions["ictal"])]
    alarm = not_ictal["proba"] >= not_ictal["threshold"]
    edges = np.arange(0, max_seconds + bin_seconds, bin_seconds)
    rate = alarm.groupby(pd.cut(not_ictal["tto"], edges), observed=False).mean()
    ax.plot(-(edges[:-1] + bin_seconds / 2), rate.to_numpy(), marker="o", ms=3, label=name)


def write_figure(predictions, out_dir):
    fig, (ax_roc, ax_pr, ax_alarm) = plt.subplots(1, 3, figsize=(19, 5.5))
    for model, folds in predictions.items():
        pooled = pd.concat(folds.values())
        plot_roc(ax_roc, model, pooled)
        plot_precision_recall(ax_pr, model, pooled)
        plot_alarm_rate(ax_alarm, model, pooled)

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_alarm.axvspan(-(SPH_SEC + HORIZON_SEC), -SPH_SEC, color="tab:red", alpha=0.12)
    ax_roc.set(title="ROC", xlabel="FPR", ylabel="TPR")
    ax_pr.set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax_alarm.set(title="Alarm rate vs time to onset", xlabel="Time to onset (s)", ylabel="Alarm rate")
    for ax in (ax_roc, ax_pr, ax_alarm):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "results.png", dpi=200)
    plt.close(fig)


TABLE_METRICS = ["ROC AUC", "Accuracy", "Recall", "Precision", "F1", "F1 Macro"]


def write_metrics_table(metrics, ictal_label, context_seconds, out_dir):
    grouped = metrics.groupby("model", sort=False)
    mean, std = grouped[TABLE_METRICS].mean(), grouped[TABLE_METRICS].std()
    cells = [[f"{mean.loc[m, c]:.3f} ± {std.loc[m, c]:.3f}" for c in TABLE_METRICS] for m in mean.index]

    fig, ax = plt.subplots(figsize=(12, 1 + 0.5 * len(mean)))
    ax.axis("off")
    table = ax.table(cellText=cells, rowLabels=list(mean.index), colLabels=TABLE_METRICS, loc="center")
    table.scale(1, 1.6)
    ax.set_title(f"EEG {context_seconds} s · ictal {ictal_label} · mean ± std over folds")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_table.png", dpi=200)
    plt.close(fig)


def report(ictal_label, context_seconds, out_dir):
    predictions = load_predictions(out_dir)
    if not predictions:
        log(f"No results in {out_dir}")
        return
    metrics = metrics_per_fold(predictions)
    metrics.to_csv(out_dir / "loso_fold_metrics.csv", index=False)
    write_markdown(metrics, ictal_label, context_seconds, out_dir)
    write_figure(predictions, out_dir)
    write_metrics_table(metrics, ictal_label, context_seconds, out_dir)
    log(f"Report: {out_dir / 'loso_report.md'}")


def write_overall_summary():
    rows = []
    for context_seconds in CONTEXT_SECONDS:
        for ictal_label in ICTAL_LABELS:
            out_dir = OUTPUT_DIR / f"context_{context_seconds}s" / f"ictal_{ictal_label}"
            if not (out_dir / "loso_fold_metrics.csv").exists():
                continue
            metrics = pd.read_csv(out_dir / "loso_fold_metrics.csv")
            means = metrics.groupby("model", sort=False)[TABLE_METRICS + ["Context rows"]].mean()
            means.insert(0, "folds", metrics.groupby("model", sort=False).size())
            means.insert(0, "ictal", ictal_label)
            means.insert(0, "EEG seconds", context_seconds)
            rows.append(means.reset_index())
    if rows:
        summary = pd.concat(rows, ignore_index=True)
        summary.to_csv(OUTPUT_DIR / "summary_all_variants.csv", index=False)
        (OUTPUT_DIR / "summary_all_variants.md").write_text(summary.round(4).to_markdown(index=False))
        log(f"Summary of all variants: {OUTPUT_DIR / 'summary_all_variants.md'}")


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILES[:] = [OUTPUT_DIR / "experiment.log"]
    log(f"Start: models {MODELS} · EEG seconds {CONTEXT_SECONDS} · ictal {ICTAL_LABELS}")
    pids = sorted(f.name.split("_")[0] for f in FEATURES_DIR.glob("chb*_ov0.25.parquet"))
    raw = {pid: load_patient(pid) for pid in pids}
    log(f"Loaded {len(pids)} patients")

    for context_seconds in CONTEXT_SECONDS:
        for ictal_label in ICTAL_LABELS:
            out_dir = OUTPUT_DIR / f"context_{context_seconds}s" / f"ictal_{ictal_label}"
            out_dir.mkdir(parents=True, exist_ok=True)
            LOG_FILES[:] = [OUTPUT_DIR / "experiment.log", out_dir / "log.txt"]
            log(f"=== Variant: EEG {context_seconds}s · ictal {ictal_label} → {out_dir}")
            run_loso(raw, ictal_label, context_seconds, out_dir)
            report(ictal_label, context_seconds, out_dir)

    LOG_FILES[:] = [OUTPUT_DIR / "experiment.log"]
    write_overall_summary()
    log("Finished")
