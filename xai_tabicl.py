"""
Explainability (XAI) of TabICL for LOSO seizure prediction on CHB-MIT: SHAP, LIME and a
bipolar-montage topographic map.

For each LOSO fold the model is rebuilt exactly as in loso_foundation_model_prediction.py
(same validation patients, features, history summary and 10 000-row context) and explained on
windows of the held-out test patient:

  SHAP   how much each column pushed P(preictal) up or down for every explained window
         (TabICL's own explainer: a column is "removed" by setting it to NaN).
  LIME   for two windows per fold (the strongest true alarm and the strongest false alarm),
         a local linear rule with the 10 columns that drive that single prediction.

Importances are mean |SHAP| per column, normalised to a share of 100 % inside each fold (the
selected columns change between folds) and averaged over folds, then grouped by:

  derivation  the bipolar channel the feature comes from (FP1-F7 is FP1 minus F7)
  family      the tsfresh calculator (fft_coefficient, sample_entropy, ...)
  block       the window itself vs. its history summary (change / std / trend / change_index)

CHB-MIT channels are differences between two electrodes, so the map does not paint electrodes:
(a) places each derivation at the midpoint of its two electrodes and (b) draws it as a line
between them. T8-P8-0 / T8-P8-1 and T7-P7 / P7-T7 record the same pair and are merged.

Run from main.py:  python main.py --experiment xai_tabicl --context 600 --ictal excluded --history trend10
"""

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import loso_foundation_model_prediction as P
from loso_foundation_model_prediction import log

# ===========================================================================
# Configuration
# ===========================================================================
CONTEXT_SECONDS = 600
ICTAL_LABEL = "excluded"
HISTORY_MODE = "trend10"

N_POSITIVE = 40          # preictal windows of the test patient explained with SHAP (all if fewer)
N_NEGATIVE = 40          # interictal windows explained with SHAP
LIME_SAMPLES = 1000      # perturbed rows per LIME explanation
LIME_FEATURES = 10
LIME_BACKGROUND = 2000   # context rows LIME uses to learn each column's distribution
TOP_FEATURES = 20
KV_CACHE = "repr"        # TabICL reuses the context representation between SHAP batches

SEED = P.SEED
OUTPUT_ROOT = P.OUTPUT_DIR.parent / f"xai_tabicl_chbmit_h{P.HORIZON_SEC:g}_sph{P.SPH_SEC:g}"
OUTPUT_DIR = None   # set by run()

SERIES, INK, INK_2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"

# CHB-MIT electrode labels -> MNE standard_1020 names
ELECTRODE = {"FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz"}


# ===========================================================================
# Column groups
# ===========================================================================
def derivation(column):
    """'T8-P8-1__fft_coefficient__...' -> ('T8', 'P8'); None for columns without a channel."""
    if "__" not in column:
        return None
    a, b = re.sub(r"-\d+$", "", column.split("__")[0]).split("-")
    return a, b


def family(column):
    return column.split("__")[1] if column.count("__") >= 1 else column


def block(column):
    for suffix in ("change", "std", "trend"):
        if column.endswith(f"__{suffix}"):
            return suffix
    return "change_index" if column == "change_index" else "window"


def pair_label(pair):
    return "-".join(pair)


# ===========================================================================
# One fold
# ===========================================================================
def build_fold(data, pids, i, feature_names):
    """Same fold as loso_foundation_model_prediction.run_loso (same rng sequence)."""
    test_pid = pids[i]
    rng = np.random.RandomState(SEED + i)
    others = [p for p in pids if p != test_pid]
    val_pids = sorted(rng.choice([p for p in others if data[p]["y"][data[p]["usable"]].any()],
                                 P.N_VAL_PATIENTS, replace=False))
    train = [data[p] for p in others if p not in val_pids]
    n_previous = P.history_windows(CONTEXT_SECONDS)
    X_tr, (X_te,), columns = P.design(train, [data[test_pid]], n_previous, return_columns=True)
    X_ctx, y_ctx = P.context(X_tr, np.concatenate([p["y"][p["usable"]] for p in train]), rng)
    test = data[test_pid]
    return X_ctx, y_ctx, X_te, test["y"][test["usable"]], P.column_names(feature_names, columns, n_previous)


def pick_windows(y, rng):
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.sort(np.concatenate([
        rng.choice(positives, min(N_POSITIVE, len(positives)), replace=False),
        rng.choice(negatives, min(N_NEGATIVE, len(negatives)), replace=False),
    ]))


def explain_shap(clf, X, names):
    # Same as tabicl.shap.get_shap_explainer (which also needs shapiq): an all-NaN background,
    # so a "removed" column is NaN and TabICL's preprocessing fills it.
    import shap
    explainer = shap.Explainer(clf.predict_proba, np.full((1, X.shape[1]), np.nan), feature_names=names)
    sv = explainer(X, max_evals=2 * X.shape[1] + 1)
    return sv.values[..., 1], sv.base_values[:, 1]   # contributions to P(preictal)


def explain_lime(clf, X_ctx, x, names, rng):
    import lime.lime_tabular
    background = X_ctx[rng.choice(len(X_ctx), min(LIME_BACKGROUND, len(X_ctx)), replace=False)]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        background, feature_names=names, class_names=["interictal", "preictal"],
        discretize_continuous=True, random_state=SEED)
    return explainer.explain_instance(x, clf.predict_proba, labels=(1,),
                                      num_features=LIME_FEATURES, num_samples=LIME_SAMPLES)


def run_fold(data, pids, i, feature_names):
    from tabicl import TabICLClassifier
    test_pid = pids[i]
    X_ctx, y_ctx, X_te, y_te, names = build_fold(data, pids, i, feature_names)
    rng = np.random.RandomState(SEED + 2000 + i)
    rows = pick_windows(y_te, rng)
    log(f"[XAI · EEG {CONTEXT_SECONDS}s · {HISTORY_MODE} · ictal={ICTAL_LABEL}] Fold {i + 1}/{len(pids)} {test_pid}: "
        f"{len(X_ctx)} context rows × {X_ctx.shape[1]} columns, explaining {len(rows)} windows "
        f"({int(y_te[rows].sum())} preictal)")

    clf = TabICLClassifier(random_state=SEED, kv_cache=KV_CACHE).fit(X_ctx, y_ctx)
    try:
        proba = clf.predict_proba(X_te[rows])[:, 1]
        values, base = explain_shap(clf, X_te[rows], names)
        shap_frame = pd.DataFrame(values, columns=names).assign(row=rows, y=y_te[rows], proba=proba, base=base)
        shap_frame.to_parquet(OUTPUT_DIR / "shap" / f"{test_pid}.parquet", index=False)
        plot_fold_beeswarm(values, X_te[rows], names, test_pid)

        # LIME: the strongest true alarm and the strongest false alarm of the explained windows
        cases = {"true_alarm": y_te[rows] == 1, "false_alarm": y_te[rows] == 0}
        for case, mask in cases.items():
            if not mask.any():
                continue
            k = np.flatnonzero(mask)[np.argmax(proba[mask])]
            explanation = explain_lime(clf, X_ctx, X_te[rows[k]], names, rng)
            save_lime(explanation, test_pid, case, proba[k])
    finally:
        clf = None
        P.release_gpu()
    log(f"  {test_pid}: SHAP + LIME saved")


# ===========================================================================
# Per-fold figures
# ===========================================================================
def plot_fold_beeswarm(values, X, names, pid):
    import shap
    explanation = shap.Explanation(values=values, data=X, feature_names=names)
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=15, show=False, color_bar_label="Column value (standardised)")
    plt.title(f"{pid} · SHAP on P(preictal) · {len(values)} windows", loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fold_plots" / f"{pid}_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def save_lime(explanation, pid, case, proba):
    weights = explanation.as_list(label=1)
    pd.DataFrame(weights, columns=["rule", "weight"]).to_csv(OUTPUT_DIR / "lime" / f"{pid}_{case}.csv", index=False)
    rules, w = zip(*reversed(weights))
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(rules) + 1.4))
    ax.barh(range(len(rules)), w, color=[SERIES if v > 0 else "#c3c2b7" for v in w], height=0.6)
    ax.set_yticks(range(len(rules)), [r if len(r) < 90 else r[:87] + "..." for r in rules], fontsize=8, color=INK)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("Weight towards preictal (blue) / interictal (grey)", color=INK_2)
    ax.set_title(f"{pid} · LIME · {case.replace('_', ' ')} · P(preictal) = {proba:.3f}", loc="left", fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lime" / f"{pid}_{case}.png", dpi=150)
    plt.close(fig)


# ===========================================================================
# Aggregation over folds
# ===========================================================================
def fold_shares():
    """Mean |SHAP| per column, as a % of the fold total, one row per (fold, column)."""
    rows = []
    for f in sorted((OUTPUT_DIR / "shap").glob("*.parquet")):
        frame = pd.read_parquet(f).drop(columns=["row", "y", "proba", "base"])
        importance = frame.abs().mean()
        share = 100 * importance / importance.sum()
        rows.append(pd.DataFrame({"fold": f.stem, "column": share.index, "share": share.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def grouped(shares, key):
    shares = shares.assign(group=shares["column"].map(key)).dropna(subset=["group"])
    per_fold = shares.groupby(["fold", "group"])["share"].sum().unstack(fill_value=0)
    return per_fold.mean().sort_values(ascending=False)


def plot_bars(ax, series, title, xlabel="Mean share of |SHAP| per fold (%)"):
    series = series.iloc[::-1]
    ax.barh(range(len(series)), series.to_numpy(), color=SERIES, height=0.62)
    ax.set_yticks(range(len(series)), series.index, fontsize=8, color=INK)
    for k, v in enumerate(series.to_numpy()):
        ax.text(v, k, f" {v:.1f}", va="center", fontsize=7, color=INK_2)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_summary(shares, n_folds):
    per_column = shares.pivot_table(index="column", columns="fold", values="share", fill_value=0)
    top = per_column.mean(axis=1).sort_values(ascending=False).head(TOP_FEATURES)
    folds_selected = shares.groupby("column")["fold"].nunique()
    top.index = [f"{c}  ({folds_selected[c]}/{n_folds} folds)" for c in top.index]

    fig, ax = plt.subplots(figsize=(11, 0.32 * TOP_FEATURES + 1.5))
    plot_bars(ax, top, f"Top {TOP_FEATURES} columns · TabICL · EEG {CONTEXT_SECONDS}s · {HISTORY_MODE}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_top_features.png", dpi=200)
    plt.close(fig)

    by_family, by_block = grouped(shares, family).head(15), grouped(shares, block)
    fig, (ax_f, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    plot_bars(ax_f, by_family, "By tsfresh feature family (top 15)")
    plot_bars(ax_b, by_block, "Current window vs. history summary")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_by_family_and_block.png", dpi=200)
    plt.close(fig)
    return top, by_family, by_block


def plot_bipolar_map(shares):
    import mne
    key = lambda c: None if derivation(c) is None else frozenset(derivation(c))
    labels = {}
    for c in shares["column"]:
        if key(c) is not None:
            labels.setdefault(key(c), pair_label(derivation(c)))
    by_pair = grouped(shares, key)
    by_pair.index = [labels[p] for p in by_pair.index]

    positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    pos = lambda e: positions[ELECTRODE.get(e, e)]
    pairs = [tuple(label.split("-")) for label in by_pair.index]

    def flat(p):   # azimuthal equidistant projection of the head (Cz at the centre, nose up)
        x, y, z = p / np.linalg.norm(p)
        r = np.arccos(np.clip(z, -1, 1)) / (np.pi / 2)
        phi = np.arctan2(y, x)
        return np.array([r * np.cos(phi), r * np.sin(phi)])

    def angle(a, b):
        u, v = pos(a) / np.linalg.norm(pos(a)), pos(b) / np.linalg.norm(pos(b))
        return np.degrees(np.arccos(np.clip(u @ v, -1, 1)))

    def draw_head(ax):
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=INK_2, lw=1, zorder=2))
        ax.plot([-0.09, 0, 0.09], [0.995, 1.09, 0.995], color=INK_2, lw=1)
        ax.set_aspect("equal")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.25, 1.2)
        ax.axis("off")

    values = by_pair.to_numpy()
    norm = plt.Normalize(0, values.max())
    cmap = plt.get_cmap("Blues")
    fig, (ax_map, ax_lines) = plt.subplots(1, 2, figsize=(14, 6.8), layout="constrained")

    # (a) each derivation at the midpoint of its two electrodes (same projection as b), values
    # interpolated only inside the area the derivations cover. A derivation spanning the head
    # (FT9-FT10) has no meaningful midpoint, so it is only drawn in (b).
    from scipy.interpolate import griddata
    local = [(label, pair) for label, pair in zip(by_pair.index, pairs) if angle(*pair) <= 60]
    spanning = [label for label, pair in zip(by_pair.index, pairs) if angle(*pair) > 60]
    points = np.array([(flat(pos(a)) + flat(pos(b))) / 2 for _, (a, b) in local])
    grid_x, grid_y = np.meshgrid(np.linspace(-1.3, 1.3, 300), np.linspace(-1.2, 1.2, 300))
    surface = griddata(points, by_pair[[label for label, _ in local]].to_numpy(), (grid_x, grid_y), method="cubic")
    image = ax_map.pcolormesh(grid_x, grid_y, surface, cmap=cmap, norm=norm, shading="auto")
    ax_map.contour(grid_x, grid_y, surface, levels=4, colors=INK_2, linewidths=0.5)
    for (label, _), (x, y) in zip(local, points):
        ax_map.plot(x, y, ".", color=INK, ms=3)
        ax_map.text(x, y + 0.04, label, ha="center", va="bottom", fontsize=6.5, color=INK)
    draw_head(ax_map)
    ax_map.set_title("(a) Each derivation at the midpoint of its two electrodes", loc="left", fontsize=10)
    if spanning:
        ax_map.text(0, -1.2, f"{', '.join(spanning)} runs across the whole head and is only drawn in (b)",
                    ha="center", fontsize=8, color=INK_2)

    # (b) each derivation as a line between its electrodes, darker and thicker = more important
    for (a, b), v in sorted(zip(pairs, values), key=lambda t: t[1]):
        (xa, ya), (xb, yb) = flat(pos(a)), flat(pos(b))
        ax_lines.plot([xa, xb], [ya, yb], color=cmap(norm(v)), lw=1.5 + 6 * norm(v), solid_capstyle="round")
    for e in sorted({e for pair in pairs for e in pair}):
        x, y = flat(pos(e))
        ax_lines.plot(x, y, "o", ms=13, color="white", mec=INK_2, mew=1, zorder=3)
        ax_lines.text(x, y, ELECTRODE.get(e, e), ha="center", va="center", fontsize=6, color=INK, zorder=4)
    draw_head(ax_lines)
    ax_lines.set_title("(b) Each derivation as a line between its electrodes", loc="left", fontsize=10)
    fig.colorbar(image, ax=[ax_map, ax_lines], shrink=0.7, label="Mean share of |SHAP| per fold (%) · darker = more important")
    fig.suptitle(f"Where TabICL looks · EEG {CONTEXT_SECONDS}s · {HISTORY_MODE} · ictal {ICTAL_LABEL}",
                 x=0.02, ha="left", fontsize=13)
    fig.savefig(OUTPUT_DIR / "topomap_bipolar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return by_pair


def report():
    shares = fold_shares()
    n_folds = shares["fold"].nunique()
    top, by_family, by_block = plot_summary(shares, n_folds)
    by_pair = plot_bipolar_map(shares)
    as_table = lambda s, name: s.rename("share %").round(2).rename_axis(name).to_markdown()
    sections = [
        "# XAI — TabICL — LOSO seizure prediction — CHB-MIT",
        f"EEG seen per prediction {CONTEXT_SECONDS} s · history {HISTORY_MODE} · ictal windows {ICTAL_LABEL} · "
        f"{n_folds} folds · SHAP on up to {N_POSITIVE} preictal + {N_NEGATIVE} interictal test windows per fold. "
        "Share = mean |SHAP| of a column as % of the fold total, averaged over folds "
        "(a column not selected in a fold counts 0 there).",
        f"## Top {TOP_FEATURES} columns", as_table(top, "column"),
        "## By bipolar derivation", as_table(by_pair, "derivation"),
        "## By tsfresh feature family (top 15)", as_table(by_family, "family"),
        "## Current window vs. history summary", as_table(by_block, "block"),
        "Figures: shap_top_features.png, shap_by_family_and_block.png, topomap_bipolar.png, "
        "fold_plots/*_shap_beeswarm.png, lime/*.png",
    ]
    (OUTPUT_DIR / "xai_report.md").write_text("\n\n".join(sections))
    log(f"Report: {OUTPUT_DIR / 'xai_report.md'}")


def run(context_seconds=None, ictal_label=None, history=None, patients=None):
    global CONTEXT_SECONDS, ICTAL_LABEL, HISTORY_MODE, OUTPUT_DIR
    CONTEXT_SECONDS = context_seconds or CONTEXT_SECONDS
    ICTAL_LABEL = ICTAL_LABEL if ictal_label is None else ictal_label
    HISTORY_MODE = P.HISTORY_MODE = history or HISTORY_MODE
    OUTPUT_DIR = OUTPUT_ROOT / P.variant_dir(CONTEXT_SECONDS, ICTAL_LABEL).relative_to(P.OUTPUT_DIR)
    for folder in ("shap", "lime", "fold_plots"):
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)
    P.LOG_FILES[:] = [OUTPUT_DIR / "log.txt"]
    log(f"Start: XAI TabICL · EEG {CONTEXT_SECONDS}s · history {HISTORY_MODE} · ictal {ICTAL_LABEL}")

    pids = sorted(f.name.split("_")[0] for f in P.FEATURES_DIR.glob("chb*_ov0.25.parquet"))
    feature_names = (pd.read_parquet(P.FEATURES_DIR / f"{pids[0]}_ov0.25.parquet")
                     .drop(columns=["window_id", "patient_id", "label"]).columns.to_numpy())
    data = {pid: P.with_ictal_label(P.load_patient(pid), ICTAL_LABEL) for pid in pids}
    log(f"Loaded {len(pids)} patients")

    for i, pid in enumerate(pids):
        if patients and pid not in patients:
            continue
        if (OUTPUT_DIR / "shap" / f"{pid}.parquet").exists() or not data[pid]["y"][data[pid]["usable"]].any():
            continue
        try:
            run_fold(data, pids, i, feature_names)
        except Exception as e:
            reason = "out of GPU memory" if "OutOfMemory" in type(e).__name__ else f"{type(e).__name__}: {e}"
            log(f"  {pid}: skipped ({reason})")
    report()
    log("Finished")


if __name__ == "__main__":
    run()
