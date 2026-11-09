"""
LoRA fine-tuning of TabICL for LOSO seizure prediction on CHB-MIT.

Same protocol as loso_foundation_model_prediction.py (labels, folds, validation patients,
feature design, 10 000-row context, max-F1 threshold, metrics) for one TabICL variant
(CONTEXT_SECONDS, ICTAL_LABEL). The only change is that, in every fold,
TabICL's attention and feed-forward weights get a trainable low-rank update W + (alpha/r)·B·A
learnt on the fold's training patients; the pretrained weights stay frozen.

Training is episodic and mimics LOSO: each step takes the queries from one training patient
and the in-context rows from the others. Early stopping on the fold's validation patients
(the same ones that pick the threshold), so the test patient never influences training.

Run from main.py:  python main.py --experiment lora_tabicl [--context 300] [--ictal excluded]
"""

import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn.utils import parametrize

import loso_foundation_model_prediction as P
from loso_foundation_model_prediction import log

# ===========================================================================
# Configuration
# ===========================================================================
CONTEXT_SECONDS = 120
ICTAL_LABEL = "excluded"
MODEL_KEY = "tabicl_lora"

LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGETS = ("col_embedder", "row_interactor", "icl_predictor")

STEPS = 300
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EPISODE_CONTEXT_ROWS = 1024
EPISODE_QUERY_ROWS = 256
QUERY_POSITIVE_FRACTION = 0.25
EVAL_EVERY = 50
PATIENCE = 3
VAL_EVAL_ROWS = 3000
EARLY_STOP_METRIC = "ROC AUC"   # or "PR AUC"

SEED = P.SEED
VARIANT = BASELINE_DIR = OUTPUT_DIR = None   # set by set_variant() in run()


# ===========================================================================
# LoRA
# ===========================================================================
class LoRAWeight(nn.Module):
    """Parametrization W -> W + (alpha / r) · B·A, with B = 0 at start."""

    def __init__(self, weight, rank, alpha):
        super().__init__()
        out_features, in_features = weight.shape
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=weight.device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, W):
        return W + self.scaling * (self.lora_B @ self.lora_A).to(W.dtype)


def lora_weights(model):
    # TabICL's attention passes in_proj_weight / out_proj.weight as raw tensors to
    # F.linear, so wrapping the nn.Linear modules would be silently bypassed; a
    # parametrization changes the tensor itself and works on every code path.
    from tabicl._model.layers import MultiheadAttention, MultiheadAttentionBlock
    for name, module in model.named_modules():
        if not name.startswith(LORA_TARGETS):
            continue
        if isinstance(module, MultiheadAttention):
            yield f"{name}.in_proj_weight", module, "in_proj_weight"
            yield f"{name}.out_proj.weight", module.out_proj, "weight"
        elif isinstance(module, MultiheadAttentionBlock):
            yield f"{name}.linear1.weight", module.linear1, "weight"
            yield f"{name}.linear2.weight", module.linear2, "weight"


def inject_lora(model):
    for p in model.parameters():
        p.requires_grad_(False)
    names = []
    for name, module, attr in lora_weights(model):
        parametrize.register_parametrization(module, attr, LoRAWeight(getattr(module, attr), LORA_RANK, LORA_ALPHA))
        names.append(name)
    trainable = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in trainable:
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for n, p in model.named_parameters() if "lora_" not in n)
    log(f"  LoRA r={LORA_RANK} alpha={LORA_ALPHA}: {len(names)} weight matrices, "
        f"{n_trainable:,} trainable / {n_total:,} pretrained ({100 * n_trainable / n_total:.2f}%)")
    return trainable


def lora_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def load_lora_state(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, value in state.items():
            params[n].copy_(value)


# ===========================================================================
# Episodes
# ===========================================================================
def episode(clf, X, y, patient, context_positive_fraction, rng):
    """One LOSO-like task: queries from one training patient, context from the others,
    preprocessed by the same ensemble pipeline TabICLClassifier applies at inference."""
    from tabicl._sklearn.preprocessing import EnsembleGenerator

    query_patient = rng.choice(np.unique(patient[y == 1]))
    in_query = patient == query_patient

    def draw(candidates, n_positive, n_total):
        pos, neg = np.flatnonzero(candidates & (y == 1)), np.flatnonzero(candidates & (y == 0))
        n_positive = min(n_positive, n_total)
        return np.concatenate([rng.choice(pos, n_positive, replace=len(pos) < n_positive),
                               rng.choice(neg, n_total - n_positive, replace=len(neg) < n_total - n_positive)])

    ctx = draw(np.logical_not(in_query), max(1, round(EPISODE_CONTEXT_ROWS * context_positive_fraction)),
               EPISODE_CONTEXT_ROWS)
    query = draw(in_query, round(EPISODE_QUERY_ROWS * QUERY_POSITIVE_FRACTION), EPISODE_QUERY_ROWS)

    generator = EnsembleGenerator(
        classification=True, n_estimators=1,
        norm_methods=clf.norm_methods or ["none", "power"],
        feat_shuffle_method=clf.feat_shuffle_method, class_shuffle_method=clf.class_shuffle_method,
        outlier_threshold=clf.outlier_threshold, random_state=int(rng.randint(2 ** 31 - 1)),
    ).fit(X[ctx], y[ctx])
    norm_method, (Xs, ys) = next(iter(generator.transform(X[query], mode="both").items()))
    class_shuffle = np.array(generator.class_shuffles_[norm_method][0])
    return Xs, ys, class_shuffle[y[query]]


def train_step(model, optimizer, trainable, Xs, ys, y_query, device):
    model.train()
    X = torch.from_numpy(Xs).float().to(device)
    y_train = torch.from_numpy(ys).float().to(device)
    target = torch.from_numpy(y_query).long().to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(X, y_train)[0, :, :2]   # (queries, max_classes) -> the two classes in use
    loss = F.cross_entropy(logits.float(), target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
    optimizer.step()
    return loss.item()


# ===========================================================================
# Validation
# ===========================================================================
def val_subset(y_val, rng):
    if len(y_val) <= VAL_EVAL_ROWS:
        return np.arange(len(y_val))
    positives = np.flatnonzero(y_val == 1)
    negatives = rng.choice(np.flatnonzero(y_val == 0), VAL_EVAL_ROWS - len(positives), replace=False)
    return np.sort(np.concatenate([positives, negatives]))


def score(y, p):
    return roc_auc_score(y, p) if EARLY_STOP_METRIC == "ROC AUC" else average_precision_score(y, p)


def predict(clf, X):
    clf.model_.eval()
    with parametrize.cached():
        return P.predict_positive(clf, X)


# ===========================================================================
# Fine-tuning per fold
# ===========================================================================
def finetune(clf, X_tr, y_tr, patient, context_positive_fraction, X_val, y_val, rng):
    model, device = clf.model_, clf.device_
    trainable = inject_lora(model)
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_score = score(y_val, predict(clf, X_val))
    best_step, best_state = 0, lora_state(model)
    history = [{"step": 0, "loss": np.nan, f"val {EARLY_STOP_METRIC}": best_score}]
    log(f"    step 0 (pretrained): val {EARLY_STOP_METRIC}={best_score:.4f}")
    losses, evals_without_gain = [], 0

    for step in range(1, STEPS + 1):
        losses.append(train_step(model, optimizer, trainable,
                                 *episode(clf, X_tr, y_tr, patient, context_positive_fraction, rng), device))
        if step % EVAL_EVERY and step != STEPS:
            continue
        val_score = score(y_val, predict(clf, X_val))
        history.append({"step": step, "loss": np.mean(losses), f"val {EARLY_STOP_METRIC}": val_score})
        log(f"    step {step}: loss={np.mean(losses):.4f} val {EARLY_STOP_METRIC}={val_score:.4f}")
        losses = []
        if val_score > best_score:
            best_score, best_step, best_state, evals_without_gain = val_score, step, lora_state(model), 0
        else:
            evals_without_gain += 1
            if evals_without_gain >= PATIENCE:
                break

    load_lora_state(model, best_state)
    log(f"    best step {best_step}: val {EARLY_STOP_METRIC}={best_score:.4f}")
    return best_state, best_step, pd.DataFrame(history)


def run_loso(raw):
    pids = list(raw)
    data = {pid: P.with_ictal_label(p, ICTAL_LABEL) for pid, p in raw.items()}
    for folder in ("fold_results", "adapters", "training"):
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)

    for i, test_pid in enumerate(pids):
        test = data[test_pid]
        result = OUTPUT_DIR / "fold_results" / f"{MODEL_KEY}__{test_pid}.parquet"
        if result.exists() or not test["y"][test["usable"]].any():
            continue
        # Same rng sequence as loso_foundation_model_prediction.run_loso -> same val patients and context.
        rng = np.random.RandomState(SEED + i)
        others = [p for p in pids if p != test_pid]
        val_pids = sorted(rng.choice([p for p in others if data[p]["y"][data[p]["usable"]].any()],
                                     P.N_VAL_PATIENTS, replace=False))
        train = [data[p] for p in others if p not in val_pids]
        val = [data[p] for p in val_pids]

        X_tr, (*X_val, X_te) = P.design(train, val + [test], P.history_windows(CONTEXT_SECONDS))
        y_tr = np.concatenate([p["y"][p["usable"]] for p in train])
        patient = np.repeat(np.arange(len(train)), [p["usable"].sum() for p in train])
        X_ctx, y_ctx = P.context(X_tr, y_tr, rng)
        X_val, y_val = np.concatenate(X_val), np.concatenate([p["y"][p["usable"]] for p in val])
        log(f"[LoRA TabICL · EEG {CONTEXT_SECONDS}s · ictal={ICTAL_LABEL}] Fold {i + 1}/{len(pids)} {test_pid}: "
            f"{X_ctx.shape[0]} context rows ({int(y_ctx.sum())} positives) × {X_ctx.shape[1]} columns, "
            f"val {val_pids}")

        ft_rng = np.random.RandomState(SEED + 1000 + i)
        torch.manual_seed(SEED + i)
        try:
            clf = P.build_model("tabicl", len(X_ctx))
            clf.fit(X_ctx, y_ctx)
            sub = val_subset(y_val, ft_rng)
            state, best_step, history = finetune(clf, X_tr, y_tr, patient, y_ctx.mean(),
                                                 X_val[sub], y_val[sub], ft_rng)
            thr = P.best_f1_threshold(y_val, predict(clf, X_val))
            proba = predict(clf, X_te)
        except Exception as e:
            reason = "out of GPU memory" if "OutOfMemory" in type(e).__name__ else f"{type(e).__name__}: {e}"
            log(f"  {MODEL_KEY}: skipped ({reason})")
            continue
        finally:
            clf = None
            P.release_gpu()

        torch.save(state, OUTPUT_DIR / "adapters" / f"{test_pid}.pt")
        history.assign(best_step=best_step).to_csv(OUTPUT_DIR / "training" / f"{test_pid}.csv", index=False)
        u = test["usable"]
        fold = pd.DataFrame({"y": test["y"][u], "ictal": test["ictal"][u], "tto": test["tto"][u],
                             "event": test["event"][u], "proba": proba, "threshold": thr,
                             "context_rows": len(X_ctx)})
        fold.to_parquet(result, index=False)
        log(f"  {MODEL_KEY}: ROC={roc_auc_score(fold['y'], proba):.3f} (best step {best_step})")


# ===========================================================================
# Report: fine-tuned vs. pretrained TabICL on the same folds
# ===========================================================================
def copy_baseline():
    for f in sorted((BASELINE_DIR / "fold_results").glob("tabicl__*.parquet")):
        target = OUTPUT_DIR / "fold_results" / f.name
        if not target.exists():
            shutil.copy(f, target)


def report():
    predictions = {}
    for key in ("tabicl", MODEL_KEY):
        files = sorted((OUTPUT_DIR / "fold_results").glob(f"{key}__*.parquet"))
        if files:
            predictions[key] = {f.stem.split("__")[1]: pd.read_parquet(f) for f in files}
    if MODEL_KEY not in predictions:
        log(f"No results in {OUTPUT_DIR}")
        return
    # Compare on the folds both models have.
    common = set.intersection(*(set(folds) for folds in predictions.values()))
    predictions = {k: {p: f for p, f in folds.items() if p in common} for k, folds in predictions.items()}

    metrics = P.metrics_per_fold(predictions)
    metrics.to_csv(OUTPUT_DIR / "loso_fold_metrics.csv", index=False)
    P.write_markdown(metrics, ICTAL_LABEL, CONTEXT_SECONDS, OUTPUT_DIR)
    P.write_figure(predictions, OUTPUT_DIR)
    P.write_metrics_table(metrics, ICTAL_LABEL, CONTEXT_SECONDS, OUTPUT_DIR)
    log(f"Report: {OUTPUT_DIR / 'loso_report.md'}")


def set_variant(context_seconds, ictal_label, history):
    global CONTEXT_SECONDS, ICTAL_LABEL, VARIANT, BASELINE_DIR, OUTPUT_DIR
    CONTEXT_SECONDS, ICTAL_LABEL, P.HISTORY_MODE = context_seconds, ictal_label, history
    BASELINE_DIR = P.variant_dir(CONTEXT_SECONDS, ICTAL_LABEL)
    VARIANT = BASELINE_DIR.relative_to(P.OUTPUT_DIR)
    OUTPUT_DIR = P.OUTPUT_DIR.parent / f"loso_fm_lora_tabicl_chbmit_h{P.HORIZON_SEC:g}_sph{P.SPH_SEC:g}" / VARIANT


def run(context_seconds=None, ictal_label=None, history=None):
    set_variant(context_seconds or CONTEXT_SECONDS, ICTAL_LABEL if ictal_label is None else ictal_label,
                history or P.HISTORY_MODE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    P.LOG_FILES[:] = [OUTPUT_DIR / "log.txt"]
    log(f"Start: LoRA TabICL · EEG {CONTEXT_SECONDS}s · history {P.HISTORY_MODE} · ictal {ICTAL_LABEL} · "
        f"r={LORA_RANK} alpha={LORA_ALPHA} "
        f"· {STEPS} steps lr={LEARNING_RATE} · episodes {EPISODE_CONTEXT_ROWS}+{EPISODE_QUERY_ROWS} rows")
    if not (BASELINE_DIR / "fold_results").exists():
        log(f"Warning: no pretrained TabICL baseline in {BASELINE_DIR}; the report will show LoRA only. "
            f"Run first: python main.py --experiment loso_fm --models tabicl "
            f"--context {CONTEXT_SECONDS} --ictal {ICTAL_LABEL} --history {P.HISTORY_MODE}")
    pids = sorted(f.name.split("_")[0] for f in P.FEATURES_DIR.glob("chb*_ov0.25.parquet"))
    raw = {pid: P.load_patient(pid) for pid in pids}
    log(f"Loaded {len(pids)} patients")

    run_loso(raw)
    copy_baseline()
    report()
    log("Finished")


if __name__ == "__main__":
    run()
