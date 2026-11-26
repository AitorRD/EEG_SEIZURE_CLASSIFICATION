"""
Chronos-2 as a seizure-prediction classifier on raw CHB-MIT EEG (LOSO).

Same protocol as loso_foundation_model_prediction.py: same windows, labels (onset within
(SPH, SPH + HORIZON] of the window end), ictal variants, folds, validation patients, max-F1
threshold on the validation patients and metrics. What changes is the input: instead of
tsfresh features, Chronos-2 reads the last CONTEXT_SECONDS of raw EEG before the window end
(18 bipolar channels, 100 Hz), with its group attention mixing the channels of one window.

Classification head (replaces the quantile forecasting head):
    per channel   [REG] token + mean of the context patch tokens of the last encoder layer
    per window    mean and max over the channels present + log amplitude of every channel
                  (Chronos-2 standardises each series, so amplitude is otherwise lost)
    output        one linear layer -> 2 logits

Two models per fold, trained only on the fold's training patients:
    chronos2_probe   frozen Chronos-2 + logistic regression on the head features (linear probe).
                     Embeddings do not depend on the fold, so they are computed once per patient.
    chronos2_lora    LP-FT: the head starts from the probe and is trained jointly with LoRA
                     adapters (W + alpha/r · B·A) on q/k/v/o of every time and group attention.
                     Early stopping on the validation patients; the test patient is never seen.

Run from main.py:
    python main.py --experiment chronos2 --context 30 --ictal excluded --device cuda:1
    # split folds across GPUs (results are per fold, re-runs skip finished folds):
    python main.py --experiment chronos2 --device cuda:0 --patients chb01 chb02 ...
"""

import functools
import importlib
import math
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint

import loso_foundation_model_prediction as P
from finetune_lora_tabicl import LoRAWeight
from loso_foundation_model_prediction import log

# ===========================================================================
# Configuration
# ===========================================================================
CONTEXT_SECONDS = 30
ICTAL_LABEL = "excluded"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "amazon/chronos-2"
MAX_SAMPLES = 8192              # Chronos-2 context length; longer inputs are decimated to fit
PATCH = 16

FS = 100
RAW_DIR = Path("data/raw/csv-data-v2")
CACHE_DIR = Path("data/processed/chbmit_raw_npy")
# Double-banana bipolar montage, present in every CHB-MIT session (or derivable from the
# referential montages some chb12 sessions switch to).
CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ"]
CHANNEL_ALIASES = {"T8-P8": ["T8-P8-0", "T8-P8"]}
ELECTRODE_ALIASES = {"O1": ["O1", "01"]}

PROBE_C = [1e-3, 1e-2, 1e-1, 1.0]   # inverse L2 strength, chosen on the validation patients
SIGMA_FLOOR = 0.1                    # feature std floor, as a fraction of the median feature std

LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGETS = (".q", ".k", ".v", ".o")   # MHA projections of the time and group attentions
STEPS = 600
BATCH_WINDOWS = 8
POSITIVE_FRACTION = 0.25
LEARNING_RATE = 1e-4            # LoRA
HEAD_LEARNING_RATE = 1e-4       # head starts from the probe, so it only needs small updates
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EVAL_EVERY = 100
PATIENCE = 3
VAL_EVAL_WINDOWS = 2000
EARLY_STOP_METRIC = "ROC AUC"   # or "PR AUC"
INFER_BATCH_WINDOWS = 16        # group attention is quadratic in windows × channels

SEED = P.SEED
OUTPUT_DIR = BASELINE_DIR = None   # set by set_variant()
BASELINE_MODELS = P.MODELS


# ===========================================================================
# Raw EEG
# ===========================================================================
def _read_session(csv_path):
    columns = pd.read_csv(csv_path, nrows=0).columns
    df = pd.read_csv(csv_path, engine="pyarrow")
    out = np.full((len(df), len(CHANNELS)), np.nan, np.float32)
    referential = {c.split("-")[0].upper(): c for c in columns if c.endswith("-CS2")}
    referential |= {c.upper(): c for c in columns if "-" not in c and c.upper() not in
                    ("TIME (S)", "SEIZURE", "IDPATIENT", "IDSESSION")}

    def electrode(name):
        for alias in ELECTRODE_ALIASES.get(name, [name]):
            if alias in referential:
                return df[referential[alias]].to_numpy(np.float64)
        return None

    for j, channel in enumerate(CHANNELS):
        bipolar = next((c for c in CHANNEL_ALIASES.get(channel, [channel]) if c in columns), None)
        if bipolar is not None:
            out[:, j] = df[bipolar].to_numpy(np.float64) * 1e6   # CSV values are ~1e-8; rescale for float32
            continue
        a, b = (electrode(e) for e in channel.split("-"))
        if a is not None and b is not None:
            out[:, j] = (a - b) * 1e6
    return out


@functools.lru_cache(maxsize=128)
def session_signal(session):
    """(samples, channels) float32 (CSV units × 1e6), NaN where a channel is missing; cached as .npy."""
    path = CACHE_DIR / f"{session}.npy"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pid = session[:5]   # chb17a_03_clipped lives in chb17/
        signal = _read_session(RAW_DIR / pid / f"{session}.csv")
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, signal)
        tmp.rename(path)
    return np.load(path, mmap_mode="r")


def input_length(context_seconds):
    decimate = max(1, math.ceil(context_seconds * FS / MAX_SAMPLES))
    return decimate, PATCH * (context_seconds * FS // decimate // PATCH)


def load_patient(pid, context_seconds):
    windows, patient = P.load_windows(pid)
    end_row = windows["start_row"].to_numpy() + int(P.WINDOW_SEC * FS)
    # History may reach back to the first gap-free window of the segment, not across a gap.
    first_row = (pd.Series(np.where(windows["gap"], np.iinfo(np.int64).max, windows["start_row"]))
                 .groupby(patient["segment"]).transform("min").to_numpy())
    first_row = np.minimum(first_row, end_row - int(P.WINDOW_SEC * FS))
    return {**patient, "session": windows["session"].to_numpy(), "end_row": end_row, "first_row": first_row}


def window_batch(patient, rows, context_seconds):
    """(len(rows), channels, length) float32 with NaN where there is no EEG."""
    decimate, length = input_length(context_seconds)
    span = length * decimate
    out = np.full((len(rows), len(CHANNELS), span), np.nan, np.float32)
    for k, r in enumerate(rows):
        signal = session_signal(patient["session"][r])
        start = max(patient["end_row"][r] - span, patient["first_row"][r])
        piece = signal[start:patient["end_row"][r]].T
        out[k, :, span - piece.shape[1]:] = piece
    if decimate > 1:   # mean over blocks: anti-aliasing low-pass + downsample
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # blocks without EEG stay NaN
            out = np.nanmean(out.reshape(len(rows), len(CHANNELS), length, decimate), axis=-1)
    return out


# ===========================================================================
# Model
# ===========================================================================
def import_chronos():
    # A torchvision/torchaudio build that does not match torch breaks `import transformers`.
    # Chronos needs neither: if they fail to import, hide them so transformers skips them.
    for name in ("torchvision", "torchaudio"):
        try:
            importlib.import_module(name)
        except Exception:
            for loaded in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
                del sys.modules[loaded]
            sys.modules[name] = None
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline


class Chronos2Classifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        d = backbone.model_dim
        self.n_features = 4 * d + len(CHANNELS)
        self.head = nn.Linear(self.n_features, 2)
        # Standardisation of the head features (fitted on the training patients, fixed).
        self.register_buffer("mu", torch.zeros(self.n_features))
        self.register_buffer("sigma", torch.ones(self.n_features))

    def enable_checkpointing(self):
        for block in self.backbone.encoder.block:
            block.forward = functools.partial(
                lambda f, h, **kw: checkpoint(f, h, use_reentrant=False, **kw), block.forward)

    def features(self, x):
        """x: (windows, channels, length) with NaN for missing EEG -> (windows, n_features)."""
        n, c, length = x.shape
        series = x.reshape(n * c, length)
        observed = torch.isfinite(series)
        groups = torch.arange(n, device=x.device).repeat_interleave(c)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            out, (_, scale), _, n_patches = self.backbone.encode(context=series, group_ids=groups)
        hidden = out.last_hidden_state.float()
        patch_mask = observed.reshape(n * c, n_patches, PATCH).any(-1).float()
        patch_mean = (hidden[:, :n_patches] * patch_mask[..., None]).sum(1) / patch_mask.sum(1, keepdim=True).clamp(min=1)
        per_channel = torch.cat([hidden[:, n_patches], patch_mean], -1).reshape(n, c, -1)

        present = observed.any(-1).reshape(n, c)
        weights = present.float()[..., None]
        mean = (per_channel * weights).sum(1) / weights.sum(1).clamp(min=1)
        maximum = per_channel.masked_fill(~present[..., None], -torch.inf).amax(1)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
        log_amplitude = torch.where(present, torch.log(scale.float().reshape(n, c).clamp(min=1e-3)), 0.0)
        return torch.cat([mean, maximum, log_amplitude], -1)

    def forward(self, x):
        return self.head((self.features(x) - self.mu) / self.sigma)


def load_backbone():
    """Chronos-2 with its pretrained weights, also under transformers>=5.

    chronos-forecasting requires transformers<5, but the venv has 5.x (Mitra needs it). Under 5.x
    `from_pretrained` re-initialises every weight at random after reading the checkpoint, and
    leaves the non-persistent buffers (RoPE inv_freq, quantiles) as uninitialised memory, with
    no warning. So the checkpoint is loaded explicitly and the buffers are rebuilt."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    model = import_chronos().from_pretrained(MODEL_ID, device_map="cpu").model
    snapshot = Path(snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json"]))
    state = {}
    for f in sorted(snapshot.glob("*.safetensors")):
        state |= load_file(f)
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        model.quantiles.copy_(torch.tensor(model.chronos_config.quantiles, dtype=model.quantiles.dtype))
        for module in model.modules():
            if hasattr(module, "inv_freq") and hasattr(module, "base"):
                module.inv_freq.copy_(1.0 / (module.base ** (
                    torch.arange(0, module.dim, 2, dtype=torch.int64).float() / module.dim)))
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items()), "Chronos-2 weights not loaded"
    return model.to(DEVICE)


def build_model():
    model = Chronos2Classifier(load_backbone()).to(DEVICE)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def embed(model, patient, rows, context_seconds):
    model.eval()
    feats = []
    for s in range(0, len(rows), INFER_BATCH_WINDOWS):
        x = torch.from_numpy(window_batch(patient, rows[s:s + INFER_BATCH_WINDOWS], context_seconds)).to(DEVICE)
        feats.append(model.features(x).cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, model.n_features), np.float32)


@torch.no_grad()
def predict(model, patient, rows, context_seconds):
    model.eval()
    probs = []
    with parametrize.cached():
        for s in range(0, len(rows), INFER_BATCH_WINDOWS):
            x = torch.from_numpy(window_batch(patient, rows[s:s + INFER_BATCH_WINDOWS], context_seconds)).to(DEVICE)
            probs.append(torch.softmax(model(x).float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


# ===========================================================================
# Frozen embeddings + linear probe
# ===========================================================================
def patient_embeddings(model, pid, patient):
    """Head features of every usable window; frozen backbone, so shared by every fold."""
    path = OUTPUT_DIR / "embeddings" / f"{pid}.npy"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = np.flatnonzero(patient["usable"])
        feats = embed(model, patient, rows, CONTEXT_SECONDS).astype(np.float32)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, feats)
        tmp.rename(path)
        log(f"  embeddings {pid}: {feats.shape}")
    return np.load(path)


def score(y, p):
    return roc_auc_score(y, p) if EARLY_STOP_METRIC == "ROC AUC" else average_precision_score(y, p)


def standardisation(X):
    """Mean and std per feature, with the std floored so near-constant features (tiny train
    variance) are not blown up into large, patient-specific values."""
    mu = X.mean(0, dtype=np.float64)
    sigma = X.std(0, dtype=np.float64)
    sigma = np.maximum(sigma, SIGMA_FLOOR * np.median(sigma[sigma > 0]))
    return mu.astype(np.float32), sigma.astype(np.float32)


def fit_probe(X_tr, y_tr, X_val, y_val):
    mu, sigma = standardisation(X_tr)
    Z_tr, Z_val = (X_tr - mu) / sigma, (X_val - mu) / sigma
    best = None
    for C in PROBE_C:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Z_tr, y_tr)
        s = score(y_val, clf.predict_proba(Z_val)[:, 1])
        log(f"    probe C={C:g}: val {EARLY_STOP_METRIC}={s:.4f}")
        if best is None or s > best[0]:
            best = (s, C, clf)
    _, C, clf = best
    head = {"mu": mu, "sigma": sigma, "w": clf.coef_[0], "b": clf.intercept_[0]}
    return C, head, lambda X: clf.predict_proba((X - mu) / sigma)[:, 1]


def init_head(model, head):
    """Head = probe: standardisation as fixed buffers, logits [0, w·z + b] (same probabilities)."""
    with torch.no_grad():
        model.mu.copy_(torch.from_numpy(head["mu"]))
        model.sigma.copy_(torch.from_numpy(head["sigma"]))
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.weight[1] = torch.from_numpy(head["w"]).float()
        model.head.bias[1] = float(head["b"])


# ===========================================================================
# LoRA
# ===========================================================================
def inject_lora(model):
    names = []
    for name, module in model.backbone.encoder.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(LORA_TARGETS):
            parametrize.register_parametrization(module, "weight", LoRAWeight(module.weight, LORA_RANK, LORA_ALPHA))
            names.append(name)
    lora = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in lora:
        p.requires_grad_(True)
    n_total = sum(p.numel() for n, p in model.backbone.named_parameters() if "lora_" not in n)
    log(f"    LoRA r={LORA_RANK} alpha={LORA_ALPHA}: {len(names)} matrices, "
        f"{sum(p.numel() for p in lora):,} trainable / {n_total:,} pretrained")
    return lora


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, value in state.items():
            params[n].copy_(value)


class TrainSampler:
    """Batches of training windows with a fixed fraction of positives."""

    def __init__(self, train, rng):
        self.train, self.rng = train, rng
        self.index = [(k, r) for k, p in enumerate(train) for r in np.flatnonzero(p["usable"])]
        labels = np.array([train[k]["y"][r] for k, r in self.index])
        self.pos, self.neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)

    def batch(self):
        n_pos = max(1, round(BATCH_WINDOWS * POSITIVE_FRACTION))
        picks = np.concatenate([self.rng.choice(self.pos, n_pos), self.rng.choice(self.neg, BATCH_WINDOWS - n_pos)])
        x = np.concatenate([window_batch(self.train[self.index[i][0]], [self.index[i][1]], CONTEXT_SECONDS)
                            for i in picks])
        y = np.array([self.train[self.index[i][0]]["y"][self.index[i][1]] for i in picks])
        return torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).long().to(DEVICE)


def val_rows(val, rng, limit=None):
    """(patient index, row) of the validation windows, all or a subset with every positive."""
    index = [(k, r) for k, p in enumerate(val) for r in np.flatnonzero(p["usable"])]
    if limit is None or len(index) <= limit:
        return index
    y = np.array([val[k]["y"][r] for k, r in index])
    pos = np.flatnonzero(y == 1)
    neg = rng.choice(np.flatnonzero(y == 0), max(0, limit - len(pos)), replace=False)
    return [index[i] for i in np.sort(np.concatenate([pos, neg]))]


def predict_rows(model, val, index):
    proba = np.empty(len(index))
    for k in range(len(val)):
        at = [i for i, (kk, _) in enumerate(index) if kk == k]
        if at:
            proba[at] = predict(model, val[k], np.array([index[i][1] for i in at]), CONTEXT_SECONDS)
    return proba


def finetune(model, train, val, rng):
    lora = inject_lora(model)
    model.head.requires_grad_(True)
    model.enable_checkpointing()
    optimizer = torch.optim.AdamW([{"params": lora, "lr": LEARNING_RATE},
                                   {"params": model.head.parameters(), "lr": HEAD_LEARNING_RATE}],
                                  weight_decay=WEIGHT_DECAY)
    sampler = TrainSampler(train, rng)
    sub = val_rows(val, rng, VAL_EVAL_WINDOWS)
    y_sub = np.array([val[k]["y"][r] for k, r in sub])

    best_score = score(y_sub, predict_rows(model, val, sub))
    best_step, best_state = 0, trainable_state(model)
    history = [{"step": 0, "loss": np.nan, f"val {EARLY_STOP_METRIC}": best_score}]
    log(f"    step 0 (probe head): val {EARLY_STOP_METRIC}={best_score:.4f}")
    losses, evals_without_gain = [], 0
    for step in range(1, STEPS + 1):
        model.train()
        x, y = sampler.batch()
        loss = F.cross_entropy(model(x).float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], GRAD_CLIP)
        optimizer.step()
        losses.append(loss.item())
        if step % EVAL_EVERY and step != STEPS:
            continue
        val_score = score(y_sub, predict_rows(model, val, sub))
        history.append({"step": step, "loss": np.mean(losses), f"val {EARLY_STOP_METRIC}": val_score})
        log(f"    step {step}: loss={np.mean(losses):.4f} val {EARLY_STOP_METRIC}={val_score:.4f}")
        losses = []
        if val_score > best_score:
            best_score, best_step, best_state, evals_without_gain = val_score, step, trainable_state(model), 0
        else:
            evals_without_gain += 1
            if evals_without_gain >= PATIENCE:
                break
    load_trainable_state(model, best_state)
    log(f"    best step {best_step}: val {EARLY_STOP_METRIC}={best_score:.4f}")
    return best_state, best_step, pd.DataFrame(history)


# ===========================================================================
# LOSO
# ===========================================================================
def save_fold(key, test_pid, test, proba, thr, n_train):
    u = test["usable"]
    fold = pd.DataFrame({"y": test["y"][u], "ictal": test["ictal"][u], "tto": test["tto"][u],
                         "event": test["event"][u], "proba": proba, "threshold": thr, "context_rows": n_train})
    fold.to_parquet(OUTPUT_DIR / "fold_results" / f"{key}__{test_pid}.parquet", index=False)
    log(f"  {key}: ROC={roc_auc_score(fold['y'], proba):.3f}")


def run_loso(data, patients=None):
    pids = list(data)
    for folder in ("fold_results", "adapters", "training"):
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)

    todo = [p for p in pids if (patients is None or p in patients) and data[p]["y"][data[p]["usable"]].any()]
    model = None
    for i, test_pid in enumerate(pids):
        if test_pid not in todo:
            continue
        done = {k: (OUTPUT_DIR / "fold_results" / f"{k}__{test_pid}.parquet").exists()
                for k in ("chronos2_probe", "chronos2_lora")}
        if all(done.values()):
            continue
        # Same rng sequence as loso_foundation_model_prediction.run_loso -> same validation patients.
        rng = np.random.RandomState(SEED + i)
        others = [p for p in pids if p != test_pid]
        val_pids = sorted(rng.choice([p for p in others if data[p]["y"][data[p]["usable"]].any()],
                                     P.N_VAL_PATIENTS, replace=False))
        train_pids = [p for p in others if p not in val_pids]
        train, val, test = [data[p] for p in train_pids], [data[p] for p in val_pids], data[test_pid]
        y_tr = np.concatenate([p["y"][p["usable"]] for p in train])
        y_val = np.concatenate([p["y"][p["usable"]] for p in val])
        log(f"[Chronos-2 · EEG {CONTEXT_SECONDS}s · ictal={ICTAL_LABEL}] Fold {i + 1}/{len(pids)} {test_pid}: "
            f"{len(y_tr)} train windows ({int(y_tr.sum())} positives), val {val_pids}")

        if model is None:
            model = build_model()
        emb = {p: patient_embeddings(model, p, data[p]) for p in pids}
        C, head, probe = fit_probe(np.concatenate([emb[p] for p in train_pids]), y_tr,
                                   np.concatenate([emb[p] for p in val_pids]), y_val)
        if not done["chronos2_probe"]:
            thr = P.best_f1_threshold(y_val, probe(np.concatenate([emb[p] for p in val_pids])))
            save_fold("chronos2_probe", test_pid, test, probe(emb[test_pid]), thr, len(y_tr))
        if done["chronos2_lora"]:
            continue

        model = None
        P.release_gpu()
        torch.manual_seed(SEED + i)
        try:
            ft = build_model()
            init_head(ft, head)
            state, best_step, history = finetune(ft, train, val, np.random.RandomState(SEED + 1000 + i))
            all_val = val_rows(val, None)
            thr = P.best_f1_threshold(np.array([val[k]["y"][r] for k, r in all_val]), predict_rows(ft, val, all_val))
            proba = predict(ft, test, np.flatnonzero(test["usable"]), CONTEXT_SECONDS)
        except Exception as e:
            reason = "out of GPU memory" if "OutOfMemory" in type(e).__name__ else f"{type(e).__name__}: {e}"
            log(f"  chronos2_lora: skipped ({reason})")
            continue
        finally:
            ft = None
            P.release_gpu()
        torch.save({"state": state, "probe_C": C, "mu": head["mu"], "sigma": head["sigma"]},OUTPUT_DIR / "adapters" / f"{test_pid}.pt")
        history.assign(best_step=best_step).to_csv(OUTPUT_DIR / "training" / f"{test_pid}.csv", index=False)
        save_fold("chronos2_lora", test_pid, test, proba, thr, len(y_tr))


# ===========================================================================
# Report: Chronos-2 vs. the tabular foundation models on the same folds
# ===========================================================================
def copy_baseline():
    for key in BASELINE_MODELS:
        for f in sorted((BASELINE_DIR / "fold_results").glob(f"{key}__*.parquet")):
            target = OUTPUT_DIR / "fold_results" / f.name
            if not target.exists():
                shutil.copy(f, target)


def report():
    predictions = {}
    for key in (*BASELINE_MODELS, "chronos2_probe", "chronos2_lora"):
        files = sorted((OUTPUT_DIR / "fold_results").glob(f"{key}__*.parquet"))
        if files:
            predictions[key] = {f.stem.split("__")[1]: pd.read_parquet(f) for f in files}
    chronos = [k for k in predictions if k.startswith("chronos2")]
    if not chronos:
        log(f"No results in {OUTPUT_DIR}")
        return
    common = set.intersection(*(set(predictions[k]) for k in chronos))
    predictions = {k: {p: f for p, f in folds.items() if p in common} for k, folds in predictions.items()}
    predictions = {k: v for k, v in predictions.items() if len(v) == len(common)}

    metrics = P.metrics_per_fold(predictions)
    metrics.to_csv(OUTPUT_DIR / "loso_fold_metrics.csv", index=False)
    P.write_markdown(metrics, ICTAL_LABEL, CONTEXT_SECONDS, OUTPUT_DIR)
    P.write_figure(predictions, OUTPUT_DIR)
    P.write_metrics_table(metrics, ICTAL_LABEL, CONTEXT_SECONDS, OUTPUT_DIR)
    log(f"Report ({len(common)} folds): {OUTPUT_DIR / 'loso_report.md'}")


def set_variant(context_seconds, ictal_label, device):
    global CONTEXT_SECONDS, ICTAL_LABEL, DEVICE, OUTPUT_DIR, BASELINE_DIR
    CONTEXT_SECONDS, ICTAL_LABEL, DEVICE = context_seconds, ictal_label, device or DEVICE
    BASELINE_DIR = P.variant_dir(CONTEXT_SECONDS, ICTAL_LABEL, "stats")
    OUTPUT_DIR = (P.OUTPUT_DIR.parent / f"loso_chronos2_chbmit_h{P.HORIZON_SEC:g}_sph{P.SPH_SEC:g}"
                  / f"context_{CONTEXT_SECONDS}s" / f"ictal_{ICTAL_LABEL}")


def run(context_seconds=None, ictal_label=None, device=None, patients=None):
    set_variant(context_seconds or CONTEXT_SECONDS, ICTAL_LABEL if ictal_label is None else ictal_label, device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    P.LOG_FILES[:] = [OUTPUT_DIR / "log.txt"]
    decimate, length = input_length(CONTEXT_SECONDS)
    log(f"Start: Chronos-2 · EEG {CONTEXT_SECONDS}s ({length} samples at {FS / decimate:g} Hz × "
        f"{len(CHANNELS)} channels) · ictal {ICTAL_LABEL} · {DEVICE} · LoRA r={LORA_RANK} alpha={LORA_ALPHA} "
        f"· {STEPS} steps × {BATCH_WINDOWS} windows lr={LEARNING_RATE}")
    pids = sorted(f.name.split("_")[0] for f in (P.FEATURES_DIR / "preictal_meta").glob("chb*_ov0.25.parquet"))
    data = {pid: P.with_ictal_label(load_patient(pid, CONTEXT_SECONDS), ICTAL_LABEL) for pid in pids}
    log(f"Loaded {len(pids)} patients")

    run_loso(data, patients)
    copy_baseline()
    report()
    log("Finished")


if __name__ == "__main__":
    run()
