"""Data loading and dataset builders for the sequence model.

Inputs per session (first 50 bars):
  - bar features: ret, hl_range, close_norm, oc_ratio,
                  rolling_vol_5, rolling_mom_5                 (6 channels)
  - headline features: impact, news_count_5                    (2 channels)
Total input dim = 6 + 2 = 8.

All 8 channels are z-scored using per-channel stats computed once on the SSL
corpus (train_seen + public + private) and stored in
`sequence_model/ckpt/feature_stats.npz`. This both gives the encoder meaningful
gradients and makes the recon task non-trivial (predicting 0 in standardized
space gives MSE ~1 per channel, not ~0).

`impact[template_id]` is computed once from **train sessions only**:
    impact[tid] = mean over all train headlines of template tid of
                  (close[t+5]/close[t] - 1) - session_mean_5bar_return

Supervised target: y = close[99] / close[49] - 1 (available only for train).
Next-return SSL head predicts sign(ret_{t+1}) via BCE (not regression).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "hrt-eth-zurich-datathon-2026" / "data"

N_TEMPLATES = 50
SEQ_LEN = 50
EPS = 1e-8

BAR_FEATURE_DIM = 6   # ret, hl_range, close_norm, oc_ratio, rolling_vol_5, rolling_mom_5
HEADLINE_FEATURE_DIM = 2  # impact, news_count_5
INPUT_DIM = BAR_FEATURE_DIM + HEADLINE_FEATURE_DIM
FWD_IMPACT_HORIZON = 5
ROLLING_WINDOW = 5

HEADLINE_PARQUETS = {
    "train": [
        PROJECT_ROOT / "headline_features_train.parquet",
        PROJECT_ROOT / "headline_features.parquet",
    ],
    "public": [PROJECT_ROOT / "headline_features_public.parquet"],
    "private": [PROJECT_ROOT / "headline_features_private.parquet"],
}


# --------------------------------------------------------------------------- #
# Headline amplitude parsing
# --------------------------------------------------------------------------- #

_DOLLAR_RE = re.compile(r"\$([\d.]+)([BMK]?)")
_PCT_RE = re.compile(r"([\d.]+)%")


def _parse_dollar(s) -> float:
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return 0.0
    m = _DOLLAR_RE.match(str(s))
    if not m:
        return 0.0
    val = float(m.group(1))
    suf = m.group(2)
    if suf == "B":
        val *= 1000.0
    elif suf == "K":
        val *= 0.001
    return val


def _parse_percentage(s) -> float:
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return 0.0
    m = _PCT_RE.match(str(s))
    return float(m.group(1)) if m else 0.0


# --------------------------------------------------------------------------- #
# Raw loaders
# --------------------------------------------------------------------------- #

def load_bars(split: str) -> pd.DataFrame:
    if split == "train_seen":
        return pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    if split == "train_full":
        seen = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
        unseen = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
        return (pd.concat([seen, unseen])
                .sort_values(["session", "bar_ix"])
                .reset_index(drop=True))
    if split == "public":
        return pd.read_parquet(DATA_DIR / "bars_seen_public_test.parquet")
    if split == "private":
        return pd.read_parquet(DATA_DIR / "bars_seen_private_test.parquet")
    raise ValueError(f"unknown bars split: {split}")


def load_headlines(split: str) -> pd.DataFrame:
    """Load preprocessed headline features and attach an `amplitude` column."""
    paths = HEADLINE_PARQUETS.get(split)
    if paths is None:
        raise ValueError(f"unknown headline split: {split}")
    df = None
    for p in paths:
        if p.exists():
            df = pd.read_parquet(p)
            break
    if df is None:
        raise FileNotFoundError(f"No headline parquet for split={split}")
    cols = ["session", "bar_ix", "template_index", "dollar", "percentage"]
    df = df[cols].copy()
    df["template_index"] = (pd.to_numeric(df["template_index"], errors="coerce")
                              .fillna(-1).astype(int))
    df["bar_ix"] = (pd.to_numeric(df["bar_ix"], errors="coerce")
                      .fillna(-1).astype(int))
    df["session"] = pd.to_numeric(df["session"], errors="coerce").astype(int)
    dollar_val = df["dollar"].apply(_parse_dollar) / 9600.0
    pct_val = df["percentage"].apply(_parse_percentage) / 25.0
    dollar_val = dollar_val.clip(0, 1)
    pct_val = pct_val.clip(0, 1)
    df["amplitude"] = np.where(dollar_val > 0, dollar_val,
                               np.where(pct_val > 0, pct_val, 0.5)).astype(np.float32)
    df = df[df["bar_ix"].between(0, SEQ_LEN - 1)].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Per-session feature builders
# --------------------------------------------------------------------------- #

def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling std with shrinking window at the start (no future leak)."""
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        chunk = arr[start:i + 1]
        out[i] = float(np.std(chunk)) if len(chunk) > 1 else 0.0
    return out


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling sum with shrinking window at the start."""
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = float(np.sum(arr[start:i + 1]))
    return out


def build_bar_feature_matrix(session_bars: pd.DataFrame,
                             max_bars: int = SEQ_LEN) -> np.ndarray:
    bars = session_bars.sort_values("bar_ix").head(max_bars)
    o = bars["open"].to_numpy(dtype=np.float32)
    h = bars["high"].to_numpy(dtype=np.float32)
    low = bars["low"].to_numpy(dtype=np.float32)
    c = bars["close"].to_numpy(dtype=np.float32)
    denom = np.abs(c) + EPS

    ret = np.zeros_like(c)
    if len(c) > 1:
        ret[1:] = c[1:] / (c[:-1] + EPS) - 1.0
    hl_range = (h - low) / denom
    c0 = c[0] if len(c) else 1.0
    close_norm = c / (c0 + EPS) - 1.0
    oc_ratio = (c - o) / denom

    rolling_vol = _rolling_std(ret, ROLLING_WINDOW)
    rolling_mom = _rolling_sum(ret, ROLLING_WINDOW)

    feats = np.stack(
        [ret, hl_range, close_norm, oc_ratio, rolling_vol, rolling_mom],
        axis=1,
    ).astype(np.float32)
    if feats.shape[0] < max_bars:
        pad = np.zeros((max_bars - feats.shape[0], BAR_FEATURE_DIM), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=0)
    return feats


def compute_template_impacts(bars_full_df: pd.DataFrame,
                             hls_train_df: pd.DataFrame,
                             fwd_horizon: int = FWD_IMPACT_HORIZON,
                             n_templates: int = N_TEMPLATES
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-template average forward-H-bar return minus session baseline.

    Uses TRAIN-ONLY data. For each train headline at (session s, bar t) with
    template tid:
        adj = (close[t+H]/close[t] - 1) - session_baseline[s]
    where session_baseline[s] is the mean of (close[t+H]/close[t] - 1) over all
    valid t in session s.

    Returns (impacts [n_templates], counts [n_templates]).
    """
    close_by_session: Dict[int, np.ndarray] = {}
    for s, g in bars_full_df.groupby("session"):
        g = g.sort_values("bar_ix")
        close_by_session[int(s)] = g["close"].to_numpy(dtype=np.float64)

    session_baseline: Dict[int, float] = {}
    for s, c in close_by_session.items():
        if len(c) <= fwd_horizon:
            session_baseline[s] = 0.0
            continue
        fwd = c[fwd_horizon:] / (c[:-fwd_horizon] + EPS) - 1.0
        session_baseline[s] = float(fwd.mean())

    sums = np.zeros(n_templates, dtype=np.float64)
    counts = np.zeros(n_templates, dtype=np.int64)
    for row in hls_train_df.itertuples(index=False):
        s = int(row.session)
        t = int(row.bar_ix)
        tid = int(row.template_index)
        if tid < 0 or tid >= n_templates:
            continue
        c = close_by_session.get(s)
        if c is None or t + fwd_horizon >= len(c) or t < 0:
            continue
        fwd_ret = c[t + fwd_horizon] / (c[t] + EPS) - 1.0
        sums[tid] += fwd_ret - session_baseline.get(s, 0.0)
        counts[tid] += 1

    impacts = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0).astype(np.float32)
    return impacts, counts


def build_headline_impact_matrix(session_hls: pd.DataFrame,
                                 template_impacts: np.ndarray,
                                 max_bars: int = SEQ_LEN) -> np.ndarray:
    """Per-bar headline signal.

    Channel 0: mean of impact[tid] over headlines firing in that bar.
    Channel 1: rolling count of headlines over the last ROLLING_WINDOW bars,
               normalized by ROLLING_WINDOW (so ~1.0 means one headline per bar).

    Shape: (max_bars, HEADLINE_FEATURE_DIM). Bars with no headlines get 0
    on the impact channel; the count channel is always non-negative.
    """
    feats = np.zeros((max_bars, HEADLINE_FEATURE_DIM), dtype=np.float32)
    n_templates = len(template_impacts)
    per_bar_count = np.zeros(max_bars, dtype=np.float32)
    if len(session_hls) > 0:
        for t, group in session_hls.groupby("bar_ix"):
            t = int(t)
            if t < 0 or t >= max_bars:
                continue
            tids = group["template_index"].to_numpy(dtype=int)
            tids = tids[(tids >= 0) & (tids < n_templates)]
            per_bar_count[t] = float(len(tids))
            if len(tids) > 0:
                feats[t, 0] = float(np.mean(template_impacts[tids]))
    rolling_count = _rolling_sum(per_bar_count, ROLLING_WINDOW) / float(ROLLING_WINDOW)
    feats[:, 1] = rolling_count
    return feats


def build_session_inputs(bars_df: pd.DataFrame,
                         hls_df: pd.DataFrame,
                         template_impacts: np.ndarray,
                         max_bars: int = SEQ_LEN
                         ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    sessions = sorted(bars_df["session"].unique())
    bars_grouped = {int(s): g for s, g in bars_df.groupby("session")}
    hl_grouped = {int(s): g for s, g in hls_df.groupby("session")}
    empty_hls = hls_df.iloc[0:0]
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for s in sessions:
        s = int(s)
        bf = build_bar_feature_matrix(bars_grouped[s], max_bars=max_bars)
        hf = build_headline_impact_matrix(hl_grouped.get(s, empty_hls),
                                          template_impacts, max_bars=max_bars)
        out[s] = (bf, hf)
    return out


def compute_feature_stats(session_inputs: Dict[int, Tuple[np.ndarray, np.ndarray]]
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std across all (session, bar) positions.

    Returns (mean[INPUT_DIM], std[INPUT_DIM]) as float32. std has a small floor
    added so channels with zero variance don't divide-by-zero.
    """
    bars_list, hls_list = [], []
    for bf, hf in session_inputs.values():
        bars_list.append(bf)
        hls_list.append(hf)
    bars = np.concatenate(bars_list, axis=0)   # (N*T, bar_dim)
    hls = np.concatenate(hls_list, axis=0)     # (N*T, hl_dim)
    x = np.concatenate([bars, hls], axis=1)    # (N*T, INPUT_DIM)
    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def save_feature_stats(path, mean: np.ndarray, std: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=np.asarray(mean, dtype=np.float32),
             std=np.asarray(std, dtype=np.float32))


def load_feature_stats(path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["mean"].astype(np.float32), d["std"].astype(np.float32)


def compute_supervised_targets(bars_full_df: pd.DataFrame) -> Dict[int, float]:
    """y = close[99] / close[49] - 1 for train sessions with 100 bars."""
    out: Dict[int, float] = {}
    for s, g in bars_full_df.groupby("session"):
        g = g.sort_values("bar_ix")
        if len(g) < 100:
            continue
        c49 = float(g.iloc[49]["close"])
        c99 = float(g.iloc[99]["close"])
        out[int(s)] = c99 / (c49 + EPS) - 1.0
    return out


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

class SessionSSLDataset(Dataset):
    """Masked-modeling + sign(next-return) SSL targets.

    Inputs are z-scored with (feature_mean, feature_std). Masked bar channels
    are zeroed *in standardized space* (i.e. replaced with the per-channel
    mean). Recon target is the standardized bar features — predicting 0 then
    costs ~1 per channel, not ~0 like on raw bp-scale data.
    """

    def __init__(self, session_inputs: Dict,
                 feature_mean: np.ndarray, feature_std: np.ndarray,
                 mask_prob: float = 0.15):
        self.keys = list(session_inputs.keys())
        self.inputs = session_inputs
        self.mean = np.asarray(feature_mean, dtype=np.float32)
        self.std = np.asarray(feature_std, dtype=np.float32)
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        bf, hf = self.inputs[key]
        x_raw = np.concatenate([bf, hf], axis=1).astype(np.float32)   # (T, 8)
        x_std = ((x_raw - self.mean) / self.std).astype(np.float32)

        T = x_std.shape[0]
        mask = (np.random.rand(T) < self.mask_prob).astype(np.float32)
        masked_x = x_std.copy()
        masked_x[mask > 0, :BAR_FEATURE_DIM] = 0.0  # 0 in std-space = mean in raw

        recon_target = x_std[:, :BAR_FEATURE_DIM].copy()   # standardized bar feats
        sign_target = (bf[:, 0] > 0).astype(np.float32)    # sign of raw return

        return {
            "key": str(key),
            "x": torch.from_numpy(masked_x),
            "recon_target": torch.from_numpy(recon_target),
            "sign_target": torch.from_numpy(sign_target),
            "mask": torch.from_numpy(mask),
        }


class SessionSupervisedDataset(Dataset):
    """Emits (x, y, session, sign_target). Inputs are z-scored identically to
    the SSL pipeline. y may be a dummy 0.0 for inference."""

    def __init__(self, session_inputs: Dict[int, Tuple[np.ndarray, np.ndarray]],
                 session_targets: Dict[int, float],
                 feature_mean: np.ndarray, feature_std: np.ndarray):
        self.sessions = [int(s) for s in session_inputs.keys() if int(s) in session_targets]
        self.inputs = session_inputs
        self.targets = session_targets
        self.mean = np.asarray(feature_mean, dtype=np.float32)
        self.std = np.asarray(feature_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx):
        s = self.sessions[idx]
        bf, hf = self.inputs[s]
        x_raw = np.concatenate([bf, hf], axis=1).astype(np.float32)
        x_std = ((x_raw - self.mean) / self.std).astype(np.float32)
        sign_target = (bf[:, 0] > 0).astype(np.float32)
        return {
            "session": int(s),
            "x": torch.from_numpy(x_std),
            "y": torch.tensor(float(self.targets[s]), dtype=torch.float32),
            "sign_target": torch.from_numpy(sign_target),
        }


def collate_ssl(batch):
    return {
        "key": [b["key"] for b in batch],
        "x": torch.stack([b["x"] for b in batch]),
        "recon_target": torch.stack([b["recon_target"] for b in batch]),
        "sign_target": torch.stack([b["sign_target"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }


def collate_supervised(batch):
    return {
        "session": torch.tensor([b["session"] for b in batch], dtype=torch.long),
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "sign_target": torch.stack([b["sign_target"] for b in batch]),
    }
