from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODELS_DIR))

import baseline_template as bt  # noqa: E402
import template_z_linear as tz  # noqa: E402
from features import extract_event  # noqa: E402


def quantile_bin(series: pd.Series, n_bins: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    qs = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    edges = np.unique(qs)
    if len(edges) <= 2:
        return pd.Series(0, index=series.index)
    return pd.Series(np.digitize(values, edges[1:-1], right=False), index=series.index)


def context_feature(bars_seen: pd.DataFrame, kind: str) -> tuple[pd.Series, pd.Series]:
    if kind == "none":
        sessions = np.sort(bars_seen["session"].unique())
        base = pd.Series(0.0, index=sessions)
        bins = pd.Series(0, index=sessions)
        return base, bins

    base_kind = {
        "dir49_3": "dir_0_49",
        "meansign20_3": "mean_sign_20",
        "dir49_sign": "dir_0_49",
        "meansign20_sign": "mean_sign_20",
    }[kind]
    base = tz.direction_feature(bars_seen, base_kind)
    if kind.endswith("_3"):
        bins = quantile_bin(base, 3)
    elif kind.endswith("_sign"):
        bins = (base > base.median()).astype(int)
    else:
        raise ValueError(kind)
    return base.sort_index(), bins.sort_index()


def fit_contextual_template_stats(
    hls_train: pd.DataFrame,
    target: pd.Series,
    ctx_bins: pd.Series,
    prior_k: float,
    key_kind: str,
) -> tuple[pd.Series, pd.Series]:
    h = hls_train.copy()
    triples = [extract_event(headline) for headline in h["headline"]]
    h["_tid"] = [t[0] for t in triples]
    h["_sec"] = [t[1] for t in triples]
    h = h[(h["_tid"] >= 0) & (h["bar_ix"] <= 49)].copy()
    h["y"] = h["session"].map(target)
    h["ctx"] = h["session"].map(ctx_bins)

    global_mean = float(target.mean())
    global_std = float(target.std(ddof=0))

    tid_stats = h.groupby("_tid")["y"].agg(["mean", "std", "count"])
    tid_stats["std"] = tid_stats["std"].replace(0.0, np.nan).fillna(global_std)
    tid_stats["base"] = (tid_stats["mean"] - global_mean) / tid_stats["std"]

    if key_kind == "tid_ctx":
        pair_stats = h.groupby(["_tid", "ctx"])["y"].agg(["mean", "std", "count"])
    elif key_kind == "tidsec_ctx":
        pair_stats = h.groupby(["_tid", "_sec", "ctx"])["y"].agg(["mean", "std", "count"])
    else:
        raise ValueError(key_kind)
    pair_stats["std"] = pair_stats["std"].replace(0.0, np.nan).fillna(global_std)
    pair_stats = pair_stats.join(tid_stats["base"].rename("parent"), on="_tid")
    pair_stats["raw"] = (pair_stats["mean"] - global_mean) / pair_stats["std"]
    pair_stats["value"] = (
        pair_stats["count"] * pair_stats["raw"] + prior_k * pair_stats["parent"]
    ) / (pair_stats["count"] + prior_k)

    return tid_stats["base"], pair_stats["value"]


def session_contextual_template_feature(
    hls: pd.DataFrame,
    sessions: np.ndarray,
    ctx_bins: pd.Series,
    base: pd.Series,
    pair: pd.Series,
    agg: str,
    key_kind: str,
) -> pd.Series:
    h = hls.copy()
    triples = [extract_event(headline) for headline in h["headline"]]
    h["_tid"] = [t[0] for t in triples]
    h["_sec"] = [t[1] for t in triples]
    h = h[(h["_tid"] >= 0) & (h["bar_ix"] <= 49)].copy()
    h["ctx"] = h["session"].map(ctx_bins)
    if key_kind == "tid_ctx":
        h["value"] = [
            pair.get((int(t), int(c)), base.get(int(t), 0.0))
            for t, c in zip(h["_tid"], h["ctx"])
        ]
    elif key_kind == "tidsec_ctx":
        h["value"] = [
            pair.get((int(t), str(s), int(c)), base.get(int(t), 0.0))
            for t, s, c in zip(h["_tid"], h["_sec"], h["ctx"])
        ]
    else:
        raise ValueError(key_kind)

    if agg == "sum":
        score = h.groupby("session")["value"].sum()
    elif agg == "mean":
        score = h.groupby("session")["value"].mean()
    elif agg == "recent_sum":
        score = (h["value"] * np.exp(-(49.0 - h["bar_ix"].to_numpy()) / 10.0)).groupby(h["session"]).sum()
    else:
        raise ValueError(agg)

    out = pd.Series(0.0, index=sessions, name="template_contextual")
    out.loc[score.index] = score.to_numpy(dtype=float)
    return out.sort_index()


def build_feature_matrices(
    bars_seen_train: pd.DataFrame,
    bars_unseen_train: pd.DataFrame,
    hls_train: pd.DataFrame,
    bars_seen_test: pd.DataFrame,
    hls_test: pd.DataFrame,
    dir_kind: str,
    context_kind: str,
    agg: str,
    prior_k: float,
    key_kind: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    sessions_train = np.sort(bars_seen_train["session"].unique())
    sessions_test = np.sort(bars_seen_test["session"].unique())
    target = (
        bars_unseen_train.sort_values("bar_ix").groupby("session")["close"].last()
        / bars_seen_train.sort_values("bar_ix").groupby("session")["close"].last()
        - 1.0
    ).rename("y").reindex(sessions_train)

    dir_train = tz.direction_feature(bars_seen_train, dir_kind).reindex(sessions_train).fillna(0.0)
    dir_test = tz.direction_feature(bars_seen_test, dir_kind).reindex(sessions_test).fillna(0.0)

    _, ctx_train = context_feature(bars_seen_train, context_kind)
    _, ctx_test = context_feature(bars_seen_test, context_kind)
    ctx_train = ctx_train.reindex(sessions_train).fillna(0).astype(int)
    ctx_test = ctx_test.reindex(sessions_test).fillna(0).astype(int)

    base, pair = fit_contextual_template_stats(hls_train, target, ctx_train, prior_k=prior_k, key_kind=key_kind)
    tpl_train = session_contextual_template_feature(hls_train, sessions_train, ctx_train, base, pair, agg=agg, key_kind=key_kind)
    tpl_test = session_contextual_template_feature(hls_test, sessions_test, ctx_test, base, pair, agg=agg, key_kind=key_kind)

    z_tpl_train, z_tpl_test = tz.standardize(tpl_train, tpl_test)
    z_dir_train, z_dir_test = tz.standardize(dir_train, dir_test)
    X_train = pd.DataFrame({"template_ctx": z_tpl_train, "direction": z_dir_train}, index=sessions_train)
    X_test = pd.DataFrame({"template_ctx": z_tpl_test, "direction": z_dir_test}, index=sessions_test)
    return X_train, target, X_test


def fit_model(model_kind: str, X_train: pd.DataFrame, y_train: pd.Series):
    if model_kind == "ols":
        model = LinearRegression()
    elif model_kind == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 3, 25))
    else:
        raise ValueError(model_kind)
    model.fit(X_train, y_train)
    return model


def choose_alpha(pred_train: pd.Series, y_train: pd.Series, z_cap: float | None) -> tuple[float, float]:
    z_pred = bt.zscore_with_train_stats(pred_train, float(pred_train.mean()), float(pred_train.std(ddof=0)))
    if z_cap is not None:
        z_pred = z_pred.clip(-z_cap, z_cap)
    best = (-np.inf, 1.0)
    for alpha in [0.75, 1.0, 1.1, 1.25, 1.5]:
        score = bt.sharpe((1.0 + alpha * z_pred).to_numpy(dtype=float), y_train.to_numpy(dtype=float))
        if score > best[0]:
            best = (score, alpha)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=["ols", "ridge"], default="ols")
    parser.add_argument("--dir-kind", choices=["dir_0_49", "mean_sign_10", "mean_sign_20", "mean_bodyret_10"], default="mean_sign_20")
    parser.add_argument("--context-kind", choices=["none", "dir49_3", "meansign20_3", "dir49_sign", "meansign20_sign"], default="meansign20_3")
    parser.add_argument("--agg", choices=["sum", "mean", "recent_sum"], default="sum")
    parser.add_argument("--prior-k", type=float, default=20.0)
    parser.add_argument("--key-kind", choices=["tid_ctx", "tidsec_ctx"], default="tid_ctx")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--z-cap", type=float, default=2.5)
    args = parser.parse_args()

    bars_seen_train, bars_unseen_train, hls_train, bars_seen_test, hls_test = bt.load_data()
    X_train, y_train, X_test = build_feature_matrices(
        bars_seen_train=bars_seen_train,
        bars_unseen_train=bars_unseen_train,
        hls_train=hls_train,
        bars_seen_test=bars_seen_test,
        hls_test=hls_test,
        dir_kind=args.dir_kind,
        context_kind=args.context_kind,
        agg=args.agg,
        prior_k=args.prior_k,
        key_kind=args.key_kind,
    )

    model = fit_model(args.model, X_train, y_train)
    pred_train = pd.Series(model.predict(X_train), index=X_train.index)
    pred_test = pd.Series(model.predict(X_test), index=X_test.index)

    if args.alpha is None:
        best_sharpe, alpha = choose_alpha(pred_train, y_train, z_cap=args.z_cap)
        print(
            f"best in-sample sharpe={best_sharpe:.3f} alpha={alpha} model={args.model} "
            f"dir={args.dir_kind} ctx={args.context_kind} agg={args.agg} prior_k={args.prior_k} "
            f"key_kind={args.key_kind} z_cap={args.z_cap}"
        )
    else:
        alpha = float(args.alpha)
        z_pred = bt.zscore_with_train_stats(pred_train, float(pred_train.mean()), float(pred_train.std(ddof=0)))
        if args.z_cap is not None:
            z_pred = z_pred.clip(-args.z_cap, args.z_cap)
        score = bt.sharpe((1.0 + alpha * z_pred).to_numpy(dtype=float), y_train.to_numpy(dtype=float))
        print(
            f"forced in-sample sharpe={score:.3f} alpha={alpha} model={args.model} "
            f"dir={args.dir_kind} ctx={args.context_kind} agg={args.agg} prior_k={args.prior_k} "
            f"key_kind={args.key_kind} z_cap={args.z_cap}"
        )

    if args.model == "ridge":
        print(f"ridge_alpha={float(model.alpha_):.6f}")
    print("coefficients:")
    for name, coef in zip(X_train.columns, model.coef_):
        print(f"  {name}: {float(coef):+.6f}")

    z_pred_test = bt.zscore_with_train_stats(pred_test, float(pred_train.mean()), float(pred_train.std(ddof=0)))
    if args.z_cap is not None:
        z_pred_test = z_pred_test.clip(-args.z_cap, args.z_cap)
    submission = pd.DataFrame({
        "session": X_test.index.astype(int),
        "target_position": (1.0 + alpha * z_pred_test).to_numpy(dtype=float),
    }).sort_values("session")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    submission.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(submission["target_position"].describe().to_string())


if __name__ == "__main__":
    main()
