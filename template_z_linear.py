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
from features import extract_event  # noqa: E402


def direction_feature(bars_seen: pd.DataFrame, kind: str) -> pd.Series:
    bars = bars_seen.sort_values(["session", "bar_ix"]).copy()
    g = bars.groupby("session", sort=True)
    bars["bar_ret"] = bars.groupby("session")["close"].pct_change().fillna(0.0)
    bars["body_ret"] = (bars["close"] / bars["open"] - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    bars["dir_sign"] = np.sign(bars["body_ret"])

    def ret_to(df: pd.DataFrame, idx: int) -> float:
        j = min(idx, len(df) - 1)
        return float(df["close"].iloc[j] / df["close"].iloc[0] - 1.0)

    def mean_prefix(col: str, n: int) -> pd.Series:
        sub = bars[bars["bar_ix"] < n]
        return sub.groupby("session")[col].mean().rename(f"{col}_{n}").sort_index()

    if kind == "dir_0_49":
        return (g["close"].last() / g["close"].first() - 1.0).rename(kind).sort_index()
    if kind == "dir_0_04":
        return g.apply(lambda df: ret_to(df, 4)).rename(kind).sort_index()
    if kind == "dir_0_09":
        return g.apply(lambda df: ret_to(df, 9)).rename(kind).sort_index()
    if kind == "dir_0_19":
        return g.apply(lambda df: ret_to(df, 19)).rename(kind).sort_index()
    if kind == "mean_sign_10":
        return mean_prefix("dir_sign", 10)
    if kind == "mean_sign_20":
        return mean_prefix("dir_sign", 20)
    if kind == "mean_bodyret_10":
        return mean_prefix("body_ret", 10)
    raise ValueError(kind)


def fit_template_stats(
    hls_train: pd.DataFrame,
    target: pd.Series,
    shrink_k: float,
) -> pd.DataFrame:
    h = hls_train.copy()
    h["_tid"] = [extract_event(headline)[0] for headline in h["headline"]]
    h = h[(h["_tid"] >= 0) & (h["bar_ix"] <= 49)].copy()
    h["y"] = h["session"].map(target)

    stats = h.groupby("_tid")["y"].agg(["mean", "std", "count"])
    global_mean = float(target.mean())
    global_std = float(target.std(ddof=0))
    stats["std"] = stats["std"].replace(0.0, np.nan).fillna(global_std)
    stats["z_raw"] = (stats["mean"] - global_mean) / stats["std"]
    if shrink_k > 0:
        weight = np.sqrt(stats["count"] / (stats["count"] + shrink_k))
        stats["z_score"] = stats["z_raw"] * weight
    else:
        stats["z_score"] = stats["z_raw"]
    return stats


def session_template_feature(
    hls: pd.DataFrame,
    stats: pd.DataFrame,
    sessions: np.ndarray,
    agg: str,
) -> pd.Series:
    h = hls.copy()
    h["_tid"] = [extract_event(headline)[0] for headline in h["headline"]]
    h = h[(h["_tid"] >= 0) & (h["bar_ix"] <= 49)].copy()
    h["value"] = h["_tid"].map(stats["z_score"]).fillna(0.0)

    if agg == "sum":
        score = h.groupby("session")["value"].sum()
    elif agg == "mean":
        score = h.groupby("session")["value"].mean()
    else:
        raise ValueError(agg)

    out = pd.Series(0.0, index=sessions, name=f"template_z_{agg}")
    out.loc[score.index] = score.to_numpy(dtype=float)
    return out.sort_index()


def standardize(train_s: pd.Series, test_s: pd.Series) -> tuple[pd.Series, pd.Series]:
    mu = float(train_s.mean())
    sd = float(train_s.std(ddof=0))
    return bt.zscore_with_train_stats(train_s, mu, sd), bt.zscore_with_train_stats(test_s, mu, sd)


def build_feature_matrices(
    bars_seen_train: pd.DataFrame,
    bars_unseen_train: pd.DataFrame,
    hls_train: pd.DataFrame,
    bars_seen_test: pd.DataFrame,
    hls_test: pd.DataFrame,
    dir_kind: str,
    agg: str,
    shrink_k: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    sessions_train = np.sort(bars_seen_train["session"].unique())
    sessions_test = np.sort(bars_seen_test["session"].unique())

    target = (
        bars_unseen_train.sort_values("bar_ix").groupby("session")["close"].last()
        / bars_seen_train.sort_values("bar_ix").groupby("session")["close"].last()
        - 1.0
    ).rename("y").reindex(sessions_train)

    stats = fit_template_stats(hls_train, target, shrink_k=shrink_k)
    tpl_train = session_template_feature(hls_train, stats, sessions_train, agg=agg)
    tpl_test = session_template_feature(hls_test, stats, sessions_test, agg=agg)
    dir_train = direction_feature(bars_seen_train, dir_kind).reindex(sessions_train).fillna(0.0)
    dir_test = direction_feature(bars_seen_test, dir_kind).reindex(sessions_test).fillna(0.0)

    z_tpl_train, z_tpl_test = standardize(tpl_train, tpl_test)
    z_dir_train, z_dir_test = standardize(dir_train, dir_test)

    X_train = pd.DataFrame({"template_z": z_tpl_train, "direction": z_dir_train}, index=sessions_train)
    X_test = pd.DataFrame({"template_z": z_tpl_test, "direction": z_dir_test}, index=sessions_test)
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


def choose_alpha(pred_train: pd.Series, y_train: pd.Series, z_cap: float | None = None) -> tuple[float, float]:
    z_pred_train = bt.zscore_with_train_stats(
        pred_train, float(pred_train.mean()), float(pred_train.std(ddof=0))
    )
    if z_cap is not None:
        z_pred_train = z_pred_train.clip(-z_cap, z_cap)
    best = (-np.inf, 1.0)
    for alpha in [0.50, 0.75, 1.00, 1.25, 1.50]:
        pos = 1.0 + alpha * z_pred_train
        score = bt.sharpe(pos.to_numpy(dtype=float), y_train.to_numpy(dtype=float))
        if score > best[0]:
            best = (score, alpha)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=["ols", "ridge"], default="ols")
    parser.add_argument(
        "--dir-kind",
        choices=["dir_0_49", "dir_0_04", "dir_0_09", "dir_0_19", "mean_sign_10", "mean_sign_20", "mean_bodyret_10"],
        default="dir_0_49",
    )
    parser.add_argument("--agg", choices=["sum", "mean"], default="sum")
    parser.add_argument("--shrink-k", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--z-cap", type=float, default=None)
    args = parser.parse_args()

    bars_seen_train, bars_unseen_train, hls_train, bars_seen_test, hls_test = bt.load_data()
    X_train, y_train, X_test = build_feature_matrices(
        bars_seen_train=bars_seen_train,
        bars_unseen_train=bars_unseen_train,
        hls_train=hls_train,
        bars_seen_test=bars_seen_test,
        hls_test=hls_test,
        dir_kind=args.dir_kind,
        agg=args.agg,
        shrink_k=args.shrink_k,
    )

    model = fit_model(args.model, X_train, y_train)
    pred_train = pd.Series(model.predict(X_train), index=X_train.index)
    pred_test = pd.Series(model.predict(X_test), index=X_test.index)

    if args.alpha is None:
        best_sharpe, alpha = choose_alpha(pred_train, y_train, z_cap=args.z_cap)
        print(
            f"best in-sample sharpe={best_sharpe:.3f} alpha={alpha} "
            f"model={args.model} dir={args.dir_kind} agg={args.agg} shrink_k={args.shrink_k} z_cap={args.z_cap}"
        )
    else:
        alpha = float(args.alpha)
        z_pred_train = bt.zscore_with_train_stats(
            pred_train, float(pred_train.mean()), float(pred_train.std(ddof=0))
        )
        if args.z_cap is not None:
            z_pred_train = z_pred_train.clip(-args.z_cap, args.z_cap)
        score = bt.sharpe((1.0 + alpha * z_pred_train).to_numpy(dtype=float), y_train.to_numpy(dtype=float))
        print(
            f"forced in-sample sharpe={score:.3f} alpha={alpha} "
            f"model={args.model} dir={args.dir_kind} agg={args.agg} shrink_k={args.shrink_k} z_cap={args.z_cap}"
        )

    if args.model == "ridge":
        print(f"ridge_alpha={float(model.alpha_):.6f}")
    print("coefficients:")
    for name, coef in zip(X_train.columns, model.coef_):
        print(f"  {name}: {float(coef):+.6f}")

    z_pred_test = bt.zscore_with_train_stats(
        pred_test, float(pred_train.mean()), float(pred_train.std(ddof=0))
    )
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
