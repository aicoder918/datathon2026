from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import baseline_template as bt  # noqa: E402


def realized_vol_0_49(bars_seen: pd.DataFrame) -> pd.Series:
    bars = bars_seen.sort_values(["session", "bar_ix"])
    vol = bars.groupby("session")["close"].apply(
        lambda s: np.diff(np.log(s.to_numpy(dtype=float))).std() if len(s) > 1 else 0.0
    )
    return vol.rename("vol_0_49").sort_index()


def add_features(X: pd.DataFrame, vol: pd.Series, include_interaction: bool) -> pd.DataFrame:
    out = X.copy()
    out["vol_0_49"] = vol.reindex(out.index).fillna(0.0)
    if include_interaction:
        out["template_x_direction"] = out["template_score"] * out["direction_0_49"]
    return out


def standardize(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mu = train_df.mean()
    sd = train_df.std(ddof=0).replace(0.0, 1.0)
    return (train_df - mu) / sd, (test_df - mu) / sd


def fit_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    if model_name == "ols":
        model = LinearRegression()
    elif model_name == "ridgecv":
        model = RidgeCV(alphas=np.logspace(-3, 3, 25))
    else:
        raise ValueError(f"unknown model: {model_name}")
    model.fit(X_train, y_train)
    return model


def choose_alpha(z_pred_train: pd.Series, y_train: pd.Series) -> tuple[float, float]:
    best = (-np.inf, 0.5)
    for alpha in [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]:
        pos = 1.0 + alpha * z_pred_train
        s = bt.sharpe(pos.to_numpy(dtype=float), y_train.to_numpy(dtype=float))
        if s > best[0]:
            best = (s, alpha)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=["ols", "ridgecv"], default="ols")
    parser.add_argument("--with-interaction", action="store_true")
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()

    bars_seen_train, bars_unseen_train, hls_train, bars_seen_test, hls_test = bt.load_data()
    X_train_base, y_train, impacts = bt.build_train_features(bars_seen_train, bars_unseen_train, hls_train)
    X_test_base = bt.build_test_features(bars_seen_test, hls_test, impacts)

    vol_train = realized_vol_0_49(bars_seen_train)
    vol_test = realized_vol_0_49(bars_seen_test)

    X_train = add_features(X_train_base, vol_train, args.with_interaction)
    X_test = add_features(X_test_base, vol_test, args.with_interaction)
    X_train_std, X_test_std = standardize(X_train, X_test)

    model = fit_model(args.model, X_train_std, y_train)
    pred_train = pd.Series(model.predict(X_train_std), index=X_train_std.index)
    pred_test = pd.Series(model.predict(X_test_std), index=X_test_std.index)

    mu = float(pred_train.mean())
    sd = float(pred_train.std(ddof=0))
    z_pred_train = bt.zscore_with_train_stats(pred_train, mu, sd)
    z_pred_test = bt.zscore_with_train_stats(pred_test, mu, sd)

    if args.alpha is None:
        best_sharpe, alpha = choose_alpha(z_pred_train, y_train)
        print(f"best in-sample sharpe={best_sharpe:.3f} alpha={alpha}")
    else:
        alpha = float(args.alpha)
        pos_train = 1.0 + alpha * z_pred_train
        print(
            "forced params in-sample sharpe="
            f"{bt.sharpe(pos_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float)):.3f}"
        )

    if args.model == "ridgecv":
        print(f"ridge_alpha={float(model.alpha_):.6f}")

    print("coefficients:")
    for name, coef in zip(X_train_std.columns, model.coef_):
        print(f"  {name}: {float(coef):+.6f}")

    positions = 1.0 + alpha * z_pred_test
    submission = pd.DataFrame({
        "session": X_test_std.index.astype(int),
        "target_position": positions.to_numpy(dtype=float),
    }).sort_values("session")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    submission.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(submission["target_position"].describe().to_string())


if __name__ == "__main__":
    main()
