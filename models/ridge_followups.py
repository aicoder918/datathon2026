"""Follow-up submissions after ridge_all beat CatBoost on LB.

Generates:
  - ridge_all_killshorts.csv
  - ridge_all_plus_catblend_50.csv
  - ridge_all_strong.csv
  - elasticnet_all.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from features import load_train, load_test, shape_positions, finalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_KIND = "thresholded_inv_vol"
THRESHOLD_Q = 0.35
RIDGE_ALPHAS_BASE = np.logspace(-3, 3, 13)
RIDGE_ALPHAS_STRONG = np.logspace(1, 6, 16)
ENET_L1_RATIOS = [0.05, 0.1, 0.2, 0.4, 0.6]


def fit_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    kind: str,
) -> tuple[np.ndarray, dict[str, float]]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    if kind == "ridge_base":
        model = RidgeCV(alphas=RIDGE_ALPHAS_BASE)
        model.fit(Xtr, y_train)
        pred = np.asarray(model.predict(Xte), dtype=float)
        return pred, {"alpha": float(model.alpha_)}

    if kind == "ridge_strong":
        model = RidgeCV(alphas=RIDGE_ALPHAS_STRONG)
        model.fit(Xtr, y_train)
        pred = np.asarray(model.predict(Xte), dtype=float)
        return pred, {"alpha": float(model.alpha_)}

    if kind == "enet_all":
        model = ElasticNetCV(
            l1_ratio=ENET_L1_RATIOS,
            cv=5,
            max_iter=10000,
            n_jobs=1,
            random_state=42,
        )
        model.fit(Xtr, y_train)
        pred = np.asarray(model.predict(Xte), dtype=float)
        nnz = float(np.count_nonzero(np.abs(model.coef_) > 1e-12))
        return pred, {"alpha": float(model.alpha_), "l1_ratio": float(model.l1_ratio_), "nnz": nnz}

    raise ValueError(kind)


def to_positions(pred: np.ndarray, test_vol: np.ndarray, killshorts: bool = False) -> np.ndarray:
    out = pred.copy()
    if killshorts:
        out[out < 0] = 0.0
    pos = shape_positions(out, test_vol, BEST_KIND, threshold_q=THRESHOLD_Q)
    return finalize(pos)


def save_submission(sessions: np.ndarray, positions: np.ndarray, name: str) -> pd.Series:
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": positions,
    })
    out_path = OUT_DIR / name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission.describe())
    return pd.Series(positions, index=sessions)


X_train_df, y_train_s = load_train()
X_test_df = load_test()
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

X_train = X_train_df.to_numpy(dtype=np.float64)
X_test = X_test_df.to_numpy(dtype=np.float64)
y_train = y_train_s.to_numpy(dtype=np.float64)
test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
sessions = X_test_df.index.to_numpy()

pred_ridge, meta_ridge = fit_linear(X_train, y_train, X_test, "ridge_base")
print(f"ridge_base alpha={meta_ridge['alpha']:.6f} pred mean={pred_ridge.mean():+.5f} std={pred_ridge.std():.5f}")

pos_ridge_kill = to_positions(pred_ridge, test_vol, killshorts=True)
save_submission(sessions, pos_ridge_kill, "ridge_all_killshorts.csv")

champ_path = OUT_DIR / "catboost_bars_seed50_killshorts_blend.csv"
champ = pd.read_csv(champ_path).set_index("session")["target_position"].reindex(sessions)
blend_50 = 0.5 * to_positions(pred_ridge, test_vol, killshorts=False) + 0.5 * champ.to_numpy(dtype=float)
save_submission(sessions, blend_50, "ridge_all_plus_catblend_50.csv")

pred_strong, meta_strong = fit_linear(X_train, y_train, X_test, "ridge_strong")
print(f"\nridge_strong alpha={meta_strong['alpha']:.6f} pred mean={pred_strong.mean():+.5f} std={pred_strong.std():.5f}")
save_submission(sessions, to_positions(pred_strong, test_vol, killshorts=False), "ridge_all_strong.csv")

pred_enet, meta_enet = fit_linear(X_train, y_train, X_test, "enet_all")
print(
    f"\nelasticnet_all alpha={meta_enet['alpha']:.6f} l1_ratio={meta_enet['l1_ratio']:.3f} "
    f"nnz={int(meta_enet['nnz'])} pred mean={pred_enet.mean():+.5f} std={pred_enet.std():.5f}"
)
save_submission(sessions, to_positions(pred_enet, test_vol, killshorts=False), "elasticnet_all.csv")
