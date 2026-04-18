"""Mean head (Ridge) + variance head (log-linear) + position rule.

Sharpe-optimal position per plan:
    p_raw = (mu_hat + gamma) / sigma2_hat
    p_raw = clip(p_raw, q01, q99)     # winsorize
    p_model = p_raw / median(|p_raw|)  # normalize scale (Sharpe-invariant)
    p = (1 - lam) * p_const + lam * p_model    # shrink toward constant-long

Keep everything small and regularized: N=1000 training sessions is very little.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from models.features import FEATURE_COLS, VOL_FEATURES


# --------------------------------------------------------------- Sharpe metric

def sharpe(pnl: np.ndarray) -> float:
    s = float(np.std(pnl, ddof=1))
    if s <= 1e-18:
        return 0.0
    return float(np.mean(pnl) / s * 16.0)


# --------------------------------------------------------------- mean head

@dataclass
class MeanHead:
    alpha: float = 10.0
    features: list = field(default_factory=lambda: list(FEATURE_COLS))
    scaler: StandardScaler | None = None
    model: Ridge | None = None

    def fit(self, X: pd.DataFrame, r: np.ndarray) -> "MeanHead":
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X[self.features].to_numpy())
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model.fit(Xs, r)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X[self.features].to_numpy())
        return self.model.predict(Xs)

    def coefs(self) -> dict:
        return dict(zip(self.features, self.model.coef_))


# Stable subset: features whose Ridge coef sign agreed >= 9/10 outer folds on v2.
# Pruned: acf1, rv_late_ratio, rv_last10, sent_exp_w, last_hl_signed, body_frac,
#         trend_slope_norm, ret_last10, rv  (rv kept for variance head separately).
STABLE_FEATURES = [
    "ret_total", "ret_last5", "ret_last20", "pk_vol", "drawdown",
    "n_headlines", "sent_sum", "sent_recent_w", "sent_last10",
    "sent_neg_sum", "last_hl_age", "sent_change", "neg_frac",
]


@dataclass
class CatMeanHead:
    """Strongly-regularized CatBoost regressor as alternative mean head.

    Tiny model: shallow trees, few iterations, strong L2. Aimed at picking up
    interactions Ridge misses without exploding overfit at N=1000.
    """
    iterations: int = 300
    depth: int = 3
    learning_rate: float = 0.03
    l2_leaf_reg: float = 10.0
    features: list = field(default_factory=lambda: list(FEATURE_COLS))
    model: object = None

    def fit(self, X: pd.DataFrame, r: np.ndarray) -> "CatMeanHead":
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg,
            loss_function="RMSE", verbose=False, allow_writing_files=False,
            random_seed=42,
        )
        self.model.fit(X[self.features].to_numpy(), r)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features].to_numpy())

    def coefs(self) -> dict:
        # Use feature importances (signed by mean direction) as a stand-in.
        try:
            imp = self.model.get_feature_importance()
            return dict(zip(self.features, imp))
        except Exception:
            return {f: 0.0 for f in self.features}


# --------------------------------------------------------------- variance head

@dataclass
class VarianceHead:
    """log(r^2) ~ linear in a small vol-relevant feature subset."""
    alpha: float = 5.0
    features: list = field(default_factory=lambda: list(VOL_FEATURES))
    scaler: StandardScaler | None = None
    model: Ridge | None = None
    intercept_var: float = 0.0
    floor: float = 1e-8

    def fit(self, X: pd.DataFrame, r: np.ndarray) -> "VarianceHead":
        self.scaler = StandardScaler()
        Xv = self.scaler.fit_transform(X[self.features].to_numpy())
        # log((r - r_bar)^2) as the target; use a floor to keep it finite
        resid = r - float(np.mean(r))
        y = np.log(np.maximum(resid ** 2, self.floor))
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model.fit(Xv, y)
        self.intercept_var = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xv = self.scaler.transform(X[self.features].to_numpy())
        log_s2 = self.model.predict(Xv)
        s2 = np.exp(log_s2)
        floor = np.exp(self.intercept_var)  # median training variance
        return np.maximum(s2, floor)


# --------------------------------------------------------------- position rule

def positions_from_heads(
    mu: np.ndarray,
    sigma2: np.ndarray,
    gamma: float,
    lam: float,
    p_const: float = 1.0,
    winsor_q: tuple = (0.01, 0.99),
    ref_raw: np.ndarray | None = None,
) -> np.ndarray:
    """Convert (mu, sigma2) estimates to positions.

    ref_raw: if given, use it to compute the winsor clips + scale median.
    This is how we avoid double-dipping on the test set: fit the scale on
    training OOF predictions, apply it to the test predictions.
    """
    raw = (mu + gamma) / np.maximum(sigma2, 1e-12)
    if ref_raw is None:
        ref_raw = raw
    qlo = float(np.quantile(ref_raw, winsor_q[0]))
    qhi = float(np.quantile(ref_raw, winsor_q[1]))
    raw = np.clip(raw, qlo, qhi)
    med = float(np.median(np.abs(ref_raw.clip(qlo, qhi))))
    if med <= 1e-12:
        med = 1.0
    p_model = raw / med
    return (1.0 - lam) * p_const + lam * p_model
