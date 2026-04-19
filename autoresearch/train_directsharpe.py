"""Direct-Sharpe MLP trained on all features, writes submission.
Reuses autoresearch feature pipeline, swaps ridge-mean + log-var head for end-to-end."""
from __future__ import annotations
import argparse, time, numpy as np, pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from autoresearch.prepare import (
    STABLE_FEATURES, VOL_FEATURES,
    build_features, compute_train_target,
    load_finbert, load_test, load_train, sharpe,
    write_submission,
)
from autoresearch._extra_feats import build_extra, EXTRA_ALL
from autoresearch._template_feats import (
    build_template_features_oof, build_template_features_test, TPL_FEATURES,
)
from autoresearch._direct_sharpe import train_direct_sharpe, LinearHead, MLPHead, neg_sharpe_loss, _to_tensor

TPL_BASE_FEATS = ["tpl_impact", "tpl_impact_late"]
TPL_KS = (3, 5, 10)
TPL_FEATS_USE = [f"{f}_k{k}" for k in TPL_KS for f in TPL_BASE_FEATS]
ALL_FEATURES = list(STABLE_FEATURES) + list(EXTRA_ALL) + TPL_FEATS_USE

TPL_PRIOR_N = 30
TPL_TAU = 2.5
TPL_LATE_START = 0.44


def train_pipeline(X_tr_df: pd.DataFrame, r: np.ndarray, X_te_df: pd.DataFrame,
                   head: str, hidden: int, epochs: int, lr: float, wd: float,
                   out_scale: float, seed: int) -> np.ndarray:
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr_df[ALL_FEATURES].to_numpy())
    X_te = sc.transform(X_te_df[ALL_FEATURES].to_numpy())
    pos = train_direct_sharpe(
        X_tr, r, X_te, head=head, hidden=hidden,
        epochs=epochs, lr=lr, weight_decay=wd,
        seed=seed, out_scale=out_scale, verbose=False,
    )
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ds_submission.csv")
    ap.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--out-scale", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    tr = load_train()
    fb = load_finbert()

    X_tr_base = build_features(tr["bars_seen"], tr["headlines_seen"], finbert=fb)
    X_tr_ex = build_extra(tr["bars_seen"], tr["headlines_seen"], finbert=fb).reindex(X_tr_base.index)
    # Build OOF tpl features on train
    parts = []
    for K in TPL_KS:
        ft = build_template_features_oof(
            tr["headlines_seen"], tr["bars_seen"], tr["bars_unseen"],
            sessions=X_tr_base.index.to_numpy(), n_splits=5, K=K,
            prior_n=TPL_PRIOR_N, tau=TPL_TAU, late_start=TPL_LATE_START, seed=args.seed,
        ).reindex(X_tr_base.index)[TPL_BASE_FEATS].add_suffix(f"_k{K}")
        parts.append(ft)
    X_tr_tpl = pd.concat(parts, axis=1)
    X_tr = pd.concat([X_tr_base, X_tr_ex, X_tr_tpl], axis=1)

    r = compute_train_target(tr["bars_seen"], tr["bars_unseen"]).reindex(X_tr.index).to_numpy()
    print(f"train: n={len(X_tr)} feats={len(ALL_FEATURES)} r_mean={r.mean():+.5f} r_std={r.std():.5f}")

    te = load_test()
    def _bt(hd, sessions):
        parts = []
        for K in TPL_KS:
            ft = build_template_features_test(
                hd, sessions,
                bars_seen_tr=tr["bars_seen"], bars_unseen_tr=tr["bars_unseen"],
                headlines_tr=tr["headlines_seen"], K=K, prior_n=TPL_PRIOR_N,
                tau=TPL_TAU, late_start=TPL_LATE_START,
            ).reindex(sessions)[TPL_BASE_FEATS].add_suffix(f"_k{K}")
            parts.append(ft)
        return pd.concat(parts, axis=1)

    Xp = build_features(te["pub_bars"], te["pub_head"], finbert=fb)
    Xp_ex = build_extra(te["pub_bars"], te["pub_head"], finbert=fb).reindex(Xp.index)
    Xp_tpl = _bt(te["pub_head"], Xp.index.to_numpy())
    X_pub = pd.concat([Xp, Xp_ex, Xp_tpl], axis=1)

    Xq = build_features(te["prv_bars"], te["prv_head"], finbert=fb)
    Xq_ex = build_extra(te["prv_bars"], te["prv_head"], finbert=fb).reindex(Xq.index)
    Xq_tpl = _bt(te["prv_head"], Xq.index.to_numpy())
    X_prv = pd.concat([Xq, Xq_ex, Xq_tpl], axis=1)

    X_test = pd.concat([X_pub, X_prv])
    print(f"test: public={len(X_pub)} private={len(X_prv)} total={len(X_test)}")

    pos = train_pipeline(X_tr, r, X_test, head=args.head, hidden=args.hidden,
                         epochs=args.epochs, lr=args.lr, wd=args.wd,
                         out_scale=args.out_scale, seed=args.seed)

    # Diag: training-sample Sharpe
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr[ALL_FEATURES].to_numpy())
    pos_tr = train_direct_sharpe(
        Xs, r, Xs, head=args.head, hidden=args.hidden,
        epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
        seed=args.seed, out_scale=args.out_scale,
    )
    pnl = pos_tr * r
    print(f"train sharpe (in-sample): {sharpe(pnl):.4f} pos mean/std {pos_tr.mean():+.3f}/{pos_tr.std():.3f}")
    print(f"test  pos mean/std/min/max: {pos.mean():+.3f}/{pos.std():.3f}/{pos.min():+.3f}/{pos.max():+.3f}")

    # Convert position to target_position (>= 0.1): shift/clip to expected range
    # Competition expects positions roughly in [0.1, 4.0]+ — interpret pos as long-only magnitude
    # Simple mapping: target = max(0.1, 1 + pos) keeps mean ~1, centered
    target = np.maximum(0.1, 1.0 + pos)
    print(f"target mean/std/min/max: {target.mean():+.3f}/{target.std():.3f}/{target.min():+.3f}/{target.max():+.3f}")

    out_df = pd.DataFrame({"session": X_test.index, "target_position": target})
    out_df.to_csv(f"autoresearch/submissions/{args.name}", index=False)
    print(f"wrote autoresearch/submissions/{args.name}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
