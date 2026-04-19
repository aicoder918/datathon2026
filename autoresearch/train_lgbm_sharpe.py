"""Direct-Sharpe LightGBM via PyTorch autograd (SOTA_reaseach.txt method).
LGBM tree ensemble trained end-to-end on batch Sharpe ratio."""
from __future__ import annotations
import argparse, time, numpy as np, pandas as pd
import torch
import lightgbm as lgb

from autoresearch.prepare import (
    STABLE_FEATURES,
    build_features, compute_train_target,
    load_finbert, load_test, load_train, sharpe,
)
from autoresearch._extra_feats import build_extra, EXTRA_ALL
from autoresearch._template_feats import (
    build_template_features_oof, build_template_features_test,
)

TPL_BASE_FEATS = ["tpl_impact", "tpl_impact_late"]
TPL_KS = (3, 5, 10)
TPL_FEATS_USE = [f"{f}_k{k}" for k in TPL_KS for f in TPL_BASE_FEATS]
ALL_FEATURES = list(STABLE_FEATURES) + list(EXTRA_ALL) + TPL_FEATS_USE

TPL_PRIOR_N = 30
TPL_TAU = 2.5
TPL_LATE_START = 0.44


def make_sharpe_obj(out_scale: float = 2.5):
    def fobj(preds, dataset):
        targets = dataset.get_label()
        preds_t = torch.tensor(preds, dtype=torch.float32, requires_grad=True)
        r_t = torch.tensor(targets, dtype=torch.float32)
        pos = out_scale * torch.tanh(preds_t)
        pnl = pos * r_t
        mu = pnl.mean()
        sd = pnl.std(unbiased=True) + 1e-6
        loss = -mu / sd
        loss.backward()
        grad = preds_t.grad.detach().numpy().astype(np.float64)
        hess = np.ones_like(grad)
        return grad, hess
    return fobj


def make_sharpe_eval(out_scale: float = 2.5):
    def feval(preds, dataset):
        targets = dataset.get_label()
        pos = out_scale * np.tanh(preds)
        pnl = pos * targets
        mu = pnl.mean(); sd = pnl.std() + 1e-6
        return ("sharpe", mu / sd, True)
    return feval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lgbm_ds.csv")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--out-scale", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    tr = load_train()
    fb = load_finbert()

    X_base = build_features(tr["bars_seen"], tr["headlines_seen"], finbert=fb)
    X_ex = build_extra(tr["bars_seen"], tr["headlines_seen"], finbert=fb).reindex(X_base.index)
    parts = []
    for K in TPL_KS:
        ft = build_template_features_oof(
            tr["headlines_seen"], tr["bars_seen"], tr["bars_unseen"],
            sessions=X_base.index.to_numpy(), n_splits=5, K=K,
            prior_n=TPL_PRIOR_N, tau=TPL_TAU, late_start=TPL_LATE_START, seed=args.seed,
        ).reindex(X_base.index)[TPL_BASE_FEATS].add_suffix(f"_k{K}")
        parts.append(ft)
    X_tr = pd.concat([X_base, X_ex, pd.concat(parts, axis=1)], axis=1)
    r = compute_train_target(tr["bars_seen"], tr["bars_unseen"]).reindex(X_tr.index).to_numpy()
    print(f"train: n={len(X_tr)} feats={len(ALL_FEATURES)}")

    # LGBM dataset
    Xarr = X_tr[ALL_FEATURES].to_numpy()
    dtrain = lgb.Dataset(Xarr, label=r.astype(np.float32))
    params = {
        "objective": "regression",
        "learning_rate": args.lr,
        "num_leaves": args.leaves,
        "max_depth": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 40,
        "verbosity": -1,
        "seed": args.seed,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        fobj=make_sharpe_obj(args.out_scale),
        feval=make_sharpe_eval(args.out_scale),
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=max(args.rounds // 10, 20))],
    )

    pred_tr = model.predict(Xarr)
    pos_tr = args.out_scale * np.tanh(pred_tr)
    pnl_tr = pos_tr * r
    sh = pnl_tr.mean() / (pnl_tr.std() + 1e-6)
    print(f"in-sample sharpe: {sh:.4f}")

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
    Xq = build_features(te["prv_bars"], te["prv_head"], finbert=fb)
    Xq_ex = build_extra(te["prv_bars"], te["prv_head"], finbert=fb).reindex(Xq.index)
    Xq_tpl = _bt(te["prv_head"], Xq.index.to_numpy())
    X_test = pd.concat([pd.concat([Xp, Xp_ex, Xp_tpl], axis=1),
                        pd.concat([Xq, Xq_ex, Xq_tpl], axis=1)])

    pred_te = model.predict(X_test[ALL_FEATURES].to_numpy())
    pos_te = args.out_scale * np.tanh(pred_te)
    target = np.maximum(0.1, 1.0 + pos_te)
    print(f"test target mean/std/min/max: {target.mean():+.3f}/{target.std():.3f}/{target.min():+.3f}/{target.max():+.3f}")
    out = pd.DataFrame({"session": X_test.index, "target_position": target})
    out.to_csv(f"autoresearch/submissions/{args.name}", index=False)
    print(f"wrote autoresearch/submissions/{args.name}  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
