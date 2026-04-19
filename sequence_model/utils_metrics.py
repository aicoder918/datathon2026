"""Metrics, checkpoint, and seed utilities."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def sharpe(positions, returns) -> float:
    """Competition Sharpe (annualization factor 16)."""
    positions = np.asarray(positions, dtype=float)
    returns = np.asarray(returns, dtype=float)
    pnl = positions * returns
    std = pnl.std()
    if std < 1e-12:
        return 0.0
    return float(pnl.mean() / std * 16)


def save_checkpoint(path, model, optimizer: Optional[torch.optim.Optimizer] = None,
                    meta: Optional[dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "meta": meta or {},
    }
    torch.save(state, path)
    return path


def load_checkpoint(path, model, optimizer=None, map_location=None,
                    strict: bool = True, exclude_prefixes=()) -> dict:
    state = torch.load(path, map_location=map_location)
    model_state = state["model_state"]
    if exclude_prefixes:
        model_state = {k: v for k, v in model_state.items()
                       if not any(k.startswith(p) for p in exclude_prefixes)}
    model.load_state_dict(model_state, strict=strict)
    if optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    return state.get("meta", {})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
