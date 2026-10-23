"""Бизнес-экономика: прибыль портфеля, оптимальный cut-off, решение по одному заемщику."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass(frozen=True)
class Scenario:
    """Бизнес-параметры, которые меняет пользователь в сайдбаре."""
    lgd: float
    margin: float
    cutoff: float


def _pnl(df: pd.DataFrame, lgd: float, margin: float) -> np.ndarray:
    """P&L каждого кредита, если он выдан: маржа с «хорошего», потеря LGD с «плохого»."""
    return np.where(df[C.TARGET] == 0, df["loan_amount"] * margin, -df["loan_amount"] * lgd)


def portfolio_summary(scored: pd.DataFrame, cutoff: float, lgd: float, margin: float) -> dict:
    approved = scored["pd_score"] < cutoff
    n = int(approved.sum())
    return {
        "approval_rate": float(approved.mean()),
        "npl_rate": float(scored.loc[approved, C.TARGET].mean()) if n else 0.0,
        "net_profit": float(_pnl(scored[approved], lgd, margin).sum()),
    }


def profit_curve(scored: pd.DataFrame, grid: np.ndarray, lgd: float, margin: float) -> pd.DataFrame:
    """Прибыль для каждого порога за один проход (сортировка + кумулятивная сумма)."""
    d = scored.sort_values("pd_score")
    cum = np.concatenate([[0.0], np.cumsum(_pnl(d, lgd, margin))])
    n_approved = np.searchsorted(d["pd_score"].to_numpy(), grid, side="left")  # PD < cutoff
    return pd.DataFrame({"cutoff": grid, "net_profit": cum[n_approved],
                         "approval_rate": n_approved / len(d)})


@dataclass(frozen=True)
class Decision:
    pd_value: float
    expected_loss: float
    expected_profit: float
    approved: bool


def underwrite(pd_value: float, loan: float, lgd: float, margin: float, cutoff: float) -> Decision:
    expected_loss = loan * pd_value * lgd
    expected_profit = loan * (1 - pd_value) * margin - expected_loss
    return Decision(pd_value, expected_loss, expected_profit, pd_value < cutoff)
