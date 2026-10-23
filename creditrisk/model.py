"""Обучение GBDT-модели, калибровка, метрики качества и шкала скорингового балла."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from . import config as C

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    HAS_CATBOOST = False


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class SigmoidCalibrator:
    """Платт-калибровка: монотонное преобразование -> ранжирование (AUC) не меняется."""

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "SigmoidCalibrator":
        self._lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw).reshape(-1, 1), y)
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]


@dataclass
class ScoringModel:
    """Оценщик + калибратор под одним интерфейсом: один и тот же PD в тесте и в калькуляторе."""
    estimator: object
    calibrator: SigmoidCalibrator
    features: list[str]
    backend: str

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(X[self.features])[:, 1]
        return np.clip(self.calibrator.predict(raw), 0.001, 0.999)


@dataclass
class TrainResult:
    model: ScoringModel
    test_df: pd.DataFrame          # тестовая выборка + pd_score + score_points
    importances: pd.Series
    metrics: dict


# ------------------------------------------------------------ шкала баллов
def pd_to_score(pd_prob, base_score: int = C.SCORE_BASE, base_pd: float = C.SCORE_BASE_PD,
                pdo: int = C.SCORE_PDO) -> np.ndarray:
    """Классическая шкала: base_score баллов при PD=base_pd, каждые pdo баллов — odds ×2."""
    p = np.clip(np.asarray(pd_prob, dtype=float), 1e-5, 1 - 1e-5)
    factor = pdo / np.log(2)
    offset = base_score + factor * np.log(base_pd / (1 - base_pd))
    score = offset - factor * np.log(p / (1 - p))
    return np.clip(score, C.SCORE_MIN, C.SCORE_MAX).round(0).astype(int)


# ------------------------------------------------------------------ метрики
def ks_statistic(y_true, p) -> tuple[float, float]:
    fpr, tpr, thr = roc_curve(y_true, p)
    i = int(np.argmax(tpr - fpr))
    return float(tpr[i] - fpr[i]), float(thr[i])


def ks_curve(scored: pd.DataFrame) -> pd.DataFrame:
    """Кумулятивные доли bads/goods по возрастанию PD — данные для KS-графика."""
    d = scored.sort_values("pd_score")
    bad = (d[C.TARGET] == 1).to_numpy()
    return pd.DataFrame({
        "pd_score": d["pd_score"].to_numpy(),
        "cum_bad": np.cumsum(bad) / bad.sum(),
        "cum_good": np.cumsum(~bad) / (~bad).sum(),
    })


def calibration_table(scored: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Средний предсказанный PD vs фактический default rate по децилям PD."""
    bucket = pd.qcut(scored["pd_score"], n_bins, duplicates="drop")
    return (scored.groupby(bucket, observed=True)
            .agg(pred=("pd_score", "mean"), obs=(C.TARGET, "mean")).reset_index(drop=True))


def bootstrap_auc_ci(y, p, n_boot: int = 200, seed: int = C.RANDOM_STATE, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return tuple(np.quantile(aucs, [alpha / 2, 1 - alpha / 2]))


def permutation_importance_auc(predict: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame,
                               y, n_repeats: int = 5, seed: int = C.RANDOM_STATE) -> pd.Series:
    """Падение AUC при перемешивании признака (модель-независимо, считается на valid)."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict(X))
    out = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - roc_auc_score(y, predict(Xp)))
        out[col] = float(np.mean(drops))
    return pd.Series(out).sort_values()


# ------------------------------------------------------------------ обучение
def _fit_estimator(backend: str, X_tr, y_tr, X_va, y_va, seed: int):
    if backend == "catboost":
        est = CatBoostClassifier(iterations=800, learning_rate=0.05, depth=5, eval_metric="AUC",
                                 early_stopping_rounds=50, random_seed=seed, verbose=False)
        est.fit(X_tr, y_tr, eval_set=(X_va, y_va))   # valid — только для early stopping
    else:
        est = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=5,
                                             early_stopping=True, validation_fraction=0.15,
                                             random_state=seed)
        est.fit(X_tr, y_tr)
    return est


def train_scoring_model(df: pd.DataFrame, features: list[str] = C.FEATURES,
                        backend: str = "auto", seed: int = C.RANDOM_STATE) -> TrainResult:
    """train 60% / valid 20% / test 20% (стратифицированно).

    * train — обучение; valid — early stopping и калибровка; test — ТОЛЬКО финальные метрики.
    * без class weights: вероятности остаются честными, калибратор лишь подгоняет их.
    """
    if backend == "auto":
        backend = "catboost" if HAS_CATBOOST else "hgb"
    if backend == "catboost" and not HAS_CATBOOST:
        raise ImportError("catboost не установлен: pip install catboost или backend='hgb'")

    X, y = df[features], df[C.TARGET].to_numpy()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed,
                                              stratify=y_tmp)

    est = _fit_estimator(backend, X_tr, y_tr, X_va, y_va, seed)
    calibrator = SigmoidCalibrator().fit(est.predict_proba(X_va)[:, 1], y_va)
    model = ScoringModel(est, calibrator, list(features), backend)

    p_te = model.predict_pd(X_te)
    auc = roc_auc_score(y_te, p_te)
    ks, ks_thr = ks_statistic(y_te, p_te)
    base_rate = float(np.mean(y_tr))
    brier = brier_score_loss(y_te, p_te)
    lo, hi = bootstrap_auc_ci(y_te, p_te)

    test_df = df.loc[X_te.index].copy()
    test_df["pd_score"] = p_te
    test_df["score_points"] = pd_to_score(p_te)

    metrics = {
        "backend": backend, "n_train": len(X_tr), "n_valid": len(X_va), "n_test": len(X_te),
        "auc": auc, "auc_ci": (lo, hi), "gini": 2 * auc - 1, "ks": ks, "ks_threshold": ks_thr,
        "brier": brier, "brier_baseline": base_rate * (1 - base_rate),
        "mean_pd": float(p_te.mean()), "observed_rate": float(np.mean(y_te)),
    }
    importances = permutation_importance_auc(model.predict_pd, X_va, y_va)
    return TrainResult(model, test_df, importances, metrics)
