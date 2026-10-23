"""Загрузка, очистка и feature engineering датасета Give Me Some Credit.

Принципы:
* никаких «подмен» данных: нет файла/сети -> явная ошибка или явно помеченные синтетические данные;
* очистка только построчная и детерминированная (без статистик по всей выборке),
  поэтому утечки из теста в трейн через импутацию нет. Пропуски оставляем как NaN —
  CatBoost и HistGradientBoosting умеют с ними работать, а факт пропуска дохода
  дополнительно кодируется флагом is_income_missing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config as C

RAW_TARGET = "SeriousDlqin2yrs"
RAW_COLUMNS = [
    RAW_TARGET, "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
]
_DELINQ_RAW = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
               "NumberOfTimes90DaysLate"]


class DataNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class CleaningReport:
    source: str
    is_demo: bool
    raw_rows: int
    analyzed_rows: int
    missing_income: int
    missing_dependents: int
    invalid_age_removed: int
    bki_codes_fixed: int
    default_rate_pct: float

    @property
    def imbalance(self) -> float:
        r = self.default_rate_pct / 100
        return (1 - r) / r if r > 0 else float("inf")

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- загрузка
def load_raw(path=C.DATA_PATH, urls=C.DATA_URLS) -> tuple[pd.DataFrame, str]:
    """Локальный файл -> публичные зеркала -> DataNotFoundError."""
    if path.exists():
        return pd.read_csv(path, index_col=0), f"Local file: {path.name}"
    for url in urls:
        try:
            return pd.read_csv(url, index_col=0), f"Download: {url}"
        except Exception:
            continue
    raise DataNotFoundError(
        f"Датасет не найден. Скачайте cs-training.csv с Kaggle (Give Me Some Credit) в {path}."
    )


def generate_demo_data(n: int = 30_000, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Синтетика в формате сырого датасета (с «грязью»: пропуски, age=0, коды 96/98).

    Нужна для запуска без датасета, тестов и CI. В UI помечается как DEMO.
    """
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(50, 14, n), 18, 90).astype(int)
    income = rng.lognormal(8.6, 0.55, n).round()
    util = np.clip(rng.beta(1.1, 3.0, n) * 1.3, 0, 2)
    dti = rng.beta(2, 5, n)
    d30, d60, d90 = rng.poisson(0.25, n), rng.poisson(0.10, n), rng.poisson(0.08, n)
    open_lines, real_estate = rng.poisson(8, n), rng.poisson(1, n)
    deps = rng.poisson(0.8, n).astype(float)

    inc_missing = rng.random(n) < 0.2
    logit = (-4.6 + 2.2 * util + 0.55 * d30 + 0.8 * d60 + 1.0 * d90 + 0.9 * dti
             - 0.025 * (age - 45) - 0.15 * np.log(income / 5000) + 0.3 * inc_missing)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        RAW_TARGET: y, "RevolvingUtilizationOfUnsecuredLines": util.round(4), "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": d30, "DebtRatio": dti.round(4),
        "MonthlyIncome": np.where(inc_missing, np.nan, income),
        "NumberOfOpenCreditLinesAndLoans": open_lines, "NumberOfTimes90DaysLate": d90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": d60,
        "NumberOfDependents": np.where(inc_missing & (rng.random(n) < 0.5), np.nan, deps),
    })
    # воспроизводим реальные дефекты данных
    df.loc[df.sample(max(5, n // 5000), random_state=seed).index, "age"] = 0
    bad = df.sample(max(20, n // 400), random_state=seed + 1).index
    df.loc[bad, _DELINQ_RAW] = 98
    return df


# ------------------------------------------------------------ feature helpers
def weighted_delinq(d30, d60, d90):
    """Единая формула индекса тяжести просрочек (и для обучения, и для скоринга)."""
    w = C.DELINQ_WEIGHTS
    return w["d30"] * d30 + w["d60"] * d60 + w["d90"] * d90


def estimate_loan_amount(monthly_income, dti_ratio):
    """ОЦЕНКА суммы кредита (в датасете её нет) — допущение для финансовой симуляции."""
    income = monthly_income.fillna(monthly_income.median())
    amount = income * 12 * dti_ratio.clip(0.1, 1.2) * C.LOAN_INCOME_MULT
    return amount.clip(C.LOAN_MIN, C.LOAN_MAX).round(-2)


def make_applicant_row(*, age, monthly_income, dependents, income_confirmed, utilization, dti,
                       open_lines, real_estate, d30, d60, d90) -> pd.DataFrame:
    """Строка признаков одного заемщика в точности как при обучении (нет train/serve skew)."""
    income = float(monthly_income) if income_confirmed else np.nan
    return pd.DataFrame([{
        "credit_utilization": utilization, "age": age, "dti_ratio": dti,
        "monthly_income": income,
        "income_per_person": income / (1 + dependents),
        "is_income_missing": int(not income_confirmed),
        "open_credit_lines": open_lines, "real_estate_loans": real_estate,
        "dependents": dependents, "delinq_30_59_days": d30, "delinq_60_89_days": d60,
        "delinq_90_plus_days": d90, "weighted_delinq_score": weighted_delinq(d30, d60, d90),
    }])[C.FEATURES]


# ------------------------------------------------------------------ очистка
def clean_and_engineer(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Очистка сырых данных и feature engineering. Возвращает (df, счетчики для отчета)."""
    df = raw.copy()
    counters = {
        "raw_rows": len(df),
        "missing_income": int(df["MonthlyIncome"].isna().sum()),
        "missing_dependents": int(df["NumberOfDependents"].isna().sum()),
    }

    # 1) ошибки ввода возраста
    bad_age = ~df["age"].between(C.AGE_MIN, C.AGE_MAX)
    counters["invalid_age_removed"] = int(bad_age.sum())
    df = df.loc[~bad_age]

    # 2) технические коды БКИ (96/98) -> NaN (не количество просрочек); сигнал сохраняется как пропуск
    codes = df[_DELINQ_RAW] >= C.BKI_ERROR_CODE
    counters["bki_codes_fixed"] = int(codes.any(axis=1).sum())
    delinq = df[_DELINQ_RAW].mask(codes)

    # 3) доход: пропуск НЕ импутируем (флаг + нативная обработка NaN в GBDT)
    income = df["MonthlyIncome"].clip(lower=500, upper=250_000).round(0)
    deps = df["NumberOfDependents"].fillna(0).clip(0, 10).astype(int)
    dti = df["DebtRatio"].clip(0, 2.5).round(4)

    out = pd.DataFrame({
        "age": df["age"].astype(int),
        "age_group": pd.cut(df["age"], bins=C.AGE_BINS, labels=C.AGE_LABELS,
                            include_lowest=True).astype(str),
        "monthly_income": income,
        "income_per_person": (income / (1 + deps)).round(1),
        "is_income_missing": income.isna().astype(int),
        "dti_ratio": dti,
        "credit_utilization": df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 2).round(4),
        "open_credit_lines": df["NumberOfOpenCreditLinesAndLoans"].clip(0, 40).astype(int),
        "real_estate_loans": df["NumberRealEstateLoansOrLines"].clip(0, 15).astype(int),
        "delinq_30_59_days": delinq["NumberOfTime30-59DaysPastDueNotWorse"],
        "delinq_60_89_days": delinq["NumberOfTime60-89DaysPastDueNotWorse"],
        "delinq_90_plus_days": delinq["NumberOfTimes90DaysLate"],
        "dependents": deps,
        C.TARGET: df[RAW_TARGET].astype(int),
    })
    out["weighted_delinq_score"] = weighted_delinq(
        out["delinq_30_59_days"], out["delinq_60_89_days"], out["delinq_90_plus_days"]).round(1)
    out["loan_amount"] = estimate_loan_amount(out["monthly_income"], out["dti_ratio"])
    out = out.reset_index(drop=True)
    out.insert(0, "client_id", np.arange(100_001, 100_001 + len(out)))
    return out, counters


def stratified_sample(df: pd.DataFrame, n: int, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    if len(df) <= n:
        return df
    s = df.groupby(C.TARGET, group_keys=False).sample(frac=n / len(df), random_state=seed)
    return s.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_dataset(sample_limit: int = C.SAMPLE_LIMIT, allow_demo: bool = True
                 ) -> tuple[pd.DataFrame, CleaningReport]:
    """Полный пайплайн: источник -> очистка -> стратифицированная выборка -> отчет."""
    try:
        raw, source = load_raw()
        is_demo = False
    except DataNotFoundError:
        if not allow_demo:
            raise
        raw, source, is_demo = generate_demo_data(), "DEMO: синтетические данные", True

    clean, counters = clean_and_engineer(raw)
    clean = stratified_sample(clean, sample_limit)
    report = CleaningReport(
        source=source, is_demo=is_demo, analyzed_rows=len(clean),
        default_rate_pct=round(float(clean[C.TARGET].mean() * 100), 2), **counters)
    return clean, report
