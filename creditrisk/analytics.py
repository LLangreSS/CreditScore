"""EDA-аналитика: WoE / Information Value, проверка гипотез, SQL-когорты."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


# ------------------------------------------------------------------ WoE / IV
def woe_iv(df: pd.DataFrame, feature: str, target: str = C.TARGET, bins: int = 8
           ) -> tuple[pd.DataFrame, float]:
    """WoE и IV по квантильным бакетам (с поправкой 0.5 против нулевых ячеек).

    Пропуски выделяются в отдельный бакет 'Missing' — для кредитного риска это сигнал.
    Бакеты упорядочены по значению признака (а не по строке: '10' < '2').
    """
    s = df[feature]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
        _, edges = pd.qcut(s, q=bins, duplicates="drop", retbins=True)
        codes = pd.cut(s, bins=edges, include_lowest=True, labels=False)
        labels = {i: f"[{edges[i]:.3g}; {edges[i + 1]:.3g}]" for i in range(len(edges) - 1)}
        key, order = codes.map(labels), list(labels.values())
    else:
        uniq = sorted(s.dropna().unique())
        labels = {v: f"{v:g}" if not isinstance(v, str) else v for v in uniq}
        key, order = s.map(labels), list(labels.values())
    key = key.fillna("Missing").astype(str)

    g = (pd.DataFrame({"bin": key, "y": df[target]}).groupby("bin")["y"]
         .agg(total="count", bads="sum"))
    g = g.reindex([o for o in order + ["Missing"] if o in g.index])
    g["goods"] = g["total"] - g["bads"]

    k = len(g)
    dist_good = (g["goods"] + 0.5) / (g["goods"].sum() + 0.5 * k)
    dist_bad = (g["bads"] + 0.5) / (g["bads"].sum() + 0.5 * k)
    g["bad_rate"] = (g["bads"] / g["total"] * 100).round(2)
    g["woe"] = np.log(dist_good / dist_bad).round(4)
    g["iv_component"] = ((dist_good - dist_bad) * np.log(dist_good / dist_bad)).round(4)
    return g.reset_index(), round(float(g["iv_component"].sum()), 4)


def iv_strength(iv: float) -> str:
    if iv >= 0.5:
        return "Подозрительно сильная"
    if iv >= 0.3:
        return "Сильная"
    if iv >= 0.1:
        return "Средняя"
    if iv >= 0.02:
        return "Слабая"
    return "Шум (<0.02)"


# ------------------------------------------------------------------ гипотезы
def run_hypothesis_tests(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Mann-Whitney U + размер эффекта (AUC одного признака) + IV.

    На выборках в десятки тысяч p-value почти всегда ≈ 0, поэтому ранжируем по IV и
    однофакторному AUC — они показывают практическую, а не только статистическую значимость.
    """
    columns = columns or ["credit_utilization", "weighted_delinq_score", "delinq_90_plus_days",
                          "delinq_30_59_days", "delinq_60_89_days", "age", "monthly_income",
                          "income_per_person", "dti_ratio", "open_credit_lines"]
    good, bad = df[df[C.TARGET] == 0], df[df[C.TARGET] == 1]
    rows = []
    for col in columns:
        g, b = good[col].dropna(), bad[col].dropna()
        u, p = stats.mannwhitneyu(g, b, alternative="two-sided")
        auc = u / (len(g) * len(b))                       # P(good > bad)
        _, iv = woe_iv(df, col)
        rows.append({
            "feature": col, "median_good": round(g.median(), 3), "median_bad": round(b.median(), 3),
            "p_value": p, "univariate_auc": round(max(auc, 1 - auc), 3),
            "iv": iv, "strength": iv_strength(iv),
        })
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ SQL
_COHORT_SQL = """
WITH cohorts AS (
    SELECT
        age_group,
        CASE
            WHEN is_income_missing = 1    THEN 'No income proof'
            WHEN monthly_income < 4000    THEN 'Low (<$4k)'
            WHEN monthly_income <= 8500   THEN 'Middle ($4k-$8.5k)'
            ELSE 'High (>$8.5k)'
        END                                AS income_segment,
        COUNT(*)                           AS clients,
        SUM("default")                     AS defaults,
        ROUND(AVG(credit_utilization) * 100, 1) AS avg_util_pct,
        ROUND(AVG(dti_ratio) * 100, 1)     AS avg_dti_pct
    FROM loans
    GROUP BY 1, 2
    HAVING COUNT(*) >= :min_clients
)
SELECT
    age_group, income_segment, clients, avg_util_pct, avg_dti_pct,
    ROUND(100.0 * defaults / clients, 2) AS default_rate_pct,
    -- отклонение от СРЕДНЕВЗВЕШЕННОГО default rate возрастной группы
    ROUND(100.0 * defaults / clients
          - 100.0 * SUM(defaults) OVER w / SUM(clients) OVER w, 2) AS delta_vs_age_pp
FROM cohorts
WINDOW w AS (PARTITION BY age_group)
ORDER BY age_group, default_rate_pct DESC;
"""


def sql_cohort_analysis(df: pd.DataFrame, min_clients: int = 50) -> pd.DataFrame:
    """Когорты возраст × доход через SQL с оконными функциями."""
    cols = ["age_group", "is_income_missing", "monthly_income", "credit_utilization",
            "dti_ratio", C.TARGET]
    with sqlite3.connect(":memory:") as conn:
        df[cols].to_sql("loans", conn, index=False)
        return pd.read_sql_query(_COHORT_SQL, conn, params={"min_clients": min_clients})
