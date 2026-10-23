import numpy as np
import pandas as pd

from creditrisk import analytics
from tests.helpers import demo_clean


def test_woe_iv_iv_is_sum_and_nonnegative():
    w, iv = analytics.woe_iv(demo_clean(), "credit_utilization")
    assert abs(iv - w["iv_component"].sum()) < 1e-3
    assert (w["iv_component"] >= -1e-9).all()


def test_woe_utilization_is_monotone_in_risk():
    w, _ = analytics.woe_iv(demo_clean(), "credit_utilization", bins=5)
    assert w["bad_rate"].iloc[-1] > w["bad_rate"].iloc[0]


def test_discrete_bins_sorted_numerically_and_missing_bucket():
    df = demo_clean()
    w, _ = analytics.woe_iv(df, "delinq_90_plus_days")
    nums = [float(b) for b in w["bin"] if b != "Missing"]
    assert nums == sorted(nums)
    assert w["bin"].iloc[-1] == "Missing"


def test_hypothesis_table_columns_and_sorting():
    t = analytics.run_hypothesis_tests(demo_clean())
    assert t["iv"].is_monotonic_decreasing
    assert t["univariate_auc"].between(0.5, 1).all()


def test_sql_cohorts_run_and_deltas_weighted_to_zero():
    c = analytics.sql_cohort_analysis(demo_clean(), min_clients=20)
    assert not c.empty
    for _, g in c.groupby("age_group"):
        # взвешенная сумма отклонений внутри возрастной группы = 0 (если все когорты прошли HAVING)
        assert abs((g["clients"] * g["delta_vs_age_pp"]).sum()) / g["clients"].sum() < 0.5
