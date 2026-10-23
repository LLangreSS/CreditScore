import numpy as np
import pandas as pd

from creditrisk import config as C
from creditrisk import data
from tests.helpers import demo_clean


def _raw(**over):
    base = data.generate_demo_data(n=200)
    for k, v in over.items():
        base.loc[base.index[0], k] = v
    return base


def test_invalid_ages_removed_and_age_18_kept():
    raw = _raw(age=0)
    raw.loc[raw.index[1], "age"] = 18
    clean, cnt = data.clean_and_engineer(raw)
    assert cnt["invalid_age_removed"] >= 1
    assert clean["age"].between(18, 95).all()
    assert "nan" not in set(clean["age_group"])          # регрессия: age=18 не должен терять группу


def test_bki_codes_become_nan_not_counts():
    clean, cnt = data.clean_and_engineer(data.generate_demo_data(n=5_000))
    assert cnt["bki_codes_fixed"] > 0
    assert clean["delinq_90_plus_days"].max() < 90
    assert clean["delinq_90_plus_days"].isna().sum() == cnt["bki_codes_fixed"]


def test_income_not_imputed_but_flagged():
    clean = demo_clean()
    assert (clean["is_income_missing"] == clean["monthly_income"].isna().astype(int)).all()
    assert clean["is_income_missing"].sum() > 0


def test_loan_amount_within_bounds():
    la = demo_clean()["loan_amount"]
    assert la.between(C.LOAN_MIN, C.LOAN_MAX).all()


def test_stratified_sample_keeps_default_rate():
    df = demo_clean()
    s = data.stratified_sample(df, 4_000)
    assert abs(len(s) - 4_000) <= 5
    assert abs(s[C.TARGET].mean() - df[C.TARGET].mean()) < 0.003


def test_applicant_row_matches_training_features():
    row = data.make_applicant_row(age=40, monthly_income=5000, dependents=1, income_confirmed=True,
                                  utilization=.3, dti=.3, open_lines=5, real_estate=1,
                                  d30=1, d60=2, d90=1)
    assert list(row.columns) == C.FEATURES
    assert row.loc[0, "weighted_delinq_score"] == 1 * 1.0 + 2 * 2.0 + 1 * 3.5   # d60 учитывается
    no_inc = data.make_applicant_row(age=40, monthly_income=5000, dependents=1,
                                     income_confirmed=False, utilization=.3, dti=.3, open_lines=5,
                                     real_estate=1, d30=0, d60=0, d90=0)
    assert np.isnan(no_inc.loc[0, "monthly_income"]) and no_inc.loc[0, "is_income_missing"] == 1
