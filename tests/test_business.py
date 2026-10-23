import numpy as np

from creditrisk import business
from tests.helpers import trained


def test_profit_curve_matches_direct_calculation():
    t = trained().test_df
    grid = np.linspace(0.02, 0.5, 15)
    curve = business.profit_curve(t, grid, 0.7, 0.16)
    direct = [business.portfolio_summary(t, c, 0.7, 0.16)["net_profit"] for c in grid]
    assert np.allclose(curve["net_profit"], direct)


def test_zero_cutoff_means_no_loans():
    t = trained().test_df
    s = business.portfolio_summary(t, 0.0, 0.7, 0.16)
    assert s["approval_rate"] == 0 and s["net_profit"] == 0


def test_underwrite_arithmetic():
    d = business.underwrite(0.10, 10_000, lgd=0.5, margin=0.2, cutoff=0.14)
    assert d.expected_loss == 500
    assert abs(d.expected_profit - (10_000 * 0.9 * 0.2 - 500)) < 1e-9
    assert d.approved
    assert not business.underwrite(0.2, 10_000, 0.5, 0.2, 0.14).approved
