import numpy as np
from sklearn.metrics import roc_auc_score

from creditrisk import config as C
from creditrisk import model
from tests.helpers import trained


def test_score_scale_anchor_and_monotonicity():
    s = model.pd_to_score([C.SCORE_BASE_PD, C.SCORE_BASE_PD / 2, C.SCORE_BASE_PD * 2])
    assert s[0] == C.SCORE_BASE
    assert s[1] > s[0] > s[2]
    # PDO: удвоение odds = -PDO баллов
    p1 = 0.05
    odds2 = 2 * p1 / (1 - p1)
    p2 = odds2 / (1 + odds2)
    assert abs(int(model.pd_to_score(p1)) - int(model.pd_to_score(p2)) - C.SCORE_PDO) <= 1


def test_calibrator_preserves_ranking():
    rng = np.random.default_rng(0)
    p = rng.random(2000) ** 3
    y = (rng.random(2000) < p).astype(int)
    cal = model.SigmoidCalibrator().fit(p, y)
    assert abs(roc_auc_score(y, p) - roc_auc_score(y, cal.predict(p))) < 1e-9


def test_metrics_sane_and_calibrated():
    m = trained().metrics
    assert m["auc"] > 0.6 and 0 < m["ks"] < 1
    assert m["auc_ci"][0] < m["auc"] < m["auc_ci"][1]
    assert abs(m["mean_pd"] - m["observed_rate"]) < 0.01
    assert m["brier"] <= m["brier_baseline"] * 1.02


def test_split_sizes_and_test_frame_columns():
    r = trained()
    m = r.metrics
    n = m["n_train"] + m["n_valid"] + m["n_test"]
    assert abs(m["n_test"] / n - 0.2) < 0.01
    assert {"pd_score", "score_points"} <= set(r.test_df.columns)
    assert r.test_df["pd_score"].between(0.001, 0.999).all()


def test_missing_values_are_scoreable():
    r = trained()
    X = r.test_df[C.FEATURES].head(50).copy()
    X.iloc[:, 3] = np.nan
    assert not np.isnan(r.model.predict_pd(X)).any()


def test_ks_curve_matches_ks_metric():
    r = trained()
    c = model.ks_curve(r.test_df)
    assert abs((c["cum_good"] - c["cum_bad"]).max() - r.metrics["ks"]) < 0.01
    assert abs(c["cum_bad"].iloc[-1] - 1) < 1e-9 and abs(c["cum_good"].iloc[-1] - 1) < 1e-9


def test_calibration_table_shape_and_range():
    cal = model.calibration_table(trained().test_df)
    assert list(cal.columns) == ["pred", "obs"] and 5 <= len(cal) <= 10
    assert cal["pred"].is_monotonic_increasing
