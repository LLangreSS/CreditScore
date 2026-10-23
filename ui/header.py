import streamlit as st

from creditrisk import business
from creditrisk.business import Scenario
from creditrisk.data import CleaningReport
from creditrisk.model import TrainResult


def render_header(report: CleaningReport, result: TrainResult, sc: Scenario) -> None:
    """Заголовок, предупреждение о DEMO и строка KPI."""
    m = result.metrics
    st.title("Кредитный скоринг: от сырых данных до экономики портфеля")

    s = business.portfolio_summary(result.test_df, sc.cutoff, sc.lgd, sc.margin)
    kpis = [
        ("ROC-AUC", f"{m['auc']:.3f}", f"95% CI {m['auc_ci'][0]:.3f}–{m['auc_ci'][1]:.3f}"),
        ("Gini / KS", f"{m['gini']:.3f} / {m['ks']*100:.1f}%", f"KS при PD={m['ks_threshold']:.2f}"),
        ("Approval rate", f"{s['approval_rate']*100:.1f}%", f"PD < {sc.cutoff:.0%}"),
        ("NPL одобренных", f"{s['npl_rate']*100:.2f}%", f"Без модели: {m['observed_rate']*100:.2f}%"),
        ("Чистая прибыль (тест)", f"${s['net_profit']/1e6:.2f}M", "маржа − LGD"),
    ]
    for col, (label, value, hint) in zip(st.columns(len(kpis)), kpis):
        col.metric(label, value, hint, delta_color="off")
