import streamlit as st

from creditrisk import business, data, model
from creditrisk.business import Scenario
from creditrisk.model import ScoringModel


def _inputs() -> tuple[dict, float]:
    f1, f2, f3 = st.columns(3)
    with f1:
        age = st.number_input("Возраст", 18, 90, 36)
        income = st.number_input("Ежемесячный доход ($)", 500, 100_000, 5200, step=250)
        deps = st.number_input("Иждивенцы", 0, 10, 1)
        confirmed = st.checkbox("Доход подтвержден справкой", value=True)
    with f2:
        util = st.slider("Утилизация кредитных карт", 0.0, 1.5, 0.32, 0.02)
        dti = st.slider("Долговая нагрузка (DebtRatio)", 0.0, 1.5, 0.35, 0.02)
        open_lines = st.number_input("Открытых кредитных линий", 0, 35, 6)
        real_estate = st.number_input("Ипотечных кредитов", 0, 10, 1)
    with f3:
        d30 = st.number_input("Просрочки 30–59 дней", 0, 15, 0)
        d60 = st.number_input("Просрочки 60–89 дней", 0, 15, 0)
        d90 = st.number_input("Просрочки 90+ дней", 0, 15, 0)
        loan = st.number_input("Запрашиваемая сумма($)", 1500, 150_000, 18_000, step=1000)
    features = dict(age=age, monthly_income=income, dependents=deps, income_confirmed=confirmed,
                    utilization=util, dti=dti, open_lines=open_lines, real_estate=real_estate,
                    d30=d30, d60=d60, d90=d90)
    return features, loan


def render(scoring: ScoringModel, sc: Scenario) -> None:
    st.subheader("Андеррайтинг одного заемщика")
    features, loan = _inputs()

    pd_value = float(scoring.predict_pd(data.make_applicant_row(**features))[0])
    d = business.underwrite(pd_value, loan, sc.lgd, sc.margin, sc.cutoff)

    st.divider()
    k = st.columns(4)
    k[0].metric("PD", f"{pd_value*100:.2f}%")
    k[1].metric("Скоринговый балл", f"{int(model.pd_to_score(pd_value))}")
    k[2].metric("Ожидаемые потери", f"${d.expected_loss:,.0f}")
    k[3].metric("Ожидаемая прибыль", f"${d.expected_profit:,.0f}")
    verdict = f"PD {pd_value:.1%} {'<' if d.approved else '≥'} cut-off {sc.cutoff:.1%}. EV сделки: ${d.expected_profit:,.0f}"
    (st.success if d.approved else st.error)(f"{'**ОДОБРЕНО**' if d.approved else '**ОТКАЗ**'}: {verdict}")
