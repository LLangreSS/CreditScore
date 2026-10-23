import streamlit as st

from creditrisk.business import Scenario
from creditrisk.data import CleaningReport


def render_sidebar(report: CleaningReport) -> Scenario:
    """Рисует сайдбар и возвращает выбранный бизнес-сценарий."""
    st.sidebar.title("Параметры банка")
    scenario = Scenario(
        lgd=st.sidebar.slider("Потери при дефолте, %", 30, 100, 70, 5) / 100,
        margin=st.sidebar.slider("Чистая процентная маржа, %", 5, 30, 16, 1) / 100,
        cutoff=st.sidebar.slider("Cut-off по PD", 0.02, 0.45, 0.14, 0.01),
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Data quality report")
    st.sidebar.markdown(
        f"- **Источник:** {report.source}\n"
        f"- **Default rate:** {report.default_rate_pct}% (дисбаланс 1:{report.imbalance:.0f})\n"
        f"- **Пропуски дохода:** {report.missing_income:,} (флаг + NaN, без импутации)\n"
        f"- **Коды БКИ 96/98 → NaN:** {report.bki_codes_fixed:,}\n"
        f"- **Некорректный возраст удален:** {report.invalid_age_removed:,}"
    )
    return scenario
