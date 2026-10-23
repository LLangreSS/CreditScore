import pandas as pd
import streamlit as st

from creditrisk import analytics
from ui import charts

RU_STATS = {"feature": "Признак", "median_good": "Медиана (не дефолт)", "median_bad": "Медиана (дефолт)",
            "p_value": "p-value (Манн-Уитни)", "univariate_auc": "AUC признака", "iv": "IV",
            "strength": "Предсказательная сила"}
RU_COHORTS = {"age_group": "Возраст", "income_segment": "Сегмент дохода", "clients": "Клиентов",
              "avg_util_pct": "Утилизация, %", "avg_dti_pct": "DTI, %",
              "default_rate_pct": "Default rate, %", "delta_vs_age_pp": "Δ к возрастной группе, п.п."}
WOE_FEATURES = ["credit_utilization", "weighted_delinq_score", "age", "monthly_income",
                "dti_ratio", "open_credit_lines", "delinq_90_plus_days"]

def fmt_p(p: float) -> str:
    """p-value в научной записи: 3.0e-35."""
    return "< 1e-300" if p < 1e-300 else f"{p:.1e}" 
    
def render(df: pd.DataFrame) -> None:
    st.subheader("Значимость признаков: Mann-Whitney U, однофакторный AUC, Information Value")
    table = analytics.run_hypothesis_tests(df)
    table["p_value"] = table["p_value"].map(fmt_p)          
    st.dataframe(table.rename(columns=RU_STATS), width="stretch", hide_index=True)

    left, right = st.columns([1.15, 1])
    with left:
        feature = st.selectbox("Признак для анализа монотонности риска", WOE_FEATURES)
        woe, iv = analytics.woe_iv(df, feature)
        st.plotly_chart(charts.woe_profile(woe, feature, iv), width="stretch")
        with st.expander("Таблица WoE"):
            st.dataframe(woe, width="stretch", hide_index=True)
    with right:
        st.markdown("**SQL-когорты: возраст × подтвержденность дохода**")
        st.dataframe(analytics.sql_cohort_analysis(df).rename(columns=RU_COHORTS),
                     width="stretch", hide_index=True, height=380)
