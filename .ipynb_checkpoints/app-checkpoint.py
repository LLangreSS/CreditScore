"""Точка входа Streamlit: загрузка артефактов и сборка страницы. Запуск: streamlit run app.py"""
import streamlit as st

from creditrisk import data, model
from ui.header import render_header
from ui.sidebar import render_sidebar
from ui.tabs import calculator, economics, eda, quality

st.set_page_config(page_title="Credit Risk Scoring", page_icon="", layout="wide")


@st.cache_resource(show_spinner="Загрузка данных и обучение модели…")
def get_artifacts():
    df, report = data.load_dataset()
    return df, report, model.train_scoring_model(df)


df, report, result = get_artifacts()
scenario = render_sidebar(report)
render_header(report, result, scenario)

tab_eda, tab_quality, tab_econ, tab_calc = st.tabs(
    ["EDA · WoE/IV · SQL", "Качество модели", "Экономика и скор-шкала", "Калькулятор"])
with tab_eda:
    eda.render(df)
with tab_quality:
    quality.render(result)
with tab_econ:
    economics.render(result.test_df, scenario)
with tab_calc:
    calculator.render(result.model, scenario)
