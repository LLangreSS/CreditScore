import pandas as pd
import streamlit as st

from creditrisk import model
from creditrisk.model import TrainResult
from ui import charts


def render(result: TrainResult) -> None:
    m = result.metrics
    left, right = st.columns(2)
    left.plotly_chart(charts.ks_plot(model.ks_curve(result.test_df), m["ks"], m["ks_threshold"]),
                      width="stretch")
    right.plotly_chart(charts.importance_bar(result.importances), width="stretch")

    left, right = st.columns(2)
    left.plotly_chart(charts.calibration_plot(model.calibration_table(result.test_df)),
                      width="stretch")
    with right:
        st.markdown("**Метрики на тесте**")
        st.dataframe(pd.DataFrame({
            "Метрика": ["ROC-AUC", "Gini", "KS", "Brier score", "Brier",
                        "Средний PD", "Фактический default rate"],
            "Значение": [f"{m['auc']:.4f}", f"{m['gini']:.4f}", f"{m['ks']:.4f}", f"{m['brier']:.5f}",
                         f"{m['brier_baseline']:.5f}", f"{m['mean_pd']:.4f}", f"{m['observed_rate']:.4f}"],
        }), width="stretch", hide_index=True)
