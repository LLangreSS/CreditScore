import numpy as np
import pandas as pd
import streamlit as st

from creditrisk import business
from creditrisk.business import Scenario
from ui import charts


def render(test_df: pd.DataFrame, sc: Scenario) -> None:
    curve = business.profit_curve(test_df, np.linspace(0.02, 0.50, 60), sc.lgd, sc.margin)
    best = curve.loc[curve["net_profit"].idxmax()]

    left, right = st.columns(2)
    left.plotly_chart(charts.profit_curve_plot(curve, best, sc.cutoff), width="stretch")
    right.plotly_chart(charts.score_histogram(test_df), width="stretch")
