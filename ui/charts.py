"""Чистые функции: данные -> plotly Figure"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from creditrisk import analytics, config as C


def woe_profile(woe: pd.DataFrame, feature: str, iv: float) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=woe["bin"], y=woe["total"], name="Заемщиков",
                marker_color="rgba(70,130,180,.45)", secondary_y=False)
    fig.add_scatter(x=woe["bin"], y=woe["bad_rate"], name="Default rate, %", mode="lines+markers",
                    line=dict(color="#EF553B", width=3), secondary_y=True)
    fig.update_layout(title=f"{feature}: профиль риска (IV = {iv}, {analytics.iv_strength(iv)})",
                      height=410, legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    fig.update_yaxes(title_text="Заемщиков", secondary_y=False)
    fig.update_yaxes(title_text="Default rate, %", secondary_y=True)
    return fig


def ks_plot(curve: pd.DataFrame, ks: float, ks_threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=curve["pd_score"], y=curve["cum_bad"], name="Дефолты",
                    line=dict(color="#EF553B", width=2.5))
    fig.add_scatter(x=curve["pd_score"], y=curve["cum_good"], name="Хорошие",
                    line=dict(color="#00CC96", width=2.5))
    fig.add_vline(x=ks_threshold, line_dash="dash", line_color="gold",
                  annotation_text=f"KS = {ks*100:.1f}%")
    fig.update_layout(title="Разделение bads / goods", xaxis_title="PD",
                      yaxis_title="Кумулятивная доля", height=400)
    return fig


def importance_bar(importances: pd.Series) -> go.Figure:
    fig = px.bar(x=importances.values, y=importances.index, orientation="h",
                 title="Permutation importance", labels={"x": "Δ AUC", "y": ""})
    fig.update_layout(height=400)
    return fig


def calibration_plot(cal: pd.DataFrame) -> go.Figure:
    top = float(cal[["pred", "obs"]].to_numpy().max())
    fig = go.Figure()
    fig.add_scatter(x=[0, top], y=[0, top], name="Идеальная калибровка",
                    line=dict(dash="dash", color="gray"))
    fig.add_scatter(x=cal["pred"], y=cal["obs"], name="Модель", mode="lines+markers")
    fig.update_layout(title="Предсказанный PD vs фактический default rate",
                      xaxis_title="Средний предсказанный PD",
                      yaxis_title="Фактический default rate", height=380)
    return fig


def profit_curve_plot(curve: pd.DataFrame, best: pd.Series, current_cutoff: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=curve["cutoff"], y=curve["net_profit"] / 1e6, name="Чистая прибыль, $M",
                    line=dict(color="#636EFA", width=3))
    fig.add_vline(x=best["cutoff"], line_dash="dash", line_color="#00CC96",
                  annotation_text=f"Оптимум PD={best['cutoff']:.2f} (${best['net_profit']/1e6:.2f}M)")
    fig.add_vline(x=current_cutoff, line_dash="dot", line_color="#EF553B",
                  annotation_text=f"Текущий: {current_cutoff:.2f}", annotation_position="bottom right")
    fig.update_layout(title="Прибыль портфеля vs cut-off", xaxis_title="Cut-off PD",
                      yaxis_title="$M", height=420)
    return fig


def score_histogram(test_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        test_df, x="score_points", color=test_df[C.TARGET].map({0: "Не дефолт", 1: "Дефолт"}),
        nbins=45, barmode="overlay", opacity=0.75, histnorm="percent",
        color_discrete_map={"Не дефолт": "#00CC96", "Дефолт": "#EF553B"},
        title=(f"Скоринговый балл"),
        labels={"score_points": "Балл", "color": "Статус"})
    fig.update_layout(height=420, yaxis_title="Доля внутри класса, %")
    return fig
