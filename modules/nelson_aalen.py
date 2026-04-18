"""
modules/nelson_aalen.py
Estimation du risque cumulé par Nelson-Aalen.

Authors: Bouguessa Nour & Sbartai Sami
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import NelsonAalenFitter
import streamlit as st

from utils.plots import plot_na_curve, apply_defaults, GROUP_COLORS, PALETTE


@st.cache_data(show_spinner=False)
def fit_nelson_aalen(durations, events, label="Nelson-Aalen"):
    """
    Ajuste le modèle de Nelson-Aalen.

    Args:
        durations (pd.Series): Temps.
        events (pd.Series): Indicateur événement.
        label (str): Étiquette.

    Returns:
        NelsonAalenFitter

    Example:
        >>> naf = fit_nelson_aalen(df['Time_to_Event'], df['Event_Observed'])
    """
    if len(durations) < 5:
        raise ValueError("Effectif insuffisant (minimum 5).")
    naf = NelsonAalenFitter()
    naf.fit(durations=durations, event_observed=events, label=label)
    return naf


def plot_na_global(naf):
    """
    Courbe H(t) globale avec IC 95%.

    Args:
        naf: NelsonAalenFitter ajusté.

    Returns:
        go.Figure
    """
    fig = plot_na_curve(naf, label="H(t) — Nelson-Aalen", color=PALETTE["secondary"])
    return apply_defaults(fig, title="Risque cumulé — Nelson-Aalen",
                          xaxis_title="Temps (mois)", yaxis_title="H(t)")


def plot_na_vs_km(naf, kmf):
    """
    Superpose H(t) Nelson-Aalen et -log(S(t)) Kaplan-Meier.

    Args:
        naf: NelsonAalenFitter ajusté.
        kmf: KaplanMeierFitter ajusté.

    Returns:
        go.Figure
    """
    fig = go.Figure()
    t_na = naf.cumulative_hazard_.index.values
    h_na = naf.cumulative_hazard_.iloc[:, 0].values
    fig.add_trace(go.Scatter(x=t_na, y=h_na, mode="lines",
                             name="H(t) Nelson-Aalen",
                             line=dict(color=PALETTE["secondary"], width=2.5)))

    sf = kmf.survival_function_
    t_km = sf.index.values
    s_km = sf.iloc[:, 0].values
    with np.errstate(divide="ignore", invalid="ignore"):
        neg_log_s = -np.log(np.where(s_km > 0, s_km, np.nan))
    fig.add_trace(go.Scatter(x=t_km, y=neg_log_s, mode="lines",
                             name="-log(S(t)) Kaplan-Meier",
                             line=dict(color=PALETTE["primary"], width=2.5, dash="dash")))

    return apply_defaults(fig,
                          title="Nelson-Aalen H(t) vs −log(S(t)) Kaplan-Meier",
                          xaxis_title="Temps (mois)", yaxis_title="Risque cumulé")


def plot_na_stratified(df, time_col, event_col, strat_var):
    """
    Courbes Nelson-Aalen stratifiées.

    Args:
        df (pd.DataFrame): Données.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.
        strat_var (str): Variable de stratification.

    Returns:
        go.Figure
    """
    fig = go.Figure()
    groups = sorted(df[strat_var].dropna().unique(), key=str)
    for i, grp in enumerate(groups):
        sub = df[df[strat_var] == grp]
        if len(sub) < 5:
            continue
        naf = fit_nelson_aalen(sub[time_col], sub[event_col], label=str(grp))
        fig = plot_na_curve(naf, label=str(grp),
                            color=GROUP_COLORS[i % len(GROUP_COLORS)], fig=fig)
    return apply_defaults(fig, title=f"Risque cumulé Nelson-Aalen par {strat_var}",
                          xaxis_title="Temps (mois)", yaxis_title="H(t)")
