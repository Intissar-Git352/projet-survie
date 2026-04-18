"""
modules/kaplan_meier.py
Analyse de Kaplan-Meier : estimation, courbes, tests.

Authors: Bouguessa Nour & Sbartai Sami
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import streamlit as st
from itertools import combinations

from utils.plots import plot_km_curve, apply_defaults, GROUP_COLORS, PALETTE
from utils.stats_helpers import bonferroni_correction


@st.cache_data(show_spinner=False)
def fit_kaplan_meier(durations, events, label="Survie globale"):
    """
    Ajuste un modèle de Kaplan-Meier.

    Args:
        durations (pd.Series): Temps jusqu'à l'événement.
        events (pd.Series): Indicateur (1=événement, 0=censure).
        label (str): Étiquette de la courbe.

    Returns:
        KaplanMeierFitter

    Raises:
        ValueError: Si données insuffisantes ou négatives.

    Example:
        >>> kmf = fit_kaplan_meier(df['Time_to_Event'], df['Event_Observed'])
    """
    if len(durations) == 0:
        raise ValueError("Aucune observation disponible.")
    if (durations < 0).any():
        raise ValueError("Les durées ne peuvent pas être négatives.")
    if len(durations) < 5:
        raise ValueError("Effectif insuffisant (minimum 5).")
    kmf = KaplanMeierFitter()
    kmf.fit(durations=durations, event_observed=events, label=label)
    return kmf


def plot_km_global(kmf):
    """
    Courbe KM globale avec IC 95% et médiane.

    Args:
        kmf: KaplanMeierFitter ajusté.

    Returns:
        go.Figure
    """
    fig = plot_km_curve(kmf, label="Survie globale", color=PALETTE["primary"])
    median = kmf.median_survival_time_
    if np.isfinite(float(median)):
        fig.add_hline(y=0.5, line_dash="dot",
                      line_color=PALETTE["neutral"], line_width=1)
        fig.add_vline(x=float(median), line_dash="dot",
                      line_color=PALETTE["neutral"], line_width=1,
                      annotation_text=f"Médiane : {median:.1f} mois",
                      annotation_position="top right")
    fig = apply_defaults(fig,
                         title="Courbe de survie Kaplan-Meier — Cohorte globale",
                         xaxis_title="Temps (mois)",
                         yaxis_title="Probabilité de survie S(t)")
    fig.update_layout(yaxis=dict(range=[0, 1.05]))
    return fig


def plot_km_stratified(df, time_col, event_col, strat_var):
    """
    Courbes KM stratifiées par groupe.

    Args:
        df (pd.DataFrame): Données.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.
        strat_var (str): Variable de stratification.

    Returns:
        tuple: (go.Figure, dict {groupe: KaplanMeierFitter})
    """
    fig = go.Figure()
    kmf_dict = {}
    groups = sorted(df[strat_var].dropna().unique(), key=str)

    for i, grp in enumerate(groups):
        sub = df[df[strat_var] == grp]
        if len(sub) < 5:
            continue
        kmf = fit_kaplan_meier(sub[time_col], sub[event_col], label=str(grp))
        fig = plot_km_curve(kmf, label=str(grp),
                            color=GROUP_COLORS[i % len(GROUP_COLORS)], fig=fig)
        kmf_dict[grp] = kmf

    fig = apply_defaults(fig,
                         title=f"Survie KM stratifiée par {strat_var}",
                         xaxis_title="Temps (mois)",
                         yaxis_title="Probabilité de survie S(t)")
    fig.update_layout(yaxis=dict(range=[0, 1.05]))
    return fig, kmf_dict


def get_survival_table(kmf):
    """
    Tableau de survie exportable.

    Args:
        kmf: KaplanMeierFitter ajusté.

    Returns:
        pd.DataFrame
    """
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_survival_function_
    et = kmf.event_table
    return pd.DataFrame({
        "Temps (mois)": sf.index.round(2),
        "S(t)": sf.iloc[:, 0].round(4),
        "IC inf 95%": ci.iloc[:, 0].round(4),
        "IC sup 95%": ci.iloc[:, 1].round(4),
        "N à risque": et["at_risk"],
        "Événements": et["observed"],
        "Censures": et["censored"],
    }).reset_index(drop=True)


def get_medians_table(kmf_dict):
    """
    Tableau des médianes de survie par groupe.

    Args:
        kmf_dict (dict): {groupe: KaplanMeierFitter}.

    Returns:
        pd.DataFrame
    """
    rows = []
    for grp, kmf in kmf_dict.items():
        med = kmf.median_survival_time_
        rows.append({
            "Groupe": grp,
            "Médiane (mois)": round(float(med), 2) if np.isfinite(float(med)) else "Non atteinte",
            "N": int(kmf.event_table["at_risk"].iloc[0]),
            "Événements": int(kmf.event_table["observed"].sum()),
        })
    return pd.DataFrame(rows)


def run_logrank_tests(df, time_col, event_col, strat_var):
    """
    Tests log-rank global et pairwise avec correction Bonferroni.

    Args:
        df (pd.DataFrame): Données.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.
        strat_var (str): Variable de stratification.

    Returns:
        dict: global et pairwise.
    """
    groups = sorted(df[strat_var].dropna().unique(), key=str)
    result = {}

    try:
        res = multivariate_logrank_test(df[time_col], df[strat_var], df[event_col])
        result["global"] = {
            "chi2": round(float(res.test_statistic), 4),
            "p_value": round(float(res.p_value), 4),
            "ddl": len(groups) - 1,
            "significant": res.p_value < 0.05,
        }
    except Exception:
        result["global"] = None

    pairs, raw_p = [], []
    for g1, g2 in combinations(groups, 2):
        m1, m2 = df[strat_var] == g1, df[strat_var] == g2
        try:
            res2 = logrank_test(df[m1][time_col], df[m2][time_col],
                                df[m1][event_col], df[m2][event_col])
            pairs.append((str(g1), str(g2),
                          round(float(res2.test_statistic), 4),
                          round(float(res2.p_value), 4)))
            raw_p.append(float(res2.p_value))
        except Exception:
            pass

    if pairs:
        corr_p = bonferroni_correction(raw_p)
        result["pairwise"] = pd.DataFrame([
            {"Groupe 1": g1, "Groupe 2": g2, "χ²": chi2,
             "p brute": round(p, 4), "p Bonferroni": round(pc, 4),
             "Significatif": "✅" if pc < 0.05 else "❌"}
            for (g1, g2, chi2, p), pc in zip(pairs, corr_p)
        ])
    else:
        result["pairwise"] = pd.DataFrame()

    return result
