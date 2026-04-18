"""
modules/cox_model.py
Modèle de Cox : ajustement, résultats, vérification PH, résidus.

Authors: Bouguessa Nour & Sbartai Sami
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import CoxPHFitter
import streamlit as st

from utils.plots import plot_forest, apply_defaults, PALETTE, GROUP_COLORS
from utils.stats_helpers import martingale_residuals, deviance_residuals

CATEGORICAL_COLS = ["Sex", "Treatment", "Physical_Activity"]


def prepare_cox_data(df, time_col, event_col, covariates):
    """
    Prépare le DataFrame pour le modèle de Cox.

    Args:
        df (pd.DataFrame): Données source.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.
        covariates (list): Covariables à inclure.

    Returns:
        pd.DataFrame: Données encodées.
    """
    cols = [time_col, event_col] + [c for c in covariates if c in df.columns]
    df_cox = df[cols].dropna().copy()
    cat_to_encode = [c for c in covariates
                     if c in CATEGORICAL_COLS and c in df_cox.columns
                     and df_cox[c].dtype == "object"]
    if cat_to_encode:
        df_cox = pd.get_dummies(df_cox, columns=cat_to_encode, drop_first=True)
    for col in df_cox.select_dtypes(include="bool").columns:
        df_cox[col] = df_cox[col].astype(int)
    return df_cox


@st.cache_resource(show_spinner=False)
def fit_cox_model(df_hash, df_cox, time_col, event_col, ties="breslow"):
    """
    Ajuste le modèle de Cox proportionnel.

    Args:
        df_hash (str): Clé de cache.
        df_cox (pd.DataFrame): Données préparées.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.
        ties (str): Méthode ex-aequo ('breslow', 'efron').

    Returns:
        CoxPHFitter

    Raises:
        ValueError: Si effectif insuffisant.

    Example:
        >>> cph = fit_cox_model(hash_str, df_cox, 'Time_to_Event', 'Event_Observed')
    """
    if len(df_cox) < 10:
        raise ValueError("Effectif insuffisant pour le modèle de Cox.")
    cph = CoxPHFitter(baseline_estimation_method=ties)
    cph.fit(df_cox, duration_col=time_col, event_col=event_col)
    return cph


def get_cox_summary(cph):
    """
    Tableau des résultats du modèle de Cox.

    Args:
        cph: CoxPHFitter ajusté.

    Returns:
        pd.DataFrame
    """
    s = cph.summary.copy().reset_index()
    s.columns = [c.strip() for c in s.columns]
    rename_map = {
        "covariate": "Variable", "coef": "β", "se(coef)": "SE",
        "z": "Wald", "exp(coef)": "HR",
        "exp(coef) lower 95%": "IC inf 95%",
        "exp(coef) upper 95%": "IC sup 95%", "p": "p-valeur",
    }
    s = s.rename(columns=rename_map)
    keep = [c for c in ["Variable", "β", "SE", "Wald", "HR",
                         "IC inf 95%", "IC sup 95%", "p-valeur"] if c in s.columns]
    s = s[keep]
    for col in ["β", "SE", "Wald", "HR", "IC inf 95%", "IC sup 95%"]:
        if col in s.columns:
            s[col] = s[col].round(4)
    if "p-valeur" in s.columns:
        s["p-valeur"] = s["p-valeur"].apply(
            lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001")
    return s


def get_forest_data(cph):
    """
    Données pour le forest plot.

    Args:
        cph: CoxPHFitter ajusté.

    Returns:
        pd.DataFrame
    """
    s = cph.summary
    return pd.DataFrame({
        "variable": s.index,
        "HR": s["exp(coef)"].values,
        "lower_95": s["exp(coef) lower 95%"].values,
        "upper_95": s["exp(coef) upper 95%"].values,
        "p": s["p"].values,
    })


def check_proportional_hazards(cph, df_cox):
    """
    Vérification de l'hypothèse des risques proportionnels.

    Args:
        cph: CoxPHFitter ajusté.
        df_cox (pd.DataFrame): Données d'entraînement.

    Returns:
        dict: table avec p-values.
    """
    try:
        cph.check_assumptions(df_cox, p_value_threshold=0.05, show_plots=False)
        ph_table = cph.summary[["p"]].copy()
        ph_table.columns = ["p-valeur Schoenfeld"]
        ph_table["Hypothèse PH"] = ph_table["p-valeur Schoenfeld"].apply(
            lambda p: "✅ Respectée" if p > 0.05 else "⚠️ Violation potentielle")
        return {"table": ph_table.reset_index()}
    except Exception as e:
        return {"table": pd.DataFrame({"Erreur": [str(e)]})}


def plot_schoenfeld_residuals(cph, df_cox, variable):
    """
    Résidus de Schoenfeld vs temps pour une variable.

    Args:
        cph: CoxPHFitter ajusté.
        df_cox (pd.DataFrame): Données.
        variable (str): Variable à tracer.

    Returns:
        go.Figure
    """
    try:
        resid = cph.compute_residuals(df_cox, kind="schoenfeld")
        if variable not in resid.columns:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=resid.index, y=resid[variable], mode="markers",
            marker=dict(color=PALETTE["primary"], size=5, opacity=0.6),
            name="Résidu Schoenfeld",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["neutral"])
        return apply_defaults(fig, title=f"Résidus Schoenfeld — {variable}",
                              xaxis_title="Temps", yaxis_title="Résidu")
    except Exception:
        return go.Figure()


def plot_martingale(cph, df_cox, variable):
    """
    Résidus de Martingale vs une variable numérique.

    Args:
        cph: CoxPHFitter ajusté.
        df_cox (pd.DataFrame): Données.
        variable (str): Variable numérique.

    Returns:
        go.Figure
    """
    try:
        mart = cph.compute_residuals(df_cox, kind="martingale")["martingale"]
        if variable not in df_cox.columns:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_cox[variable].values, y=mart, mode="markers",
            marker=dict(color=PALETTE["primary"], size=5, opacity=0.5),
            name="Résidu Martingale",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["neutral"])
        return apply_defaults(fig, title=f"Résidus Martingale vs {variable}",
                              xaxis_title=variable, yaxis_title="Résidu")
    except Exception:
        return go.Figure()


def plot_partial_effects(cph, df_cox, variable):
    """
    Courbes de survie ajustées par effet partiel.

    Args:
        cph: CoxPHFitter ajusté.
        df_cox (pd.DataFrame): Données.
        variable (str): Variable d'intérêt.

    Returns:
        go.Figure
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vals = sorted(df_cox[variable].unique())
        fig_mpl = cph.plot_partial_effects_on_outcome(
            covariates=variable, values=vals, plot_baseline=False)
        fig = go.Figure()
        ax = fig_mpl.axes[0]
        for i, line in enumerate(ax.lines):
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            if len(x_data) > 1:
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_data, mode="lines",
                    name=str(line.get_label()),
                    line=dict(width=2.5, color=GROUP_COLORS[i % len(GROUP_COLORS)]),
                ))
        plt.close("all")
        fig = apply_defaults(fig, title=f"Survie ajustée — Effet de {variable}",
                             xaxis_title="Temps (mois)", yaxis_title="S(t)")
        fig.update_layout(yaxis=dict(range=[0, 1.05]))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erreur : {e}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig


def get_cox_metrics(cph):
    """
    Métriques globales du modèle de Cox.

    Args:
        cph: CoxPHFitter ajusté.

    Returns:
        dict: concordance, AIC, log_likelihood.
    """
    return {
        "concordance": round(cph.concordance_index_, 4),
        "AIC": round(cph.AIC_partial_, 2),
        "log_likelihood": round(cph.log_likelihood_, 2),
    }
