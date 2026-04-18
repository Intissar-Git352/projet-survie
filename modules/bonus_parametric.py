"""
modules/bonus_parametric.py
Modèles de survie paramétriques : Weibull, Exponentiel, Log-Normal.

Authors: Bouguessa Nour & Sbartai Sami
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import WeibullFitter, ExponentialFitter, LogNormalFitter, KaplanMeierFitter

from utils.plots import apply_defaults, PALETTE, GROUP_COLORS


def fit_parametric_models(durations, events):
    """
    Ajuste les modèles paramétriques Weibull, Exponentiel et Log-Normal.

    Args:
        durations (pd.Series): Temps jusqu'à l'événement.
        events (pd.Series): Indicateur événement.

    Returns:
        dict: {nom_modele: fitter ajusté}
    """
    models = {
        "Weibull": WeibullFitter(),
        "Exponentiel": ExponentialFitter(),
        "Log-Normal": LogNormalFitter(),
    }
    fitted = {}
    for name, fitter in models.items():
        try:
            fitter.fit(durations, event_observed=events, label=name)
            fitted[name] = fitter
        except Exception as e:
            print(f"Erreur {name} : {e}")
    return fitted


def compare_aic(fitted_models, cph=None):
    """
    Tableau comparatif des AIC pour tous les modèles.

    Args:
        fitted_models (dict): Modèles paramétriques ajustés.
        cph: CoxPHFitter ajusté (optionnel, pour comparaison).

    Returns:
        pd.DataFrame: Tableau AIC trié.
    """
    rows = []
    for name, fitter in fitted_models.items():
        try:
            rows.append({
                "Modèle": name,
                "AIC": round(fitter.AIC_, 2),
                "Log-vraisemblance": round(fitter.log_likelihood_, 2),
                "Type": "Paramétrique",
            })
        except Exception:
            pass

    if cph is not None:
        try:
            rows.append({
                "Modèle": "Cox (semi-paramétrique)",
                "AIC": round(cph.AIC_partial_, 2),
                "Log-vraisemblance": round(cph.log_likelihood_, 2),
                "Type": "Semi-paramétrique",
            })
        except Exception:
            pass

    df_aic = pd.DataFrame(rows).sort_values("AIC")
    df_aic["Rang"] = range(1, len(df_aic) + 1)
    df_aic["Meilleur modèle"] = df_aic["Rang"].apply(lambda x: "✅" if x == 1 else "")
    return df_aic


def plot_parametric_vs_km(durations, events, fitted_models):
    """
    Superpose les courbes paramétriques et la courbe KM.

    Args:
        durations (pd.Series): Temps.
        events (pd.Series): Événements.
        fitted_models (dict): Modèles paramétriques ajustés.

    Returns:
        go.Figure
    """
    fig = go.Figure()

    # Kaplan-Meier comme référence
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label="Kaplan-Meier (référence)")
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_survival_function_
    t = sf.index.values

    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([ci.iloc[:, 1].values, ci.iloc[:, 0].values[::-1]]),
        fill="toself", fillcolor="rgba(37,99,235,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=sf.iloc[:, 0].values,
        mode="lines", name="Kaplan-Meier",
        line=dict(color=PALETTE["primary"], width=3, shape="hv"),
    ))

    # Modèles paramétriques
    colors = [PALETTE["danger"], PALETTE["success"], PALETTE["warning"]]
    t_smooth = np.linspace(durations.min(), durations.quantile(0.99), 300)

    for i, (name, fitter) in enumerate(fitted_models.items()):
        try:
            s_smooth = fitter.predict(t_smooth)
            fig.add_trace(go.Scatter(
                x=t_smooth, y=s_smooth,
                mode="lines", name=name,
                line=dict(color=colors[i % len(colors)], width=2, dash="dash"),
                hovertemplate=f"<b>{name}</b><br>t=%{{x:.1f}}<br>S(t)=%{{y:.3f}}<extra></extra>",
            ))
        except Exception:
            pass

    fig = apply_defaults(
        fig,
        title="Comparaison : Kaplan-Meier vs Modèles paramétriques",
        xaxis_title="Temps (mois)",
        yaxis_title="Probabilité de survie S(t)",
    )
    fig.update_layout(yaxis=dict(range=[0, 1.05]))
    return fig
