"""
modules/bonus_bootstrap.py
Bootstrap des intervalles de confiance pour la courbe KM.

Authors: Bouguessa Nour & Sbartai Sami
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter

from utils.plots import apply_defaults, PALETTE


def bootstrap_km(df, time_col, event_col, n_bootstrap=500, alpha=0.05, seed=42):
    """
    Calcule les IC par bootstrap pour la courbe Kaplan-Meier.

    Args:
        df (pd.DataFrame): Données.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.
        n_bootstrap (int): Nombre de répétitions bootstrap.
        alpha (float): Niveau de risque (0.05 = IC 95%).
        seed (int): Graine aléatoire.

    Returns:
        go.Figure: Courbe KM avec IC bootstrap et IC analytique.
    """
    np.random.seed(seed)
    n = len(df)

    # Grille de temps commune
    kmf_ref = KaplanMeierFitter()
    kmf_ref.fit(df[time_col], df[event_col])
    t_grid = kmf_ref.survival_function_.index.values

    # Bootstrap
    boot_curves = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        df_boot = df.iloc[idx]
        kmf_b = KaplanMeierFitter()
        kmf_b.fit(df_boot[time_col], df_boot[event_col])
        s_interp = np.array([float(kmf_b.predict(t)) for t in t_grid])
        boot_curves.append(s_interp)

    boot_matrix = np.array(boot_curves)
    lower_boot = np.percentile(boot_matrix, 100 * alpha / 2, axis=0)
    upper_boot = np.percentile(boot_matrix, 100 * (1 - alpha / 2), axis=0)
    s_ref = kmf_ref.survival_function_.iloc[:, 0].values

    # IC analytique (Greenwood)
    ci_analytic = kmf_ref.confidence_interval_survival_function_
    ci_low_a = ci_analytic.iloc[:, 0].values
    ci_high_a = ci_analytic.iloc[:, 1].values

    fig = go.Figure()

    # IC bootstrap
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([upper_boot, lower_boot[::-1]]),
        fill="toself",
        fillcolor="rgba(220,38,38,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True,
        name=f"IC {int((1-alpha)*100)}% Bootstrap",
        hoverinfo="skip",
    ))

    # IC analytique
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([ci_high_a, ci_low_a[::-1]]),
        fill="toself",
        fillcolor="rgba(37,99,235,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True,
        name=f"IC {int((1-alpha)*100)}% Analytique (Greenwood)",
        hoverinfo="skip",
    ))

    # Courbe KM
    fig.add_trace(go.Scatter(
        x=t_grid, y=s_ref,
        mode="lines", name="S(t) Kaplan-Meier",
        line=dict(color=PALETTE["primary"], width=2.5, shape="hv"),
        hovertemplate="t=%{x:.1f}<br>S(t)=%{y:.3f}<extra></extra>",
    ))

    fig = apply_defaults(
        fig,
        title=f"Courbe KM avec IC Bootstrap (N={n_bootstrap}) vs Analytique",
        xaxis_title="Temps (mois)",
        yaxis_title="Probabilité de survie S(t)",
    )
    fig.update_layout(yaxis=dict(range=[0, 1.05]))
    return fig, {
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "ic_lower_boot": lower_boot,
        "ic_upper_boot": upper_boot,
        "ic_lower_analytic": ci_low_a,
        "ic_upper_analytic": ci_high_a,
    }
