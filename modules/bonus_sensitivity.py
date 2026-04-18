"""
modules/bonus_sensitivity.py
Analyse de sensibilité à la censure (best case / worst case).

Authors: Bouguessa Nour & Sbartai Sami
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter

from utils.plots import apply_defaults, GROUP_COLORS, PALETTE


def sensitivity_analysis(df, time_col, event_col):
    """
    Analyse de sensibilité : best case, worst case, observed.

    Best case  : tous les censurés sont supposés survivants (Event=0 reste 0).
    Worst case : tous les censurés sont supposés avoir eu l'événement (Event=1).

    Args:
        df (pd.DataFrame): Données.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.

    Returns:
        go.Figure: Courbes KM pour les 3 scénarios.
    """
    fig = go.Figure()
    scenarios = {
        "Observé (référence)": df[event_col].copy(),
        "Best case (censurés = survivants)": df[event_col].copy(),
        "Worst case (censurés = événements)": df[event_col].copy(),
    }

    # Worst case : les censurés deviennent des événements
    worst = df[event_col].copy()
    worst[worst == 0] = 1
    scenarios["Worst case (censurés = événements)"] = worst

    colors = [PALETTE["primary"], PALETTE["success"], PALETTE["danger"]]

    for i, (label, events) in enumerate(scenarios.items()):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df[time_col], event_observed=events, label=label)

        sf = kmf.survival_function_
        ci = kmf.confidence_interval_survival_function_
        t = sf.index.values
        s = sf.iloc[:, 0].values
        color = colors[i]

        # IC
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([ci.iloc[:, 1].values, ci.iloc[:, 0].values[::-1]]),
            fill="toself",
            fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba")
            if "rgb" in color else f"rgba(0,0,0,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=t, y=s, mode="lines", name=label,
            line=dict(color=color, width=2.5,
                      dash="dot" if "case" in label.lower() else "solid"),
            hovertemplate=f"<b>{label}</b><br>t=%{{x:.1f}}<br>S(t)=%{{y:.3f}}<extra></extra>",
        ))

    fig = apply_defaults(
        fig,
        title="Analyse de sensibilité à la censure",
        xaxis_title="Temps (mois)",
        yaxis_title="Probabilité de survie S(t)",
    )
    fig.update_layout(yaxis=dict(range=[0, 1.05]))
    return fig
