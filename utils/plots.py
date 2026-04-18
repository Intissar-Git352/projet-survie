"""
utils/plots.py
Fonctions graphiques réutilisables basées sur Plotly.
Palette de couleurs cohérente pour toute l'application.

Authors: Bouguessa Nour & Sbartai Sami
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# ── Palette globale ────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#7C3AED",
    "success":   "#059669",
    "warning":   "#D97706",
    "danger":    "#DC2626",
    "info":      "#0891B2",
    "neutral":   "#6B7280",
    "light":     "#F3F4F6",
}

GROUP_COLORS = [
    "#2563EB", "#DC2626", "#059669", "#D97706",
    "#7C3AED", "#0891B2", "#F59E0B", "#10B981",
]

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13, color="#1F2937"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#E5E7EB",
        borderwidth=1,
    ),
)


def apply_defaults(fig, title="", xaxis_title="", yaxis_title=""):
    """
    Applique le layout standardisé à une figure Plotly.

    Args:
        fig (go.Figure): Figure à styler.
        title (str): Titre du graphique.
        xaxis_title (str): Label axe X.
        yaxis_title (str): Label axe Y.

    Returns:
        go.Figure: Figure avec le style appliqué.
    """
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=16, color="#111827"), x=0.02),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6", zeroline=False)
    return fig


def _hex_to_rgba(hex_color, alpha=1.0):
    """Convertit une couleur hex en string rgba."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def plot_km_curve(kmf, label="Survie globale", color=None, fig=None, show_ci=True):
    """
    Trace une courbe Kaplan-Meier avec intervalle de confiance.

    Args:
        kmf: KaplanMeierFitter ajusté.
        label (str): Nom de la courbe.
        color (str): Couleur hex.
        fig (go.Figure): Figure existante pour superposition.
        show_ci (bool): Afficher l'IC 95%.

    Returns:
        go.Figure
    """
    if fig is None:
        fig = go.Figure()
    if color is None:
        color = PALETTE["primary"]

    sf = kmf.survival_function_
    ci = kmf.confidence_interval_survival_function_
    t = sf.index.values
    s = sf.iloc[:, 0].values
    ci_low = ci.iloc[:, 0].values
    ci_high = ci.iloc[:, 1].values

    if show_ci:
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([ci_high, ci_low[::-1]]),
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.15),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{label} IC 95%",
        ))

    fig.add_trace(go.Scatter(
        x=t, y=s,
        mode="lines",
        name=label,
        line=dict(color=color, width=2.5, shape="hv"),
        hovertemplate=f"<b>{label}</b><br>t=%{{x:.1f}} mois<br>S(t)=%{{y:.3f}}<extra></extra>",
    ))

    try:
        event_table = kmf.event_table
        censored_t = event_table.index[event_table["censored"] > 0]
        censored_s = [float(kmf.predict(ct)) for ct in censored_t]
        if len(censored_t) > 0:
            fig.add_trace(go.Scatter(
                x=censored_t, y=censored_s,
                mode="markers",
                marker=dict(symbol="line-ns", size=8, color=color,
                            line=dict(width=1.5, color=color)),
                showlegend=False,
                hoverinfo="skip",
                name=f"{label} censures",
            ))
    except Exception:
        pass

    return fig


def plot_na_curve(naf, label="Nelson-Aalen", color=None, fig=None, show_ci=True):
    """
    Trace la courbe du risque cumulé Nelson-Aalen.

    Args:
        naf: NelsonAalenFitter ajusté.
        label (str): Nom dans la légende.
        color (str): Couleur hex.
        fig (go.Figure): Figure existante.
        show_ci (bool): Afficher l'IC 95%.

    Returns:
        go.Figure
    """
    if fig is None:
        fig = go.Figure()
    if color is None:
        color = PALETTE["secondary"]

    t = naf.cumulative_hazard_.index.values
    h = naf.cumulative_hazard_.iloc[:, 0].values
    ci = naf.confidence_interval_cumulative_hazard_
    ci_low = ci.iloc[:, 0].values
    ci_high = ci.iloc[:, 1].values

    if show_ci:
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([ci_high, ci_low[::-1]]),
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.15),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=t, y=h,
        mode="lines",
        name=label,
        line=dict(color=color, width=2.5),
        hovertemplate=f"<b>{label}</b><br>t=%{{x:.1f}}<br>H(t)=%{{y:.3f}}<extra></extra>",
    ))

    return fig


def plot_forest(coef_df):
    """
    Forest plot des Hazard Ratios du modèle de Cox.

    Args:
        coef_df (pd.DataFrame): Colonnes ['variable','HR','lower_95','upper_95','p'].

    Returns:
        go.Figure
    """
    fig = go.Figure()

    for i, row in coef_df.iterrows():
        color = PALETTE["danger"] if row["HR"] > 1 else PALETTE["success"]
        sig = "★ " if row.get("p", 1) < 0.05 else ""
        fig.add_trace(go.Scatter(
            x=[row["HR"]],
            y=[f"{sig}{row['variable']}"],
            mode="markers",
            marker=dict(color=color, size=12, symbol="diamond"),
            error_x=dict(
                type="data", symmetric=False,
                array=[row["upper_95"] - row["HR"]],
                arrayminus=[row["HR"] - row["lower_95"]],
                color=color, thickness=2, width=6,
            ),
            name=row["variable"],
            showlegend=False,
            hovertemplate=(
                f"<b>{row['variable']}</b><br>"
                f"HR={row['HR']:.3f}<br>"
                f"IC 95% [{row['lower_95']:.3f}–{row['upper_95']:.3f}]<br>"
                f"p={row.get('p', float('nan')):.4f}<extra></extra>"
            ),
        ))

    fig.add_vline(x=1, line_dash="dash", line_color=PALETTE["neutral"], line_width=1.5)
    fig = apply_defaults(fig,
                         title="Forest Plot — Hazard Ratios (modèle de Cox)",
                         xaxis_title="Hazard Ratio (échelle log)",
                         yaxis_title="Variable")
    fig.update_xaxes(type="log")
    fig.update_layout(height=max(300, len(coef_df) * 55 + 100))
    return fig


def plot_histogram_kde(df, variable, group_by=None):
    """
    Histogramme avec courbe KDE, option superposition par groupe.

    Args:
        df (pd.DataFrame): Données.
        variable (str): Variable numérique.
        group_by (str): Variable catégorielle (optionnel).

    Returns:
        go.Figure
    """
    from scipy.stats import gaussian_kde
    fig = go.Figure()

    groups = df[group_by].unique() if group_by else [None]
    for i, grp in enumerate(groups):
        data = df[variable].dropna() if grp is None else df[df[group_by] == grp][variable].dropna()
        label = str(grp) if grp is not None else variable
        color = GROUP_COLORS[i % len(GROUP_COLORS)]

        fig.add_trace(go.Histogram(
            x=data, name=label, opacity=0.55,
            marker_color=color, histnorm="probability density", nbinsx=30,
        ))

        if len(data) > 5:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 300)
            fig.add_trace(go.Scatter(
                x=x_range, y=kde(x_range),
                mode="lines", name=f"KDE {label}",
                line=dict(color=color, width=2),
            ))

    fig = apply_defaults(
        fig,
        title=f"Distribution de {variable}" + (f" par {group_by}" if group_by else ""),
        xaxis_title=variable, yaxis_title="Densité",
    )
    fig.update_layout(barmode="overlay")
    return fig


def plot_boxplot(df, variable, group_by):
    """
    Boxplot d'une variable numérique par groupe.

    Args:
        df (pd.DataFrame): Données.
        variable (str): Variable numérique.
        group_by (str): Variable de regroupement.

    Returns:
        go.Figure
    """
    groups = df[group_by].dropna().unique()
    fig = go.Figure()
    for i, grp in enumerate(groups):
        data = df[df[group_by] == grp][variable].dropna()
        fig.add_trace(go.Box(
            y=data, name=str(grp),
            marker_color=GROUP_COLORS[i % len(GROUP_COLORS)],
            boxmean="sd",
        ))
    return apply_defaults(fig, title=f"{variable} selon {group_by}",
                          xaxis_title=group_by, yaxis_title=variable)


def plot_correlation_matrix(df, variables):
    """
    Heatmap de corrélation de Pearson.

    Args:
        df (pd.DataFrame): Données.
        variables (list): Variables numériques.

    Returns:
        go.Figure
    """
    corr = df[variables].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=11), colorbar=dict(title="r"),
        zmin=-1, zmax=1,
    ))
    fig = apply_defaults(fig, title="Matrice de corrélation de Pearson")
    fig.update_layout(height=450)
    return fig


def plot_waterfall(contributions):
    """
    Diagramme en cascade des contributions au score de risque Cox.

    Args:
        contributions (dict): {variable: valeur_contribution}.

    Returns:
        go.Figure
    """
    vars_ = list(contributions.keys())
    vals_ = list(contributions.values())

    fig = go.Figure(go.Waterfall(
        name="Contribution", orientation="h",
        measure=["relative"] * len(vars_),
        y=vars_, x=vals_,
        connector=dict(line=dict(color=PALETTE["neutral"], width=1)),
        increasing=dict(marker=dict(color=PALETTE["danger"])),
        decreasing=dict(marker=dict(color=PALETTE["success"])),
        text=[f"{v:+.3f}" for v in vals_],
        textposition="outside",
    ))
    fig = apply_defaults(fig, title="Contributions au score de risque (Cox)",
                         xaxis_title="Contribution log-risque", yaxis_title="Variable")
    fig.update_layout(height=max(300, len(vars_) * 45 + 100))
    return fig


def plot_gauge(risk_score, max_score=3.0):
    """
    Jauge du niveau de risque individuel.

    Args:
        risk_score (float): Score de risque relatif.
        max_score (float): Valeur maximale.

    Returns:
        go.Figure
    """
    level = "Faible" if risk_score < 1.5 else ("Modéré" if risk_score < 2.5 else "Élevé")
    color = PALETTE["success"] if risk_score < 1.5 else (
        PALETTE["warning"] if risk_score < 2.5 else PALETTE["danger"]
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        delta=dict(reference=1.0, valueformat=".2f"),
        gauge=dict(
            axis=dict(range=[0, max_score], tickwidth=1),
            bar=dict(color=color),
            steps=[
                dict(range=[0, 1.5], color="#DCFCE7"),
                dict(range=[1.5, 2.5], color="#FEF9C3"),
                dict(range=[2.5, max_score], color="#FEE2E2"),
            ],
            threshold=dict(line=dict(color=PALETTE["danger"], width=3),
                           thickness=0.75, value=1.0),
        ),
        title=dict(text=f"Risque relatif — Niveau : <b>{level}</b>", font=dict(size=14)),
    ))
    fig.update_layout(height=280, **{k: v for k, v in LAYOUT_DEFAULTS.items()
                                      if k != "template"})
    return fig
