"""
modules/descriptive.py
Statistiques descriptives et graphiques associés.

Authors: Bouguessa Nour & Sbartai Sami
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.plots import apply_defaults, PALETTE, GROUP_COLORS
from utils.stats_helpers import describe_numeric, cramers_v_matrix, chi2_test

NUM_VARS = ["Age", "BMI", "Comorbidities", "Time_to_Event"]
CAT_VARS = ["Sex", "Smoker", "Treatment", "Physical_Activity"]


def get_numeric_stats(df):
    """
    Tableau descriptif pour les variables numériques.

    Args:
        df (pd.DataFrame): Données.

    Returns:
        pd.DataFrame
    """
    available = [v for v in NUM_VARS if v in df.columns]
    return describe_numeric(df, available)


def get_qualitative_stats(df):
    """
    Tableaux effectifs/proportions pour variables qualitatives.

    Args:
        df (pd.DataFrame): Données.

    Returns:
        dict: {variable: DataFrame}
    """
    results = {}
    for col in CAT_VARS:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        props = df[col].value_counts(normalize=True).round(4)
        cum = props.cumsum().round(4)
        results[col] = pd.DataFrame({
            "Modalité": counts.index,
            "Effectif": counts.values,
            "Proportion": props.values,
            "Proportion cumulée": cum.values,
        })
    return results


def get_cramers_matrix(df):
    """
    Matrice du V de Cramér pour variables catégorielles.

    Args:
        df (pd.DataFrame): Données.

    Returns:
        pd.DataFrame
    """
    available = [v for v in CAT_VARS if v in df.columns]
    return cramers_v_matrix(df, available)


def plot_cramers_heatmap(cramers_df):
    """
    Heatmap du V de Cramér.

    Args:
        cramers_df (pd.DataFrame): Matrice V de Cramér.

    Returns:
        go.Figure
    """
    vals = cramers_df.values.astype(float)
    fig = go.Figure(go.Heatmap(
        z=vals, x=cramers_df.columns.tolist(), y=cramers_df.index.tolist(),
        colorscale="Blues", zmin=0, zmax=1,
        text=np.round(vals, 3), texttemplate="%{text}",
        colorbar=dict(title="V de Cramér"),
    ))
    return apply_defaults(fig, title="Associations — V de Cramér")


def plot_bar_categorical(df, variable):
    """
    Diagramme en barres pour une variable qualitative.

    Args:
        df (pd.DataFrame): Données.
        variable (str): Variable catégorielle.

    Returns:
        go.Figure
    """
    counts = df[variable].value_counts()
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(), y=counts.values,
        marker_color=GROUP_COLORS[:len(counts)],
        text=counts.values, textposition="outside",
    ))
    return apply_defaults(fig, title=f"Répartition — {variable}",
                          xaxis_title=variable, yaxis_title="Effectif")


def plot_stacked_bar(df, var_x, var_group):
    """
    Barres empilées normalisées.

    Args:
        df (pd.DataFrame): Données.
        var_x (str): Variable en abscisse.
        var_group (str): Variable de groupement.

    Returns:
        go.Figure
    """
    ct = pd.crosstab(df[var_x], df[var_group], normalize="index") * 100
    fig = go.Figure()
    for i, col in enumerate(ct.columns):
        fig.add_trace(go.Bar(
            x=ct.index.astype(str), y=ct[col],
            name=str(col),
            marker_color=GROUP_COLORS[i % len(GROUP_COLORS)],
        ))
    fig.update_layout(barmode="stack")
    return apply_defaults(fig, title=f"{var_x} par {var_group} (%)",
                          xaxis_title=var_x, yaxis_title="%")


def plot_scatter_matrix(df, color_by=None):
    """
    Scatter matrix interactive sur variables numériques.

    Args:
        df (pd.DataFrame): Données.
        color_by (str): Variable catégorielle pour couleur.

    Returns:
        go.Figure
    """
    available = [v for v in NUM_VARS if v in df.columns]
    fig = px.scatter_matrix(
        df, dimensions=available,
        color=color_by if (color_by and color_by in df.columns) else None,
        color_discrete_sequence=GROUP_COLORS,
        title="Scatter matrix — variables numériques",
        template="plotly_white",
    )
    fig.update_traces(diagonal_visible=False,
                      marker=dict(size=3, opacity=0.5))
    fig.update_layout(font=dict(size=11), height=600)
    return fig


def plot_bivariate_survival(df, time_col, event_col, var):
    """
    Analyse bivariée Survie × Variable.

    Args:
        df (pd.DataFrame): Données.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.
        var (str): Variable explicative.

    Returns:
        go.Figure
    """
    groups = df[var].dropna().unique()
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"Temps de suivi par {var}",
        f"Proportion d'événements par {var}",
    ])

    for i, grp in enumerate(groups):
        data = df[df[var] == grp][time_col].dropna()
        fig.add_trace(go.Box(
            y=data, name=str(grp),
            marker_color=GROUP_COLORS[i % len(GROUP_COLORS)],
        ), row=1, col=1)

    event_props = df.groupby(var)[event_col].mean().reset_index()
    fig.add_trace(go.Bar(
        x=event_props[var].astype(str),
        y=event_props[event_col],
        marker_color=GROUP_COLORS[:len(event_props)],
        showlegend=False,
        text=(event_props[event_col] * 100).round(1).astype(str) + "%",
        textposition="outside",
    ), row=1, col=2)

    fig.update_layout(title=f"Analyse bivariée : Survie × {var}",
                      height=420, template="plotly_white")
    return fig
