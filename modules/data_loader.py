"""
modules/data_loader.py
Chargement, validation et préparation des données.

Authors: Bouguessa Nour & Sbartai Sami
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

REQUIRED_COLS = ["Time_to_Event", "Event_Observed"]

ENCODING_OPTIONS = {
    "UTF-8": "utf-8",
    "Latin-1": "latin-1",
    "ISO-8859-1": "iso-8859-1",
    "cp1252": "cp1252",
}

SEPARATOR_OPTIONS = {
    "Virgule (,)": ",",
    "Point-virgule (;)": ";",
    "Tabulation (\\t)": "\t",
}


@st.cache_data(show_spinner=False)
def load_csv(file_bytes, encoding, separator):
    """
    Charge un fichier CSV depuis des bytes.

    Args:
        file_bytes (bytes): Contenu du fichier.
        encoding (str): Encodage.
        separator (str): Séparateur de colonnes.

    Returns:
        pd.DataFrame

    Raises:
        ValueError: Si le fichier ne peut pas être parsé.
    """
    try:
        df = pd.read_csv(BytesIO(file_bytes), encoding=encoding, sep=separator)
        return df
    except Exception as e:
        raise ValueError(f"Impossible de lire le CSV : {e}")


def validate_dataframe(df, time_col, event_col):
    """
    Valide que le DataFrame contient les colonnes requises.

    Args:
        df (pd.DataFrame): Données.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.

    Returns:
        list: Messages d'avertissement.
    """
    warnings = []
    if time_col not in df.columns:
        warnings.append(f"Colonne de temps '{time_col}' introuvable.")
    elif df[time_col].isna().any():
        warnings.append(f"'{time_col}' contient des valeurs manquantes.")
    elif (df[time_col] < 0).any():
        warnings.append(f"'{time_col}' contient des valeurs négatives.")
    if event_col not in df.columns:
        warnings.append(f"Colonne d'événement '{event_col}' introuvable.")
    elif not df[event_col].isin([0, 1]).all():
        warnings.append(f"'{event_col}' doit contenir uniquement 0 et 1.")
    if len(df) < 10:
        warnings.append("Moins de 10 observations.")
    return warnings


def add_derived_variables(df):
    """
    Ajoute Tranche_Age et Tranche_BMI.

    Args:
        df (pd.DataFrame): Données source.

    Returns:
        pd.DataFrame: Données enrichies.
    """
    df = df.copy()
    if "Age" in df.columns and "Tranche_Age" not in df.columns:
        df["Tranche_Age"] = pd.cut(
            df["Age"],
            bins=[0, 50, 60, df["Age"].max() + 1],
            labels=["<50", "50-60", ">60"],
            include_lowest=True,
        )
    if "BMI" in df.columns and "Tranche_BMI" not in df.columns:
        df["Tranche_BMI"] = pd.cut(
            df["BMI"],
            bins=[0, 18, 26, df["BMI"].max() + 1],
            labels=["<18", "18-26", ">26"],
            include_lowest=True,
        )
    return df


def get_summary_metrics(df, time_col, event_col):
    """
    Calcule les métriques de résumé de la cohorte.

    Args:
        df (pd.DataFrame): Données filtrées.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.

    Returns:
        dict: Métriques.
    """
    if time_col not in df.columns or event_col not in df.columns:
        return {
            "n_total": len(df),
            "n_events": 0,
            "n_censored": 0,
            "pct_events": 0,
            "pct_censored": 0,
            "median_time": 0,
            "min_time": 0,
            "max_time": 0,
        }
    n = len(df)
    n_events = int(df[event_col].sum())
    n_censored = n - n_events
    return {
        "n_total": n,
        "n_events": n_events,
        "n_censored": n_censored,
        "pct_events": round(100 * n_events / n, 1) if n > 0 else 0,
        "pct_censored": round(100 * n_censored / n, 1) if n > 0 else 0,
        "median_time": round(df[time_col].median(), 2),
        "min_time": round(df[time_col].min(), 2),
        "max_time": round(df[time_col].max(), 2),
    }

def apply_filters(df, filters):
    """
    Applique les filtres interactifs au DataFrame.

    Args:
        df (pd.DataFrame): Données complètes.
        filters (dict): Filtres {colonne: valeurs}.

    Returns:
        pd.DataFrame: Données filtrées.
    """
    df_f = df.copy()
    for col, val in filters.items():
        if col not in df_f.columns:
            continue
        if isinstance(val, tuple) and len(val) == 2:
            df_f = df_f[(df_f[col] >= val[0]) & (df_f[col] <= val[1])]
        elif isinstance(val, list) and len(val) > 0:
            # Convertir en string pour éviter les problèmes avec category
            col_as_str = df_f[col].astype(str)
            val_as_str = [str(v) for v in val]
            df_f = df_f[col_as_str.isin(val_as_str)]
    return df_f

