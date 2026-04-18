"""
modules/preprocessing.py
Nettoyage : doublons, valeurs manquantes, imputation.

Authors: Bouguessa Nour & Sbartai Sami
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def detect_duplicates(df, event_col="Event_Observed"):
    """
    Détecte les lignes dupliquées.

    Args:
        df (pd.DataFrame): Données.
        event_col (str): Colonne événement.

    Returns:
        dict: n_total_duplicates, n_censored_duplicates.
    """
    total_dup = df.duplicated(keep="first")
    censored_dup = (df[df[event_col] == 0].duplicated(keep="first")
                    if event_col in df.columns else pd.Series(dtype=bool))
    return {
        "n_total_duplicates": int(total_dup.sum()),
        "n_censored_duplicates": int(censored_dup.sum()),
        "indices": df[total_dup].index.tolist(),
    }


def remove_duplicates(df):
    """
    Supprime les lignes dupliquées.

    Args:
        df (pd.DataFrame): Données.

    Returns:
        pd.DataFrame
    """
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def missing_summary(df):
    """
    Tableau récapitulatif des valeurs manquantes.

    Args:
        df (pd.DataFrame): Données.

    Returns:
        pd.DataFrame
    """
    rows = []
    for col in df.columns:
        n_miss = df[col].isna().sum()
        rows.append({
            "Variable": col,
            "N manquants": n_miss,
            "% manquants": round(100 * n_miss / len(df), 2),
            "Type": str(df[col].dtype),
        })
    return pd.DataFrame(rows).sort_values("% manquants", ascending=False)


def apply_imputation(df, strategies, time_col, event_col):
    """
    Applique les stratégies d'imputation choisies.

    Args:
        df (pd.DataFrame): Données source.
        strategies (dict): {colonne: stratégie}.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.

    Returns:
        tuple: (DataFrame imputé, list messages).
    """
    df = df.copy()
    messages = []
    rows_before = len(df)

    for col in [time_col, event_col]:
        if col in df.columns and df[col].isna().any():
            n = df[col].isna().sum()
            df = df.dropna(subset=[col])
            messages.append(f"✅ '{col}' : {n} lignes supprimées (variable critique).")

    for col, strategy in strategies.items():
        if col not in df.columns or col in [time_col, event_col]:
            continue
        n_miss = df[col].isna().sum()
        if n_miss == 0:
            continue

        if strategy == "Supprimer les lignes":
            df = df.dropna(subset=[col])
            messages.append(f"✅ '{col}' : {n_miss} lignes supprimées.")
        elif strategy == "Remplacer par la moyenne":
            val = df[col].mean()
            df[col] = df[col].fillna(val)
            messages.append(f"✅ '{col}' : remplacé par la moyenne ({val:.2f}).")
        elif strategy == "Remplacer par la médiane":
            val = df[col].median()
            df[col] = df[col].fillna(val)
            messages.append(f"✅ '{col}' : remplacé par la médiane ({val:.2f}).")
        elif strategy == "Remplacer par le mode":
            val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(val)
            messages.append(f"✅ '{col}' : remplacé par le mode ({val}).")
        elif strategy == "Créer une catégorie 'Inconnu'":
            df[col] = df[col].fillna("Inconnu").astype(str)
            messages.append(f"✅ '{col}' : remplacé par 'Inconnu'.")
        elif strategy == "Imputation KNN":
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if col in num_cols:
                imputer = KNNImputer(n_neighbors=5)
                df[num_cols] = imputer.fit_transform(df[num_cols])
                messages.append(f"✅ '{col}' : imputation KNN appliquée.")
        elif strategy == "Interpolation linéaire":
            df[col] = df[col].interpolate(method="linear")
            messages.append(f"✅ '{col}' : interpolation linéaire appliquée.")
        else:
            messages.append(f"ℹ️ '{col}' : conservé tel quel.")

    df = df.reset_index(drop=True)
    if len(df) != rows_before:
        messages.append(f"📊 Lignes : {rows_before} → {len(df)}.")
    return df, messages


def get_imputation_options(dtype):
    """
    Options d'imputation selon le type de variable.

    Args:
        dtype (str): Type pandas.

    Returns:
        list: Stratégies disponibles.
    """
    if dtype in ["float64", "float32", "int64", "int32"]:
        return ["Conserver tel quel", "Supprimer les lignes",
                "Remplacer par la moyenne", "Remplacer par la médiane",
                "Imputation KNN", "Interpolation linéaire"]
    else:
        return ["Conserver tel quel", "Supprimer les lignes",
                "Remplacer par le mode", "Créer une catégorie 'Inconnu'"]
