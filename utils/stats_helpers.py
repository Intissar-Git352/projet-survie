"""
utils/stats_helpers.py
Fonctions statistiques utilitaires.

Authors: Bouguessa Nour & Sbartai Sami
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, shapiro, kstest
from itertools import combinations


def test_normality(series):
    """
    Test de normalité Shapiro-Wilk (N<=5000) ou Kolmogorov-Smirnov.

    Args:
        series (pd.Series): Série numérique.

    Returns:
        dict: test, statistic, p_value, normal.
    """
    data = series.dropna()
    n = len(data)
    if n < 3:
        return {"test": "N/A", "statistic": np.nan, "p_value": np.nan, "normal": None}
    if n <= 5000:
        stat, p = shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = kstest(data, "norm", args=(data.mean(), data.std()))
        test_name = "Kolmogorov-Smirnov"
    return {"test": test_name, "statistic": round(stat, 4),
            "p_value": round(p, 4), "normal": p > 0.05}


def describe_numeric(df, variables):
    """
    Tableau de statistiques descriptives enrichi.

    Args:
        df (pd.DataFrame): Données.
        variables (list): Variables numériques.

    Returns:
        pd.DataFrame: Tableau de stats.
    """
    rows = []
    for v in variables:
        s = df[v].dropna()
        norm = test_normality(s)
        rows.append({
            "Variable": v,
            "N": len(s),
            "Manquants": df[v].isna().sum(),
            "Moyenne": round(s.mean(), 2),
            "Écart-type": round(s.std(), 2),
            "Médiane": round(s.median(), 2),
            "Q1": round(s.quantile(0.25), 2),
            "Q3": round(s.quantile(0.75), 2),
            "Min": round(s.min(), 2),
            "Max": round(s.max(), 2),
            "Asymétrie": round(s.skew(), 3),
            "Aplatissement": round(s.kurtosis(), 3),
            "Test normalité": norm["test"],
            "p-valeur": norm["p_value"],
        })
    return pd.DataFrame(rows)


def cramers_v(x, y):
    """
    Calcule le V de Cramér entre deux variables catégorielles.

    Args:
        x (pd.Series): Première variable.
        y (pd.Series): Deuxième variable.

    Returns:
        float: V de Cramér.
    """
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    confusion = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion)
    n = len(x)
    r, k = confusion.shape
    phi2 = max(0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return round(np.sqrt(phi2 / denom), 4)


def cramers_v_matrix(df, cat_vars):
    """
    Matrice du V de Cramér pour toutes les paires.

    Args:
        df (pd.DataFrame): Données.
        cat_vars (list): Variables catégorielles.

    Returns:
        pd.DataFrame: Matrice carrée.
    """
    mat = pd.DataFrame(index=cat_vars, columns=cat_vars, dtype=float)
    for v in cat_vars:
        mat.loc[v, v] = 1.0
    for v1, v2 in combinations(cat_vars, 2):
        val = cramers_v(df[v1], df[v2])
        mat.loc[v1, v2] = val
        mat.loc[v2, v1] = val
    return mat


def chi2_test(df, col1, col2):
    """
    Test du Chi² entre deux variables catégorielles.

    Args:
        df (pd.DataFrame): Données.
        col1 (str): Première variable.
        col2 (str): Deuxième variable.

    Returns:
        dict: chi2, pvalue, dof, significant, crosstab.
    """
    try:
        ct = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, _ = chi2_contingency(ct)
        return {"chi2": round(chi2, 3), "pvalue": round(p, 4),
                "dof": dof, "significant": p < 0.05, "crosstab": ct}
    except Exception as e:
        return {"chi2": np.nan, "pvalue": np.nan, "dof": 0,
                "significant": False, "error": str(e)}


def bonferroni_correction(p_values, alpha=0.05):
    """
    Correction de Bonferroni.

    Args:
        p_values (list): P-valeurs brutes.
        alpha (float): Seuil global.

    Returns:
        list: P-valeurs corrigées.
    """
    n = len(p_values)
    return [min(1.0, p * n) for p in p_values]


def compute_conditional_survival(kmf, t_condition, t_target):
    """
    Survie conditionnelle P(T > t_target | T > t_condition).

    Args:
        kmf: KaplanMeierFitter ajusté.
        t_condition (float): Temps conditionnel t0.
        t_target (float): Temps cible t.

    Returns:
        float: Probabilité conditionnelle.
    """
    if t_target <= t_condition:
        return np.nan
    s_target = kmf.predict(t_target)
    s_cond = kmf.predict(t_condition)
    if s_cond == 0:
        return np.nan
    return round(float(s_target) / float(s_cond), 4)


def get_variable_types(df):
    """
    Détermine le type statistique de chaque colonne.

    Args:
        df (pd.DataFrame): Données.

    Returns:
        pd.DataFrame: Types de variables.
    """
    rows = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct_missing = round(100 * n_missing / len(df), 2)
        card = df[col].nunique()
        dtype = str(df[col].dtype)

        if df[col].dtype in [np.float64, np.float32]:
            stat_type = "Numérique continue"
        elif df[col].dtype in [np.int64, np.int32] and card > 10:
            stat_type = "Numérique discrète"
        elif card == 2:
            stat_type = "Binaire"
        elif df[col].dtype == "object" or card <= 20:
            stat_type = "Catégorielle nominale"
        else:
            stat_type = "Autre"

        rows.append({
            "Variable": col, "Type Python": dtype,
            "Type Statistique": stat_type,
            "Manquants": n_missing, "% Manquants": pct_missing,
            "Cardinalité": card,
        })
    return pd.DataFrame(rows)


def martingale_residuals(cph, df_cox):
    """
    Résidus de Martingale du modèle de Cox.

    Args:
        cph: CoxPHFitter ajusté.
        df_cox (pd.DataFrame): Données d'entraînement.

    Returns:
        pd.Series
    """
    return cph.compute_residuals(df_cox, kind="martingale")["martingale"]


def deviance_residuals(cph, df_cox):
    """
    Résidus de Déviance du modèle de Cox.

    Args:
        cph: CoxPHFitter ajusté.
        df_cox (pd.DataFrame): Données d'entraînement.

    Returns:
        pd.Series
    """
    mart = martingale_residuals(cph, df_cox)
    events = df_cox[cph.event_col]
    dev = np.sign(events - mart) * np.sqrt(
        -2 * (mart + events * np.log(events - mart + 1e-10))
    )
    return pd.Series(dev, name="deviance")
