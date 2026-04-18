"""
modules/log_rank_tests.py
Tests de comparaison des courbes de survie.

Authors: Bouguessa Nour & Sbartai Sami
"""

import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test, multivariate_logrank_test
from itertools import combinations

from utils.stats_helpers import bonferroni_correction


def run_all_weighted_tests(df, time_col, event_col, strat_var):
    """
    Compare les tests pondérés pour 2 groupes.

    Args:
        df (pd.DataFrame): Données.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.
        strat_var (str): Variable de stratification (2 groupes).

    Returns:
        pd.DataFrame: Tableau comparatif.
    """
    groups = sorted(df[strat_var].dropna().unique(), key=str)
    if len(groups) != 2:
        return pd.DataFrame({"Info": [f"{len(groups)} groupes détectés — tests pondérés pour 2 groupes uniquement."]})

    g1, g2 = groups
    t1 = df[df[strat_var] == g1][time_col]
    t2 = df[df[strat_var] == g2][time_col]
    e1 = df[df[strat_var] == g1][event_col]
    e2 = df[df[strat_var] == g2][event_col]

    tests = [
        ("Log-rank (Mantel-Haenszel)", None, {}),
        ("Wilcoxon (Breslow)", "wilcoxon", {}),
        ("Tarone-Ware", "tarone-ware", {}),
        ("Fleming-Harrington (ρ=1, γ=0)", "fleming-harrington", {"p": 1, "q": 0}),
        ("Fleming-Harrington (ρ=0, γ=1)", "fleming-harrington", {"p": 0, "q": 1}),
    ]

    rows = []
    for name, weighting, kwargs in tests:
        try:
            kw = {"durations_A": t1, "durations_B": t2,
                  "event_observed_A": e1, "event_observed_B": e2}
            if weighting:
                kw["weightings"] = weighting
            kw.update(kwargs)
            res = logrank_test(**kw)
            rows.append({
                "Test": name,
                "Statistique": round(res.test_statistic, 4),
                "p-valeur": round(res.p_value, 4),
                "Significatif": "✅" if res.p_value < 0.05 else "❌",
            })
        except Exception as e:
            rows.append({"Test": name, "Statistique": np.nan,
                         "p-valeur": np.nan, "Significatif": f"Erreur: {e}"})

    return pd.DataFrame(rows)


def run_pairwise_logrank(df, time_col, event_col, strat_var):
    """
    Comparaisons deux à deux avec correction Bonferroni.

    Args:
        df (pd.DataFrame): Données.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.
        strat_var (str): Variable de stratification.

    Returns:
        pd.DataFrame
    """
    groups = sorted(df[strat_var].dropna().unique(), key=str)
    raw_p, stats_list, pair_list = [], [], []

    for g1, g2 in combinations(groups, 2):
        m1, m2 = df[strat_var] == g1, df[strat_var] == g2
        try:
            res = logrank_test(df[m1][time_col], df[m2][time_col],
                               df[m1][event_col], df[m2][event_col])
            raw_p.append(res.p_value)
            stats_list.append(round(res.test_statistic, 4))
        except Exception:
            raw_p.append(np.nan)
            stats_list.append(np.nan)
        pair_list.append((str(g1), str(g2)))

    n = len(raw_p)
    rows = []
    for (g1, g2), stat, p in zip(pair_list, stats_list, raw_p):
        p_corr = min(p * n, 1.0) if not np.isnan(p) else np.nan
        rows.append({
            "Groupe 1": g1, "Groupe 2": g2, "χ²": stat,
            "p brute": round(p, 4) if not np.isnan(p) else np.nan,
            "p Bonferroni": round(p_corr, 4) if not np.isnan(p_corr) else np.nan,
            "Significatif": "✅" if (not np.isnan(p_corr) and p_corr < 0.05) else "❌",
        })
    return pd.DataFrame(rows)


def interpret_logrank(p_value, groups, var):
    """
    Interprétation automatique du test log-rank.

    Args:
        p_value (float): P-valeur du test.
        groups (list): Modalités comparées.
        var (str): Variable de stratification.

    Returns:
        str: Texte d'interprétation.
    """
    if p_value < 0.001:
        sig = "très hautement significative (p < 0.001)"
    elif p_value < 0.01:
        sig = "très significative (p < 0.01)"
    elif p_value < 0.05:
        sig = f"significative (p = {p_value:.4f})"
    else:
        sig = f"non significative (p = {p_value:.4f})"

    groups_str = " / ".join(str(g) for g in groups)
    if p_value < 0.05:
        return (f"✅ **H₀ rejetée** : différence de survie entre ({groups_str}) "
                f"est {sig}. Les courbes selon **{var}** sont statistiquement différentes.")
    else:
        return (f"ℹ️ **H₀ non rejetée** : différence de survie entre ({groups_str}) "
                f"est {sig}. Aucune différence significative selon **{var}**.")
