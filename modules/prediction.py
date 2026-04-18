"""
modules/prediction.py
Prédiction individuelle de survie via KM et Cox.

Authors: Bouguessa Nour & Sbartai Sami
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import CoxPHFitter, KaplanMeierFitter

from utils.plots import apply_defaults, PALETTE

TIME_POINTS = list(range(0, 126, 6))
KEY_TIME_POINTS = [12, 24, 36, 60]


def predict_survival_cox(cph, patient_profile, df_cox):
    """
    Prédit la courbe de survie individualisée via Cox.

    Args:
        cph: CoxPHFitter ajusté.
        patient_profile (dict): Profil du patient.
        df_cox (pd.DataFrame): Données d'entraînement.

    Returns:
        tuple: (pd.DataFrame S(t), float score_risque)
    """
    profile_df = _build_profile_df(patient_profile, df_cox, cph)
    sf = cph.predict_survival_function(profile_df)
    risk_score = float(np.exp(cph.predict_log_partial_hazard(profile_df).iloc[0]))
    result = pd.DataFrame({
        "Temps (mois)": sf.index,
        "S(t) prédit": sf.iloc[:, 0].values.round(4),
    })
    return result, risk_score


def predict_survival_baseline(cph):
    """
    Survie basale du modèle de Cox.

    Args:
        cph: CoxPHFitter ajusté.

    Returns:
        pd.DataFrame
    """
    baseline = cph.baseline_survival_
    return pd.DataFrame({
        "Temps (mois)": baseline.index,
        "S₀(t) basale": baseline.iloc[:, 0].values.round(4),
    })


def plot_individual_survival(sf_individual, sf_baseline, patient_label="Patient"):
    """
    Courbe de survie individuelle vs basale.

    Args:
        sf_individual (pd.DataFrame): S(t) prédit.
        sf_baseline (pd.DataFrame): S₀(t) basale.
        patient_label (str): Label du patient.

    Returns:
        go.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sf_individual["Temps (mois)"], y=sf_individual["S(t) prédit"],
        mode="lines", name=f"🩺 {patient_label}",
        line=dict(color=PALETTE["danger"], width=3),
        hovertemplate="t=%{x:.1f} mois<br>S(t)=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sf_baseline["Temps (mois)"], y=sf_baseline["S₀(t) basale"],
        mode="lines", name="Référence (baseline)",
        line=dict(color=PALETTE["neutral"], width=2, dash="dash"),
        hovertemplate="t=%{x:.1f} mois<br>S₀(t)=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color=PALETTE["neutral"],
                  line_width=1, annotation_text="Médiane")
    fig = apply_defaults(fig, title="Courbe de survie individualisée — Modèle de Cox",
                         xaxis_title="Temps (mois)",
                         yaxis_title="Probabilité de survie S(t)")
    fig.update_layout(yaxis=dict(range=[0, 1.05]))
    return fig


def get_probability_table(sf, time_col="Temps (mois)", surv_col="S(t) prédit"):
    """
    Tableau de probabilités aux temps clés.

    Args:
        sf (pd.DataFrame): S(t) prédit.
        time_col (str): Colonne de temps.
        surv_col (str): Colonne de survie.

    Returns:
        pd.DataFrame
    """
    rows = []
    for t in TIME_POINTS:
        idx = (sf[time_col] - t).abs().idxmin()
        s = float(sf.loc[idx, surv_col])
        color = "🟢" if s > 0.75 else ("🟠" if s > 0.5 else "🔴")
        rows.append({"Temps (mois)": t, "S(t)": round(s, 4), "Niveau": color})
    return pd.DataFrame(rows)


def compute_waterfall_contributions(cph, patient_profile, df_cox):
    """
    Contributions de chaque variable au score de risque.

    Args:
        cph: CoxPHFitter ajusté.
        patient_profile (dict): Profil patient.
        df_cox (pd.DataFrame): Données d'entraînement.

    Returns:
        dict: {variable: contribution}
    """
    profile_df = _build_profile_df(patient_profile, df_cox, cph)
    coef = cph.params_
    contributions = {}
    for col in coef.index:
        if col in profile_df.columns:
            val = float(profile_df[col].iloc[0])
            contributions[col] = round(float(coef[col]) * val, 4)
    return contributions


def predict_km_group(df, time_col, event_col, patient_profile,
                     strat_var="Tranche_Age"):
    """
    S(t) aux temps clés via KM du groupe le plus proche.

    Args:
        df (pd.DataFrame): Données.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.
        patient_profile (dict): Profil patient.
        strat_var (str): Variable de stratification.

    Returns:
        pd.DataFrame
    """
    age = patient_profile.get("Age", 60)
    if strat_var == "Tranche_Age" and strat_var in df.columns:
        grp = "<50" if age < 50 else ("50-60" if age <= 60 else ">60")
    else:
        grp = df[strat_var].mode().iloc[0] if strat_var in df.columns else None

    if grp is not None and strat_var in df.columns and grp in df[strat_var].values:
        sub = df[df[strat_var] == grp]
    else:
        sub = df

    kmf = KaplanMeierFitter()
    kmf.fit(sub[time_col], sub[event_col])
    rows = []
    for t in KEY_TIME_POINTS:
        s = round(float(kmf.predict(t)), 4)
        color = "🟢" if s > 0.75 else ("🟠" if s > 0.5 else "🔴")
        rows.append({"Temps (mois)": t, "S(t) KM groupe": s, "Niveau": color})
    return pd.DataFrame(rows)


def _build_profile_df(patient_profile, df_cox, cph):
    """
    Construit le DataFrame patient aligné sur les colonnes du modèle.

    Args:
        patient_profile (dict): Profil brut.
        df_cox (pd.DataFrame): Données d'entraînement.
        cph: CoxPHFitter ajusté.

    Returns:
        pd.DataFrame: Ligne unique.
    """
    model_cols = list(cph.params_.index)
    sex = patient_profile.get("Sex", "Male")
    treatment = patient_profile.get("Treatment", "Standard")
    activity = patient_profile.get("Physical_Activity", "Moderate")
    row = {}

    for col in model_cols:
        if col == "Age":
            row[col] = float(patient_profile.get("Age", 60))
        elif col == "Smoker":
            row[col] = int(patient_profile.get("Smoker", 0))
        elif col == "BMI":
            row[col] = float(patient_profile.get("BMI", 25.0))
        elif col == "Comorbidities":
            row[col] = float(patient_profile.get("Comorbidities", 1))
        elif col == "Sex_Male":
            row[col] = 1 if sex == "Male" else 0
        elif "Treatment_Standard" in col:
            row[col] = 1 if treatment == "Standard" else 0
        elif "Physical_Activity_Low" in col:
            row[col] = 1 if activity == "Low" else 0
        elif "Physical_Activity_Moderate" in col:
            row[col] = 1 if activity == "Moderate" else 0
        else:
            row[col] = float(df_cox[col].median()) if col in df_cox.columns else 0

    return pd.DataFrame([row])
