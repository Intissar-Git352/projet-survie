"""
modules/bonus_tab.py
Contenu de l'onglet bonus pour app.py.
Appelé depuis app.py pour afficher les analyses bonus.

Authors: Bouguessa Nour & Sbartai Sami
"""

import streamlit as st
import pandas as pd
import numpy as np


def render_bonus_tab(df, time_col, event_col, cph=None):
    """
    Affiche l'onglet bonus avec toutes les analyses avancées.

    Args:
        df (pd.DataFrame): Données filtrées.
        time_col (str): Colonne de temps.
        event_col (str): Colonne d'événement.
        cph: CoxPHFitter ajusté (optionnel).
    """
    st.subheader("⭐ Analyses avancées — Niveau Thèse")

    bonus_tabs = st.tabs([
        "📊 Sensibilité censure",
        "🔁 Bootstrap IC",
        "📐 Modèles paramétriques",
    ])

    # ── Bonus 1 : Sensibilité à la censure ────────────────────────────────────
    with bonus_tabs[0]:
        st.markdown("#### Analyse de sensibilité à la censure")
        st.markdown("""
        Cette analyse teste l'impact de différentes hypothèses sur les patients censurés :
        - **Observé** : courbe KM standard
        - **Best case** : les censurés sont supposés survivants jusqu'à la fin
        - **Worst case** : les censurés sont supposés avoir eu l'événement au moment de la censure
        """)
        try:
            from modules.bonus_sensitivity import sensitivity_analysis
            with st.spinner("Calcul de la sensibilité…"):
                fig_sens = sensitivity_analysis(df, time_col, event_col)
            st.plotly_chart(fig_sens, use_container_width=True)

            with st.expander("ℹ️ Interprétation"):
                st.markdown("""
                - Si les trois courbes sont **proches** : les résultats sont **robustes** à la censure.
                - Si elles sont **très écartées** : la censure a un impact important
                  et les conclusions doivent être nuancées.
                - En pratique, on espère que la courbe observée reste proche du best case.
                """)
        except Exception as e:
            st.error(f"Erreur sensibilité : {e}")

    # ── Bonus 2 : Bootstrap IC ─────────────────────────────────────────────────
    with bonus_tabs[1]:
        st.markdown("#### Bootstrap des intervalles de confiance")
        st.markdown("""
        Compare les IC analytiques (formule de Greenwood) avec les IC estimés
        par bootstrap (rééchantillonnage avec remise).
        """)

        n_boot = st.slider("Nombre de répétitions bootstrap", 100, 1000, 300, step=100)
        alpha = st.select_slider("Niveau de confiance", [0.10, 0.05, 0.01],
                                  value=0.05,
                                  format_func=lambda x: f"{int((1-x)*100)}%")

        if st.button("🔁 Lancer le bootstrap", type="primary"):
            try:
                from modules.bonus_bootstrap import bootstrap_km
                with st.spinner(f"Bootstrap en cours ({n_boot} répétitions)…"):
                    fig_boot, stats_boot = bootstrap_km(
                        df, time_col, event_col,
                        n_bootstrap=n_boot, alpha=alpha
                    )
                st.plotly_chart(fig_boot, use_container_width=True)
                st.info(f"✅ Bootstrap terminé — {n_boot} répétitions, "
                        f"IC à {int((1-alpha)*100)}%.")

                with st.expander("ℹ️ Interprétation"):
                    st.markdown("""
                    - Les IC **bootstrap** sont non-paramétriques : ils ne supposent
                      aucune forme de distribution.
                    - Si les IC bootstrap et analytiques sont **similaires** :
                      l'approximation de Greenwood est valide.
                    - Si les IC bootstrap sont **plus larges** : la formule analytique
                      sous-estime l'incertitude.
                    """)
            except Exception as e:
                st.error(f"Erreur bootstrap : {e}")

    # ── Bonus 3 : Modèles paramétriques ───────────────────────────────────────
    with bonus_tabs[2]:
        st.markdown("#### Modèles de survie paramétriques")
        st.markdown("""
        Compare les modèles **Weibull**, **Exponentiel** et **Log-Normal**
        à la courbe Kaplan-Meier et au modèle de Cox via l'AIC.
        """)

        if st.button("📐 Ajuster les modèles paramétriques", type="primary"):
            try:
                from modules.bonus_parametric import (
                    fit_parametric_models, compare_aic, plot_parametric_vs_km
                )
                with st.spinner("Ajustement des modèles paramétriques…"):
                    fitted = fit_parametric_models(df[time_col], df[event_col])

                st.markdown("#### Comparaison des AIC")
                aic_table = compare_aic(fitted, cph)

                def color_best(row):
                    if row.get("Rang") == 1:
                        return ["background-color: #DCFCE7"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    aic_table.style.apply(color_best, axis=1),
                    use_container_width=True, hide_index=True
                )

                st.markdown("#### Courbes comparées")
                fig_param = plot_parametric_vs_km(df[time_col], df[event_col], fitted)
                st.plotly_chart(fig_param, use_container_width=True)

                with st.expander("ℹ️ Interprétation"):
                    st.markdown("""
                    - **AIC plus faible = meilleur ajustement** (pénalise la complexité).
                    - **Weibull** : le plus flexible, adapté si le taux de risque est
                      croissant ou décroissant.
                    - **Exponentiel** : risque constant dans le temps (cas particulier de Weibull).
                    - **Log-Normal** : adapté si le taux de risque augmente puis diminue.
                    - Si les courbes paramétriques s'écartent du KM : le modèle paramétrique
                      n'est pas adapté aux données.
                    """)
            except Exception as e:
                st.error(f"Erreur modèles paramétriques : {e}")
