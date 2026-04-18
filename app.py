"""
app.py — Application Streamlit d'Analyse de Survie
Réalisé par : Bouguessa Nour & Sbartai Sami
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Analyse de Survie",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563EB 60%, #7C3AED 100%);
    color: white; padding: 1.8rem 2rem; border-radius: 12px;
    margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(37,99,235,0.3);
}
.main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
.main-header p  { margin: 0.3rem 0 0; font-size: 0.9rem; opacity: 0.85; }
[data-testid="stExpander"] { border: 1px solid #E2E8F0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

from modules.data_loader import (
    load_csv, validate_dataframe, add_derived_variables,
    get_summary_metrics, apply_filters,
    ENCODING_OPTIONS, SEPARATOR_OPTIONS,
)
from modules.preprocessing import (
    detect_duplicates, remove_duplicates,
    missing_summary, apply_imputation, get_imputation_options,
)
from modules.descriptive import (
    get_numeric_stats, get_qualitative_stats, get_cramers_matrix,
    plot_cramers_heatmap, plot_bar_categorical, plot_stacked_bar,
    plot_scatter_matrix, plot_bivariate_survival,
    NUM_VARS, CAT_VARS,
)
from modules.kaplan_meier import (
    fit_kaplan_meier, plot_km_global, plot_km_stratified,
    get_survival_table, get_medians_table, run_logrank_tests,
)
from modules.nelson_aalen import (
    fit_nelson_aalen, plot_na_global, plot_na_vs_km, plot_na_stratified,
)
from modules.cox_model import (
    prepare_cox_data, fit_cox_model, get_cox_summary, get_forest_data,
    check_proportional_hazards, plot_schoenfeld_residuals,
    plot_martingale, plot_partial_effects, get_cox_metrics,
    CATEGORICAL_COLS,
)
from modules.prediction import (
    predict_survival_cox, predict_survival_baseline,
    plot_individual_survival, get_probability_table,
    compute_waterfall_contributions, predict_km_group,
    KEY_TIME_POINTS,
)
from modules.log_rank_tests import (
    run_all_weighted_tests, run_pairwise_logrank, interpret_logrank,
)
from utils.plots import (
    plot_histogram_kde, plot_boxplot, plot_correlation_matrix,
    plot_forest, plot_waterfall, plot_gauge,
    PALETTE, GROUP_COLORS,
)
from utils.stats_helpers import (
    chi2_test, compute_conditional_survival, get_variable_types,
    deviance_residuals,
)
from modules.bonus_tab import render_bonus_tab

for key, val in [("df_raw", None), ("df_clean", None),
                 ("time_col", "Time_to_Event"), ("event_col", "Event_Observed")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🩺 Analyse de Survie")

    st.markdown("### 📁 Chargement des données")
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
    col_enc, col_sep = st.columns(2)
    with col_enc:
        encoding_label = st.selectbox("Encodage", list(ENCODING_OPTIONS.keys()), key="enc")
    with col_sep:
        sep_label = st.selectbox("Séparateur", list(SEPARATOR_OPTIONS.keys()), key="sep")

    if st.button("📥 Charger les données", use_container_width=True):
        if uploaded_file is None:
            st.error("Aucun fichier sélectionné.")
        else:
            with st.spinner("Chargement…"):
                try:
                    df_loaded = load_csv(
                        uploaded_file.read(),
                        ENCODING_OPTIONS[encoding_label],
                        SEPARATOR_OPTIONS[sep_label],
                    )
                    df_loaded = add_derived_variables(df_loaded)
                    st.session_state["df_raw"] = df_loaded
                    st.session_state["df_clean"] = df_loaded.copy()
                    for k in ["cph_model", "df_cox"]:
                        st.session_state.pop(k, None)
                    st.success(f"✅ {len(df_loaded):,} lignes × {df_loaded.shape[1]} colonnes.")
                except Exception as e:
                    st.error(f"Erreur : {e}")

    st.markdown("### 🎯 Variables")
    if st.session_state["df_clean"] is not None:
        cols_avail = st.session_state["df_clean"].columns.tolist()
        st.session_state["time_col"] = st.selectbox(
            "Variable de temps", cols_avail,
            index=cols_avail.index("Time_to_Event") if "Time_to_Event" in cols_avail else 0,
        )
        st.session_state["event_col"] = st.selectbox(
            "Variable d'événement", cols_avail,
            index=cols_avail.index("Event_Observed") if "Event_Observed" in cols_avail else 0,
        )

    st.markdown("### 🔽 Filtres interactifs")
    filters = {}
    if st.session_state["df_clean"] is not None:
        df_ref = st.session_state["df_clean"]
        if "Age" in df_ref.columns:
            age_min, age_max = int(df_ref["Age"].min()), int(df_ref["Age"].max())
            filters["Age"] = st.slider("Âge", age_min, age_max, (age_min, age_max))
        for cat in ["Sex", "Smoker", "Treatment", "Physical_Activity", "Tranche_BMI"]:
            if cat in df_ref.columns:
                opts = sorted(df_ref[cat].dropna().unique().tolist(), key=str)
                sel = st.multiselect(cat, opts, default=opts, key=f"filt_{cat}")
                if sel:
                    filters[cat] = sel
        if st.button("🔄 Réinitialiser les filtres", use_container_width=True):
            for k in [k for k in st.session_state if k.startswith("filt_")]:
                del st.session_state[k]
            st.rerun()
        df_filtered = apply_filters(df_ref, filters)
        st.info(f"**{len(df_filtered):,}** patients sélectionnés sur **{len(df_ref):,}**")
    else:
        df_filtered = pd.DataFrame()

    st.markdown("### ℹ️ Informations")
    st.caption("Version 1.0\nBouguessa Nour & Sbartai Sami\nAvril 2026")

# ══════════════════════════════════════════════════════════════════════════════
# EN-TÊTE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🩺 Application d'Analyse de Survie</h1>
  <p>Kaplan-Meier · Nelson-Aalen · Modèle de Cox · Prédiction individuelle — Niveau Master/Thèse</p>
</div>
""", unsafe_allow_html=True)

if st.session_state["df_clean"] is None:
    st.info("👈 Commencez par charger un fichier CSV dans la barre latérale.")
    st.markdown("""
    **Format attendu du CSV :**
    | Colonne | Type | Description |
    |---------|------|-------------|
    | Age | Numérique | Âge du patient |
    | Sex | Catégoriel | Male / Female |
    | BMI | Numérique | Indice de masse corporelle |
    | Smoker | Binaire | 0 = Non-fumeur, 1 = Fumeur |
    | Comorbidities | Numérique | Nombre de comorbidités |
    | Treatment | Catégoriel | Standard / Experimental |
    | Physical_Activity | Catégoriel | Low / Moderate / High |
    | Time_to_Event | Numérique | Temps jusqu'à l'événement (mois) |
    | Event_Observed | Binaire | 1 = événement, 0 = censuré |
    """)
    st.stop()

df = df_filtered
time_col = st.session_state["time_col"]
event_col = st.session_state["event_col"]

if len(df) < 5:
    st.error("⛔ Moins de 5 patients après filtrage.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Visualisation des données",
    "🧹 Données manquantes",
    "📈 Statistiques descriptives",
    "🎨 Représentations graphiques",
    "📉 Survie & Courbes KM",
    "🔮 Prédiction individuelle",
    "🧬 Modèle de Cox",
    "🔬 Tests de comparaison",
    "⭐ Analyses avancées",
])

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 1
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("📊 Aperçu du jeu de données")
    metrics = get_summary_metrics(df, time_col, event_col)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Patients", f"{metrics['n_total']:,}")
    c2.metric("💀 Événements", f"{metrics['n_events']:,}", f"{metrics['pct_events']}%")
    c3.metric("⏸ Censures", f"{metrics['n_censored']:,}", f"{metrics['pct_censored']}%")
    c4.metric("⏱ Médiane suivi", f"{metrics['median_time']} mois")
    c5.metric("📏 Min / Max", f"{metrics['min_time']} / {metrics['max_time']}")
    st.divider()

    n_rows = st.slider("Nombre de lignes à afficher", 5, min(100, len(df)), 10)
    st.dataframe(df.head(n_rows).style.highlight_null(color="#FEE2E2"),
                 use_container_width=True, height=350)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Télécharger les données filtrées (CSV)",
                       csv_buf.getvalue(), "donnees_filtrees.csv", "text/csv")
    st.divider()

    st.subheader("🔍 Vérification des doublons")
    try:
        dup_info = detect_duplicates(df, event_col)
        if dup_info["n_total_duplicates"] > 0:
            st.warning(f"⚠️ **{dup_info['n_total_duplicates']}** lignes dupliquées.")
            if st.button("🗑️ Supprimer les doublons"):
                st.session_state["df_clean"] = remove_duplicates(
                    st.session_state["df_clean"])
                st.success("Doublons supprimés.")
                st.rerun()
        else:
            st.success("✅ Aucun doublon détecté.")
    except Exception as e:
        st.error(f"Erreur : {e}")
    st.divider()

    st.subheader("📋 Types de variables")
    try:
        st.dataframe(get_variable_types(df), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Erreur : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 2
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("🧹 Diagnostic des valeurs manquantes")
    try:
        miss = missing_summary(df)
        total_miss = miss["N manquants"].sum()
        if total_miss == 0:
            st.success("✅ Aucune valeur manquante.")
        else:
            st.warning(f"⚠️ **{total_miss}** valeurs manquantes au total.")
            st.dataframe(miss, use_container_width=True, hide_index=True)
            miss_nz = miss[miss["% manquants"] > 0]
            if not miss_nz.empty:
                fig_miss = go.Figure(go.Bar(
                    x=miss_nz["% manquants"], y=miss_nz["Variable"],
                    orientation="h", marker_color=PALETTE["danger"],
                    text=miss_nz["% manquants"].round(1).astype(str) + "%",
                    textposition="outside",
                ))
                fig_miss.update_layout(
                    title="% de valeurs manquantes",
                    template="plotly_white",
                    height=max(300, 40 * len(miss_nz) + 100))
                st.plotly_chart(fig_miss, use_container_width=True)

            st.divider()
            st.subheader("⚙️ Stratégies de traitement")
            strategies = {}
            for col in df.columns:
                n_miss = df[col].isna().sum()
                if n_miss == 0:
                    continue
                if col in [time_col, event_col]:
                    st.warning(f"**{col}** ({n_miss} manquants) : suppression obligatoire.")
                    continue
                strategies[col] = st.selectbox(
                    f"**{col}** ({n_miss} manquants)",
                    get_imputation_options(str(df[col].dtype)),
                    key=f"imp_{col}")

            if st.button("✅ Appliquer le traitement", type="primary"):
                with st.spinner("Application…"):
                    df_new, messages = apply_imputation(
                        st.session_state["df_clean"], strategies, time_col, event_col)
                    st.session_state["df_clean"] = df_new
                    for msg in messages:
                        st.write(msg)
                    st.success("Traitement appliqué.")
                    buf = io.StringIO()
                    df_new.to_csv(buf, index=False)
                    st.download_button("⬇️ Dataset nettoyé (CSV)",
                                       buf.getvalue(), "data_clean.csv", "text/csv")
    except Exception as e:
        st.error(f"Erreur : {e}")

    with st.expander("📚 Note méthodologique"):
        st.markdown("""
        **MCAR** : Toute méthode valide.
        **MAR** : Imputation multiple recommandée.
        **MNAR** : Situation problématique — censure informative possible.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 3
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📈 Statistiques descriptives")

    st.markdown("#### Variables numériques")
    try:
        num_stats = get_numeric_stats(df)
        if not num_stats.empty:
            def color_pval(val):
                try:
                    return ("color: #DC2626; font-weight:600"
                            if float(val) < 0.05 else "color: #6B7280")
                except Exception:
                    return ""
            st.dataframe(
                num_stats.style.map(color_pval, subset=["p-valeur"]),
                use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Erreur : {e}")

    st.divider()
    st.markdown("#### Variables qualitatives")
    try:
        qual_stats = get_qualitative_stats(df)
        cols_q = st.columns(min(max(len(qual_stats), 1), 3))
        for idx, (var, tbl) in enumerate(qual_stats.items()):
            with cols_q[idx % 3]:
                st.markdown(f"**{var}**")
                st.dataframe(tbl, hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur : {e}")

    st.divider()
    st.markdown("#### V de Cramér — Associations catégorielles")
    try:
        cramer_mat = get_cramers_matrix(df)
        if not cramer_mat.empty:
            st.plotly_chart(plot_cramers_heatmap(cramer_mat), use_container_width=True)
    except Exception as e:
        st.error(f"Erreur : {e}")

    st.divider()
    st.markdown("#### Tableau croisé interactif")
    cat_cols_avail = [c for c in CAT_VARS + ["Tranche_Age", "Tranche_BMI"]
                      if c in df.columns]
    if len(cat_cols_avail) >= 2:
        cc1, cc2 = st.columns(2)
        with cc1:
            var_row = st.selectbox("Variable ligne", cat_cols_avail, key="ct_row")
        with cc2:
            var_col_ct = st.selectbox(
                "Variable colonne",
                [c for c in cat_cols_avail if c != var_row], key="ct_col")
        try:
            ct_res = chi2_test(df, var_row, var_col_ct)
            st.dataframe(ct_res["crosstab"], use_container_width=True)
            sig_str = "✅ Significatif" if ct_res["significant"] else "❌ Non significatif"
            st.info(f"**Chi²** = {ct_res['chi2']}, p = {ct_res['pvalue']}, "
                    f"ddl = {ct_res['dof']} — {sig_str}")
        except Exception as e:
            st.error(f"Erreur : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 4
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("🎨 Représentations graphiques interactives")

    num_cols_avail = [c for c in NUM_VARS if c in df.columns]
    cat_cols_g = [c for c in CAT_VARS if c in df.columns]

    st.markdown("#### Distribution des variables numériques")
    g1c1, g1c2 = st.columns(2)
    with g1c1:
        var_hist = st.selectbox("Variable", num_cols_avail, key="var_hist")
    with g1c2:
        grp_hist = st.selectbox("Grouper par", ["Aucun"] + cat_cols_g, key="grp_hist")
    try:
        grp_val = None if grp_hist == "Aucun" else grp_hist
        st.plotly_chart(plot_histogram_kde(df, var_hist, group_by=grp_val),
                        use_container_width=True)
        if grp_val:
            st.plotly_chart(plot_boxplot(df, var_hist, grp_val),
                            use_container_width=True)
    except Exception as e:
        st.error(f"Erreur : {e}")

    st.divider()
    st.markdown("#### Variables catégorielles")
    if cat_cols_g:
        g2c1, g2c2 = st.columns(2)
        with g2c1:
            var_bar = st.selectbox("Variable principale", cat_cols_g, key="var_bar")
            try:
                st.plotly_chart(plot_bar_categorical(df, var_bar),
                                use_container_width=True)
            except Exception as e:
                st.error(f"{e}")
        with g2c2:
            if len(cat_cols_g) >= 2:
                var_sx = st.selectbox("Variable X (empilé)", cat_cols_g, key="sx")
                var_sg = st.selectbox(
                    "Variable groupe",
                    [c for c in cat_cols_g if c != var_sx], key="sg")
                try:
                    st.plotly_chart(plot_stacked_bar(df, var_sx, var_sg),
                                    use_container_width=True)
                except Exception as e:
                    st.error(f"{e}")

    st.divider()
    st.markdown("#### Matrice de corrélation de Pearson")
    if len(num_cols_avail) >= 2:
        try:
            st.plotly_chart(plot_correlation_matrix(df, num_cols_avail),
                            use_container_width=True)
        except Exception as e:
            st.error(f"Erreur : {e}")

    st.divider()
    st.markdown("#### Scatter matrix (pairplot)")
    color_sc = st.selectbox("Colorier par", ["Aucun"] + cat_cols_g, key="color_sc")
    try:
        st.plotly_chart(
            plot_scatter_matrix(df, color_by=None if color_sc == "Aucun" else color_sc),
            use_container_width=True)
    except Exception as e:
        st.error(f"Erreur : {e}")

    st.divider()
    st.markdown("#### Analyse bivariée : Survie × Variable")
    if cat_cols_g:
        var_biv = st.selectbox("Variable explicative", cat_cols_g, key="biv")
        try:
            st.plotly_chart(
                plot_bivariate_survival(df, time_col, event_col, var_biv),
                use_container_width=True)
        except Exception as e:
            st.error(f"Erreur : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 5
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("📉 Analyse de survie — Kaplan-Meier & Nelson-Aalen")

    st.markdown("#### Courbe de survie globale")
    kmf_global = None
    try:
        with st.spinner("Ajustement KM…"):
            kmf_global = fit_kaplan_meier(df[time_col], df[event_col], "Survie globale")
        st.plotly_chart(plot_km_global(kmf_global), use_container_width=True)
        med = float(kmf_global.median_survival_time_)
        st.metric("Médiane de survie",
                  f"{med:.2f} mois" if np.isfinite(med) else "Non atteinte")
        with st.expander("📋 Tableau de survie complet"):
            surv_tbl = get_survival_table(kmf_global)
            st.dataframe(surv_tbl, use_container_width=True, hide_index=True)
            buf_sv = io.StringIO()
            surv_tbl.to_csv(buf_sv, index=False)
            st.download_button("⬇️ Télécharger", buf_sv.getvalue(), "survival_table.csv")
    except Exception as e:
        st.error(f"Erreur KM global : {e}")

    st.divider()
    st.markdown("#### Survie stratifiée par groupe")
    strat_options = [c for c in ["Sex", "Smoker", "Treatment", "Tranche_Age",
                                  "Tranche_BMI", "Physical_Activity"] if c in df.columns]
    kmf_dict = {}
    strat_var_km = strat_options[0] if strat_options else None

    if strat_options:
        strat_var_km = st.selectbox("Variable de stratification",
                                     strat_options, key="strat_km")
        try:
            with st.spinner("Stratification KM…"):
                fig_strat, kmf_dict = plot_km_stratified(
                    df, time_col, event_col, strat_var_km)
            st.plotly_chart(fig_strat, use_container_width=True)
            st.dataframe(get_medians_table(kmf_dict),
                         use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Erreur stratification : {e}")

        st.markdown("#### Test log-rank")
        try:
            lr_res = run_logrank_tests(df, time_col, event_col, strat_var_km)
            if lr_res.get("global"):
                g = lr_res["global"]
                st.info(interpret_logrank(g["p_value"],
                                          list(kmf_dict.keys()), strat_var_km))
                c1, c2, c3 = st.columns(3)
                c1.metric("χ²", g["chi2"])
                c2.metric("ddl", g["ddl"])
                c3.metric("p-valeur", g["p_value"])
            if "pairwise" in lr_res and not lr_res["pairwise"].empty:
                with st.expander("📊 Comparaisons deux à deux (Bonferroni)"):
                    st.dataframe(lr_res["pairwise"],
                                 use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Erreur log-rank : {e}")

    st.divider()
    st.markdown("#### Survie conditionnelle")
    sc1, sc2 = st.columns(2)
    with sc1:
        t0_cond = st.number_input("t₀ (mois)", 0.0, float(df[time_col].max()), 12.0)
    with sc2:
        t_target = st.number_input("t cible (mois)", 0.0,
                                    float(df[time_col].max()), 36.0)
    if t_target > t0_cond and kmf_global is not None:
        try:
            cond = compute_conditional_survival(kmf_global, t0_cond, t_target)
            st.metric(f"P(T > {t_target:.0f} | T > {t0_cond:.0f})",
                      f"{cond:.4f}" if not np.isnan(cond) else "N/A")
        except Exception:
            pass
    with st.expander("📚 Formule"):
        st.latex(r"P(T > t \mid T > t_0) = \frac{S(t)}{S(t_0)}")

    st.divider()
    st.markdown("#### Risque cumulé — Nelson-Aalen")
    try:
        with st.spinner("Nelson-Aalen…"):
            naf_global = fit_nelson_aalen(df[time_col], df[event_col])
        na_tabs = st.tabs(["H(t) global", "H(t) vs −log S(t)", "Stratifié"])
        with na_tabs[0]:
            st.plotly_chart(plot_na_global(naf_global), use_container_width=True)
        with na_tabs[1]:
            if kmf_global is not None:
                st.plotly_chart(plot_na_vs_km(naf_global, kmf_global),
                                use_container_width=True)
        with na_tabs[2]:
            if strat_options:
                strat_na = st.selectbox("Stratification NA",
                                         strat_options, key="strat_na")
                st.plotly_chart(
                    plot_na_stratified(df, time_col, event_col, strat_na),
                    use_container_width=True)
    except Exception as e:
        st.error(f"Erreur Nelson-Aalen : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 6
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("🔮 Prédiction individuelle de survie")
    st.markdown("#### 🩺 Profil du patient")

    pf1, pf2, pf3 = st.columns(3)
    with pf1:
        p_age = st.number_input("Âge", 18, 100, 60, key="pred_age")
        p_sex = st.radio("Sexe", ["Male", "Female"], key="pred_sex")
        p_smoker = st.radio("Fumeur", ["Non (0)", "Oui (1)"], key="pred_smoker")
    with pf2:
        p_bmi = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.5, key="pred_bmi")
        p_treatment = st.selectbox("Traitement",
                                    ["Standard", "Experimental"], key="pred_trt")
    with pf3:
        p_activity = st.selectbox("Activité physique",
                                   ["Low", "Moderate", "High"], key="pred_act")
        p_comorbidities = st.number_input("Comorbidités", 0, 10, 1, key="pred_comorb")

    patient_profile = {
        "Age": p_age, "Sex": p_sex,
        "Smoker": 1 if "Oui" in p_smoker else 0,
        "BMI": p_bmi, "Treatment": p_treatment,
        "Physical_Activity": p_activity, "Comorbidities": p_comorbidities,
    }

    if st.button("🔍 Calculer la prédiction", type="primary", key="btn_predict"):
        with st.spinner("Calcul en cours…"):
            try:
                covariates_pred = [c for c in ["Age", "Sex", "Smoker", "Treatment",
                                                "Physical_Activity"] if c in df.columns]
                df_cox_pred = prepare_cox_data(df, time_col, event_col, covariates_pred)

                st.write(f"Lignes disponibles : {len(df_cox_pred)}")
                st.write(f"Colonnes : {list(df_cox_pred.columns)}")

                if len(df_cox_pred) < 5:
                    st.error(f"Effectif insuffisant : {len(df_cox_pred)} lignes.")
                    st.stop()

                df_hash_pred = str(pd.util.hash_pandas_object(df_cox_pred).sum())
                cph_pred = fit_cox_model(df_hash_pred, df_cox_pred, time_col, event_col)

                sf_ind, risk_score = predict_survival_cox(
                    cph_pred, patient_profile, df_cox_pred)
                sf_base = predict_survival_baseline(cph_pred)
                km_proba = predict_km_group(df, time_col, event_col, patient_profile)

                st.plotly_chart(
                    plot_individual_survival(sf_ind, sf_base, "Patient"),
                    use_container_width=True)

                st.markdown("#### 🎯 Niveau de risque")
                col_g1, col_g2 = st.columns([1, 2])
                with col_g1:
                    st.plotly_chart(plot_gauge(risk_score), use_container_width=True)
                with col_g2:
                    st.metric("Score de risque relatif", f"{risk_score:.4f}")
                    level = ("🟢 Faible" if risk_score < 1.5
                             else ("🟠 Modéré" if risk_score < 2.5 else "🔴 Élevé"))
                    st.markdown(f"**Niveau : {level}**")
                    st.caption("Score = exp(β·X) — valeur 1 = patient médian.")

                st.markdown("#### 📋 Probabilités de survie")
                col_km, col_cox = st.columns(2)
                with col_km:
                    st.markdown("**Via Kaplan-Meier (groupe)**")
                    st.dataframe(km_proba, hide_index=True, use_container_width=True)
                with col_cox:
                    st.markdown("**Via modèle de Cox**")
                    prob_table = get_probability_table(sf_ind)
                    key_probs = prob_table[
                        prob_table["Temps (mois)"].isin(KEY_TIME_POINTS)]
                    st.dataframe(key_probs, hide_index=True, use_container_width=True)

                with st.expander("📊 Tableau complet S(t) tous les 6 mois"):
                    st.dataframe(prob_table, use_container_width=True, hide_index=True)
                    buf_pred = io.StringIO()
                    prob_table.to_csv(buf_pred, index=False)
                    st.download_button("⬇️ Télécharger (CSV)",
                                       buf_pred.getvalue(), "prediction_survie.csv")

                st.markdown("#### 🌊 Contributions au score de risque (SHAP-like)")
                try:
                    contrib = compute_waterfall_contributions(
                        cph_pred, patient_profile, df_cox_pred)
                    if contrib:
                        st.plotly_chart(plot_waterfall(contrib),
                                        use_container_width=True)
                except Exception as ew:
                    st.warning(f"Contributions non disponibles : {ew}")

            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")
                st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 7
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("🧬 Modèle de Cox à risques proportionnels")

    st.markdown("#### ⚙️ Configuration du modèle")
    all_cov = [c for c in ["Age", "Sex", "Smoker", "BMI", "Treatment",
                            "Physical_Activity", "Comorbidities"] if c in df.columns]
    default_cov = [c for c in ["Age", "Sex", "Smoker", "Treatment",
                                "Physical_Activity"] if c in df.columns]
    selected_cov = st.multiselect("Covariables à inclure", all_cov,
                                   default=default_cov, key="cox_cov")
    ties_method = st.selectbox("Méthode ex-aequo", ["breslow", "efron"], key="ties")

    if st.button("🚀 Ajuster le modèle de Cox", type="primary", key="btn_cox"):
        if not selected_cov:
            st.error("Sélectionnez au moins une covariable.")
        else:
            with st.spinner("Ajustement…"):
                try:
                    df_cox_new = prepare_cox_data(
                        df, time_col, event_col, selected_cov)
                    h = str(pd.util.hash_pandas_object(df_cox_new).sum()) + ties_method
                    cph_new = fit_cox_model(
                        h, df_cox_new, time_col, event_col, ties=ties_method)
                    st.session_state["cph_model"] = cph_new
                    st.session_state["df_cox"] = df_cox_new
                    st.success("✅ Modèle ajusté avec succès.")
                except Exception as e:
                    st.error(f"Erreur : {e}")

    if "cph_model" not in st.session_state:
        try:
            df_cox_auto = prepare_cox_data(df, time_col, event_col, default_cov)
            h_auto = str(pd.util.hash_pandas_object(df_cox_auto).sum()) + "breslow"
            cph_auto = fit_cox_model(h_auto, df_cox_auto, time_col, event_col)
            st.session_state["cph_model"] = cph_auto
            st.session_state["df_cox"] = df_cox_auto
        except Exception as e:
            st.error(f"Erreur auto-ajustement : {e}")
            st.stop()

    cph = st.session_state["cph_model"]
    df_cox = st.session_state["df_cox"]

    st.markdown("#### 📊 Résultats du modèle")
    try:
        cm = get_cox_metrics(cph)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("C-index", cm["concordance"])
        mc2.metric("AIC partiel", cm["AIC"])
        mc3.metric("Log-vraisemblance", cm["log_likelihood"])

        cox_summary = get_cox_summary(cph)

        def highlight_sig(row):
            try:
                p = float(str(row.get("p-valeur", "1")).replace("<", ""))
                if p < 0.05:
                    return ["background-color: #DCFCE7"] * len(row)
            except Exception:
                pass
            return [""] * len(row)

        st.dataframe(cox_summary.style.apply(highlight_sig, axis=1),
                     use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Erreur résultats Cox : {e}")

    st.divider()
    st.markdown("#### 🌲 Forest Plot — Hazard Ratios")
    try:
        st.plotly_chart(plot_forest(get_forest_data(cph)), use_container_width=True)
    except Exception as e:
        st.error(f"Erreur forest plot : {e}")

    st.divider()
    st.markdown("#### 🔬 Vérification hypothèse PH")
    try:
        ph = check_proportional_hazards(cph, df_cox)
        if "table" in ph and not ph["table"].empty:
            st.dataframe(ph["table"], use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Test PH non disponible : {e}")

    with st.expander("📈 Résidus de Schoenfeld"):
        for col_var in cph.params_.index:
            try:
                fig_sch = plot_schoenfeld_residuals(cph, df_cox, col_var)
                if fig_sch.data:
                    st.plotly_chart(fig_sch, use_container_width=True)
            except Exception:
                pass

    st.divider()
    st.markdown("#### 📐 Survie ajustée — Effet partiel")
    try:
        cox_binary = [c for c in df_cox.columns
                      if c not in [time_col, event_col]
                      and df_cox[c].nunique() <= 5]
        if cox_binary:
            var_partial = st.selectbox("Variable d'intérêt",
                                        cox_binary, key="partial")
            fig_partial = plot_partial_effects(cph, df_cox, var_partial)
            if fig_partial.data:
                st.plotly_chart(fig_partial, use_container_width=True)
    except Exception as e:
        st.warning(f"Effets partiels non disponibles : {e}")

    st.divider()
    st.markdown("#### 🔭 Résidus du modèle")
    res_tabs = st.tabs(["Martingale", "Déviance"])
    with res_tabs[0]:
        try:
            num_cov = [c for c in df_cox.columns
                       if c not in [time_col, event_col]
                       and df_cox[c].dtype in [np.float64, np.int64]]
            if num_cov:
                var_mart = st.selectbox("Variable", num_cov, key="mart_var")
                fig_mart = plot_martingale(cph, df_cox, var_mart)
                if fig_mart.data:
                    st.plotly_chart(fig_mart, use_container_width=True)
        except Exception as e:
            st.warning(f"Résidus Martingale : {e}")

    with res_tabs[1]:
        try:
            dev = deviance_residuals(cph, df_cox)
            fig_dev = go.Figure(go.Scatter(
                y=dev.values, mode="markers",
                marker=dict(color=PALETTE["primary"], size=4, opacity=0.5)))
            fig_dev.add_hline(y=0, line_dash="dash", line_color=PALETTE["neutral"])
            fig_dev.update_layout(title="Résidus de Déviance",
                                  xaxis_title="Index", yaxis_title="Résidu",
                                  template="plotly_white")
            st.plotly_chart(fig_dev, use_container_width=True)
            top10 = dev.abs().nlargest(10)
            st.markdown("**10 patients avec résidus extrêmes :**")
            out = df_cox.iloc[top10.index].copy()
            out["Résidu_Déviance"] = dev.values[top10.index]
            st.dataframe(out.round(4), use_container_width=True)
        except Exception as e:
            st.warning(f"Résidus Déviance : {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 8
# ─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("🔬 Tests de comparaison des courbes de survie")

    strat_t8 = [c for c in ["Sex", "Smoker", "Treatment", "Tranche_Age",
                              "Tranche_BMI", "Physical_Activity"] if c in df.columns]
    if not strat_t8:
        st.warning("Aucune variable de stratification disponible.")
        st.stop()

    st.markdown("#### Test de Mantel-Haenszel (log-rank)")
    var_lr = st.selectbox("Variable de groupe", strat_t8, key="lr_var")
    groups_lr = df[var_lr].dropna().unique()
    n_groups = len(groups_lr)

    try:
        lr_res = run_logrank_tests(df, time_col, event_col, var_lr)
        if lr_res.get("global"):
            g = lr_res["global"]
            st.info(interpret_logrank(g["p_value"], list(groups_lr), var_lr))
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("χ²", g["chi2"])
            lc2.metric("ddl", g["ddl"])
            lc3.metric("p-valeur", g["p_value"])
        if "pairwise" in lr_res and not lr_res["pairwise"].empty:
            st.markdown("**Comparaisons deux à deux (Bonferroni) :**")
            st.dataframe(lr_res["pairwise"], use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Erreur log-rank : {e}")

    st.divider()
    st.markdown("#### Tests alternatifs pondérés")
    if n_groups == 2:
        try:
            st.dataframe(
                run_all_weighted_tests(df, time_col, event_col, var_lr),
                use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Erreur tests pondérés : {e}")
    else:
        st.info(f"ℹ️ Tests pondérés pour 2 groupes uniquement "
                f"('{var_lr}' en a {n_groups}).")

    with st.expander("📚 Quand utiliser chaque test ?"):
        st.markdown("""
        | Test | Pondération | Usage |
        |------|-------------|-------|
        | **Log-rank** | Uniforme | Différences constantes |
        | **Wilcoxon** | nᵢ | Différences en début |
        | **Tarone-Ware** | √nᵢ | Compromis |
        | **FH (ρ=1)** | S(t) | Différences en début |
        | **FH (γ=1)** | 1−S(t) | Différences en fin |
        """)

    st.divider()
    st.markdown("#### Comparaison multi-critères")
    multi_vars = st.multiselect(
        "Variables de stratification", strat_t8,
        default=strat_t8[:2] if len(strat_t8) >= 2 else strat_t8,
        key="multi_strat")
    if len(multi_vars) >= 2:
        ncols = 2
        nrows = (len(multi_vars) + 1) // 2
        fig_multi = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=[f"KM par {v}" for v in multi_vars])
        for idx, mv in enumerate(multi_vars):
            row_i, col_i = idx // ncols + 1, idx % ncols + 1
            for gi, grp in enumerate(sorted(df[mv].dropna().unique(), key=str)):
                sub = df[df[mv] == grp]
                if len(sub) < 5:
                    continue
                try:
                    kmf_mv = fit_kaplan_meier(
                        sub[time_col], sub[event_col], str(grp))
                    sf = kmf_mv.survival_function_
                    fig_multi.add_trace(
                        go.Scatter(
                            x=sf.index, y=sf.iloc[:, 0], mode="lines",
                            name=f"{mv}={grp}",
                            line=dict(
                                color=GROUP_COLORS[gi % len(GROUP_COLORS)],
                                width=2),
                            showlegend=(idx == 0)),
                        row=row_i, col=col_i)
                except Exception:
                    pass
        fig_multi.update_layout(
            height=380 * nrows, template="plotly_white",
            title="Comparaison multi-critères — Kaplan-Meier")
        st.plotly_chart(fig_multi, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 9
# ─────────────────────────────────────────────────────────────────────────────
with tabs[8]:
    cph_for_bonus = st.session_state.get("cph_model", None)
    render_bonus_tab(df, time_col, event_col, cph=cph_for_bonus)

st.markdown("---")
st.caption("🩺 Application d'Analyse de Survie | Bouguessa Nour & Sbartai Sami | 2026")
