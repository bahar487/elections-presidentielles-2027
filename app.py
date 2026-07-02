import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Élections Présidentielles France", page_icon="🗳️", layout="wide")

@st.cache_data
def load_data():
    
    df_elections = pd.read_csv("raw/elections_presidentielles_2002_2022.csv")
    df_insee = pd.read_csv("raw/insee_sociodemographie_departements.csv")
    df_cand = pd.read_csv("raw/candidats_2027_hypotheses.csv")
    return df_elections, df_insee, df_cand

@st.cache_resource
def train_model(df_elections, df_insee):
    df_ml = df_elections[df_elections['tour'] == 1].merge(df_insee, on=['annee', 'dept_code'])
    features = ['taux_chomage', 'revenu_median', 'part_diplomes_sup', 'age_median',
                'taux_pauvrete', 'part_ouvriers', 'part_cadres', 'densite_hab_km2',
                'part_rurale', 'annee', 'parti_encoded']
    le = LabelEncoder()
    df_ml['parti_encoded'] = le.fit_transform(df_ml['parti'])
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df_ml[features], df_ml['pct_exprimes'])
    return model, le, features

df_elections, df_insee, df_cand = load_data()
model, le, features = train_model(df_elections, df_insee)

st.sidebar.title("🗳️ Élections 2027")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 Accueil", "📊 Vue générale", "🗺️ Analyse par département",
    "📈 Corrélations sociales", "🔥 Heatmap temporelle",
    "⚖️ Comparaison élections", "🔮 Simulateur 2027", "📋 Données brutes",
])
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Données simulées à des fins pédagogiques")

if page == "🏠 Accueil":
    st.title("🗳️ Analyse des Élections Présidentielles Françaises")
    st.subheader("France 2002–2022 & Prédiction 2027")
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Élections analysées", "5", "2002 → 2022")
    col2.metric("Départements", df_elections['dept_code'].nunique())
    col3.metric("Lignes électorales", f"{len(df_elections):,}")
    col4.metric("Candidats uniques", df_elections['candidat'].nunique())
    st.markdown("---")
    st.markdown("""
    ### À propos de ce projet
    Ce projet analyse les résultats des élections présidentielles françaises de 2002 à 2022,
    croise les données électorales avec des indicateurs socio-démographiques INSEE,
    et propose une prédiction des scores pour 2027 via un modèle de Machine Learning.

    ### Stack technique
    Python · PostgreSQL · Pandas · Scikit-learn · Streamlit · Plotly
    """)

elif page == "📊 Vue générale":
    st.title("📊 Vue générale")
    col1, col2 = st.columns(2)
    tour = col1.selectbox("Tour", [1, 2])
    seuil = col2.slider("Seuil score moyen minimum (%)", 1, 20, 5)
    t = df_elections[df_elections['tour'] == tour].groupby(['annee', 'candidat'])['pct_exprimes'].mean().reset_index()
    principaux = t.groupby('candidat')['pct_exprimes'].mean()
    principaux = principaux[principaux > seuil].index.tolist()
    t = t[t['candidat'].isin(principaux)]
    fig = px.line(t, x='annee', y='pct_exprimes', color='candidat', markers=True,
                  title=f"Scores par candidat — Tour {tour}")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    part = df_elections.groupby(['annee', 'tour'])['participation'].mean().reset_index()
    part['tour'] = part['tour'].astype(str)
    fig2 = px.line(part, x='annee', y='participation', color='tour', markers=True,
                   title="Taux de participation (%)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")
    partis = df_elections[df_elections['tour'] == 1].groupby('parti')['voix'].sum().sort_values(ascending=False).reset_index()
    fig3 = px.bar(partis, x='parti', y='voix', title="Total des voix par parti (2002–2022)")
    st.plotly_chart(fig3, use_container_width=True)

elif page == "🗺️ Analyse par département":
    st.title("🗺️ Analyse par département")
    col1, col2, col3 = st.columns(3)
    annee_sel = col1.selectbox("Année", sorted(df_elections['annee'].unique(), reverse=True))
    tour_sel = col2.selectbox("Tour", [1, 2])
    nb = col3.slider("Nombre de départements", 5, 50, 15)
    df_dept = df_elections[(df_elections['annee'] == annee_sel) & (df_elections['tour'] == tour_sel)]
    parti_sel = st.selectbox("Parti", sorted(df_dept['parti'].unique()))
    df_filtered = df_dept[df_dept['parti'] == parti_sel].sort_values('pct_exprimes', ascending=True).tail(nb)
    fig = px.bar(df_filtered, x='pct_exprimes', y='dept_nom', orientation='h',
                 title=f"Top {nb} départements — {parti_sel} ({annee_sel}, Tour {tour_sel})",
                 color='pct_exprimes', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    dept_sel = st.selectbox("Département", sorted(df_dept['dept_nom'].unique()))
    df_dept_sel = df_dept[df_dept['dept_nom'] == dept_sel].sort_values('pct_exprimes', ascending=False)
    fig2 = px.bar(df_dept_sel, x='candidat', y='pct_exprimes',
                  title=f"Scores dans {dept_sel} — {annee_sel}, Tour {tour_sel}", color='parti')
    st.plotly_chart(fig2, use_container_width=True)

elif page == "📈 Corrélations sociales":
    st.title("📈 Corrélations socio-démographiques")
    annee_corr = st.selectbox("Année", sorted(df_elections['annee'].unique(), reverse=True))
    elections_sel = df_elections[(df_elections['annee'] == annee_corr) & (df_elections['tour'] == 1)].groupby(['dept_code', 'parti'])['pct_exprimes'].mean().reset_index()
    scores_pivot = elections_sel.pivot(index='dept_code', columns='parti', values='pct_exprimes').reset_index()
    df_merged = scores_pivot.merge(df_insee[df_insee['annee'] == annee_corr], on='dept_code')
    col1, col2 = st.columns(2)
    partis_dispo = [p for p in ['RN', 'FN', 'NUPES', 'LREM/RE', 'LR', 'UMP', 'PS'] if p in df_merged.columns]
    parti_x = col1.selectbox("Parti", partis_dispo)
    indicateur_y = col2.selectbox("Indicateur social", ['taux_chomage', 'revenu_median', 'part_diplomes_sup', 'age_median', 'taux_pauvrete', 'part_ouvriers', 'part_cadres'])
    fig = px.scatter(df_merged, x=indicateur_y, y=parti_x, hover_data=['dept_nom'], trendline='ols',
                     title=f"{indicateur_y} vs Score {parti_x} ({annee_corr})")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    indicateurs = ['taux_chomage', 'revenu_median', 'part_diplomes_sup', 'age_median', 'taux_pauvrete', 'part_ouvriers', 'part_cadres']
    corr_data = df_merged[partis_dispo + indicateurs].corr().loc[indicateurs, partis_dispo]
    fig2 = px.imshow(corr_data, text_auto='.2f', color_continuous_scale='RdYlGn',
                     title="Matrice de corrélation complète", zmin=-1, zmax=1)
    st.plotly_chart(fig2, use_container_width=True)

elif page == "🔥 Heatmap temporelle":
    st.title("🔥 Heatmap — Scores par année et parti")
    tour_h = st.selectbox("Tour", [1, 2])
    pivot = df_elections[df_elections['tour'] == tour_h].groupby(['annee', 'parti'])['pct_exprimes'].mean().reset_index()
    pivot_table = pivot.pivot(index='parti', columns='annee', values='pct_exprimes').fillna(0)
    fig = px.imshow(pivot_table, text_auto='.1f', color_continuous_scale='Blues',
                    title=f"Score moyen par parti et par année — Tour {tour_h}")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

elif page == "⚖️ Comparaison élections":
    st.title("⚖️ Comparaison de deux élections")
    annees = sorted(df_elections['annee'].unique())
    col1, col2 = st.columns(2)
    annee1 = col1.selectbox("Élection 1", annees, index=0)
    annee2 = col2.selectbox("Élection 2", annees, index=len(annees)-1)
    tour_c = st.selectbox("Tour", [1, 2])
    df1 = df_elections[(df_elections['annee'] == annee1) & (df_elections['tour'] == tour_c)].groupby('candidat')['pct_exprimes'].mean().reset_index()
    df1['election'] = str(annee1)
    df2 = df_elections[(df_elections['annee'] == annee2) & (df_elections['tour'] == tour_c)].groupby('candidat')['pct_exprimes'].mean().reset_index()
    df2['election'] = str(annee2)
    df_comp = pd.concat([df1, df2])
    fig = px.bar(df_comp, x='candidat', y='pct_exprimes', color='election', barmode='group',
                 title=f"Comparaison {annee1} vs {annee2} — Tour {tour_c}")
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    for col, annee in [(col1, annee1), (col2, annee2)]:
        part = df_elections[(df_elections['annee'] == annee) & (df_elections['tour'] == tour_c)]['participation'].mean()
        col.metric(f"Participation {annee}", f"{part:.1f}%")

elif page == "🔮 Simulateur 2027":
    st.title("🔮 Simulateur — Présidentielle 2027")
    st.info("Ajustez les paramètres socio-économiques pour simuler les scores 2027")
    col1, col2, col3 = st.columns(3)
    chomage = col1.slider("Taux de chômage (%)", 5.0, 20.0, 9.0, 0.5)
    revenu = col2.slider("Revenu médian (€)", 15000, 35000, 22000, 500)
    diplomes = col3.slider("Diplômés sup (%)", 10.0, 50.0, 28.0, 1.0)
    col4, col5, col6 = st.columns(3)
    pauvrete = col4.slider("Taux de pauvreté (%)", 5.0, 30.0, 14.0, 0.5)
    ouvriers = col5.slider("Part ouvriers (%)", 5.0, 40.0, 22.0, 1.0)
    rurale = col6.slider("Part rurale (%)", 5.0, 80.0, 35.0, 1.0)
    candidats_2027 = [
        {"candidat": "Marine Le Pen", "parti": "RN"},
        {"candidat": "Successeur Macron", "parti": "LREM/RE"},
        {"candidat": "Successeur Mélenchon", "parti": "NUPES"},
        {"candidat": "Candidat LR", "parti": "LR"},
        {"candidat": "Candidat PS", "parti": "PS"},
    ]
    resultats = []
    for cand in candidats_2027:
        try:
            parti_enc = le.transform([cand['parti']])[0]
        except:
            parti_enc = 0
        X_sim = pd.DataFrame([{
            'taux_chomage': chomage, 'revenu_median': revenu, 'part_diplomes_sup': diplomes,
            'age_median': 41.0, 'taux_pauvrete': pauvrete, 'part_ouvriers': ouvriers,
            'part_cadres': 18.0, 'densite_hab_km2': 120.0, 'part_rurale': rurale,
            'annee': 2027, 'parti_encoded': parti_enc
        }])
        score = model.predict(X_sim)[0]
        resultats.append({"Candidat": cand['candidat'], "Parti": cand['parti'], "Score prédit (%)": round(score, 2)})
    df_res = pd.DataFrame(resultats).sort_values('Score prédit (%)', ascending=False)
    fig = px.bar(df_res, x='Candidat', y='Score prédit (%)', color='Candidat',
                 title="Scores prédits — Tour 1, 2027", text='Score prédit (%)')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_res, use_container_width=True)
    csv = df_res.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Exporter les résultats CSV", csv, "prediction_2027.csv", "text/csv")

elif page == "📋 Données brutes":
    st.title("📋 Données brutes")
    onglet = st.selectbox("Table", ["elections", "insee", "candidats_2027"])
    if onglet == "elections":
        col1, col2, col3 = st.columns(3)
        annee_f = col1.multiselect("Année", sorted(df_elections['annee'].unique()), default=sorted(df_elections['annee'].unique()))
        tour_f = col2.multiselect("Tour", [1, 2], default=[1, 2])
        parti_f = col3.multiselect("Parti", sorted(df_elections['parti'].unique()), default=sorted(df_elections['parti'].unique()))
        df_show = df_elections[(df_elections['annee'].isin(annee_f)) & (df_elections['tour'].isin(tour_f)) & (df_elections['parti'].isin(parti_f))]
        st.write(f"{len(df_show):,} lignes")
        st.dataframe(df_show, use_container_width=True)
        csv = df_show.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Exporter CSV", csv, "elections_filtered.csv", "text/csv")
    elif onglet == "insee":
        annee_i = st.multiselect("Année", sorted(df_insee['annee'].unique()), default=sorted(df_insee['annee'].unique()))
        df_show = df_insee[df_insee['annee'].isin(annee_i)]
        st.write(f"{len(df_show):,} lignes")
        st.dataframe(df_show, use_container_width=True)
        csv = df_show.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Exporter CSV", csv, "insee_filtered.csv", "text/csv")
    else:
        st.dataframe(df_cand, use_container_width=True)
        csv = df_cand.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Exporter CSV", csv, "candidats_2027.csv", "text/csv")
