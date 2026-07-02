# 🗳️ Élections Présidentielles France 2002-2027

Analyse électorale et prédiction 2027 — SPE Data & IA, Bachelor 3, Ynov Campus

> ⚠️ Données simulées à des fins pédagogiques

## Description

Ce projet analyse les résultats des élections présidentielles françaises de 2002 à 2022, croise les données électorales avec des indicateurs socio-démographiques INSEE, et propose une prédiction des scores pour 2027 via un modèle RandomForest.

## Structure du projet
## Stack technique

- **Python** : Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Base de données** : PostgreSQL (Docker)
- **Visualisation** : Streamlit, Plotly
- **ML** : RandomForestRegressor (MAE 1%, R² 0.97)

## Installation

### Prérequis
- Docker Desktop
- Python 3.12+
- WSL2 (Windows)

### Lancement

```bash
# 1. Démarrer PostgreSQL
docker compose up -d postgres

# 2. Générer et importer les données
python generate_datasets.py
python load_elections.py

# 3. Lancer l'application
streamlit run app.py
```

L'application est accessible sur `http://localhost:8501`

## Pages de l'application

| Page | Description |
|------|-------------|
| 🏠 Accueil | Présentation et métriques clés |
| 📊 Vue générale | Évolution des scores et participation |
| 🗺️ Analyse par département | Top départements par parti |
| 📈 Corrélations sociales | Indicateurs INSEE vs résultats |
| 🔥 Heatmap temporelle | Scores par année et parti |
| ⚖️ Comparaison élections | Deux élections côte à côte |
| 🔮 Simulateur 2027 | Prédiction interactive ML |
| 📋 Données brutes | Tables filtrables + export CSV |

## Modèle ML

- **Algorithme** : RandomForestRegressor (100 arbres)
- **Features** : taux_chomage, revenu_median, part_diplomes_sup, age_median, taux_pauvrete, part_ouvriers, part_cadres, densite_hab_km2, part_rurale, annee, parti_encoded
- **Résultats** : MAE = 1%, R² = 0.97
