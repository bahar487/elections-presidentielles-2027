import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://bi_user:bi_password123@localhost:5432/elections"
DATA_DIR = "raw"

print("Connexion à PostgreSQL...")
engine = create_engine(DATABASE_URL)

print("\nChargement elections...")
df_elections = pd.read_csv(f"{DATA_DIR}/elections_presidentielles_2002_2022.csv")
df_elections.to_sql("elections", engine, if_exists="replace", index=False)
print(f"  {len(df_elections):,} lignes chargées")

print("\nChargement insee...")
df_insee = pd.read_csv(f"{DATA_DIR}/insee_sociodemographie_departements.csv")
df_insee.to_sql("insee", engine, if_exists="replace", index=False)
print(f"  {len(df_insee):,} lignes chargées")

print("\nChargement candidats_2027...")
df_cand = pd.read_csv(f"{DATA_DIR}/candidats_2027_hypotheses.csv")
df_cand.to_sql("candidats_2027", engine, if_exists="replace", index=False)
print(f"  {len(df_cand):,} lignes chargées")

print("\nVérification :")
with engine.connect() as conn:
    for table in ["elections", "insee", "candidats_2027"]:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        count = result.fetchone()[0]
        print(f"  {table} : {count:,} lignes")

print("\nImport terminé !")
