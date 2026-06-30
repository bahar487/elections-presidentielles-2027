"""
Génération des datasets simulés réalistes pour les élections présidentielles françaises.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

DEPARTEMENTS = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardèche", "08": "Ardennes",
    "09": "Ariège", "10": "Aube", "11": "Aude", "12": "Aveyron",
    "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal", "16": "Charente",
    "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze", "21": "Côte-d'Or",
    "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne", "25": "Doubs",
    "26": "Drôme", "27": "Eure", "28": "Eure-et-Loir", "29": "Finistère",
    "30": "Gard", "31": "Haute-Garonne", "32": "Gers", "33": "Gironde",
    "34": "Hérault", "35": "Ille-et-Vilaine", "36": "Indre", "37": "Indre-et-Loire",
    "38": "Isère", "39": "Jura", "40": "Landes", "41": "Loir-et-Cher",
    "42": "Loire", "43": "Haute-Loire", "44": "Loire-Atlantique", "45": "Loiret",
    "46": "Lot", "47": "Lot-et-Garonne", "48": "Lozère", "49": "Maine-et-Loire",
    "50": "Manche", "51": "Marne", "52": "Haute-Marne", "53": "Mayenne",
    "54": "Meurthe-et-Moselle", "55": "Meuse", "56": "Morbihan", "57": "Moselle",
    "58": "Nièvre", "59": "Nord", "60": "Oise", "61": "Orne",
    "62": "Pas-de-Calais", "63": "Puy-de-Dôme", "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales", "67": "Bas-Rhin",
    "68": "Haut-Rhin", "69": "Rhône", "70": "Haute-Saône", "71": "Saône-et-Loire",
    "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie", "75": "Paris",
    "76": "Seine-Maritime", "77": "Seine-et-Marne", "78": "Yvelines",
    "79": "Deux-Sèvres", "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne",
    "83": "Var", "84": "Vaucluse", "85": "Vendée", "86": "Vienne",
    "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne", "90": "Territoire de Belfort",
    "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise",
}

PROFILS_REGIONAUX = {
    "75": (0.05, 0.02, -0.05, -0.03),
    "93": (0.08, -0.03, -0.02, 0.05),
    "92": (-0.02, 0.06, -0.02, -0.02),
    "13": (0.00, -0.02, 0.06, 0.03),
    "62": (-0.02, -0.03, 0.09, 0.04),
    "59": (-0.01, -0.02, 0.07, 0.04),
    "83": (-0.02, 0.03, 0.07, 0.01),
    "67": (-0.01, 0.04, 0.02, -0.01),
    "69": (0.01, 0.02, 0.02, -0.01),
    "31": (0.04, -0.01, 0.01, 0.00),
}

ELECTIONS_HISTORIQUE = {
    2002: {
        "tour1": {
            "Jacques Chirac":      {"parti": "RPR/UMP",  "score_nat": 0.198, "couleur": "#1E90FF"},
            "Jean-Marie Le Pen":   {"parti": "FN",       "score_nat": 0.169, "couleur": "#1a1a2e"},
            "Lionel Jospin":       {"parti": "PS",       "score_nat": 0.161, "couleur": "#FF69B4"},
            "François Bayrou":     {"parti": "UDF",      "score_nat": 0.069, "couleur": "#FF8C00"},
            "Arlette Laguiller":   {"parti": "LO",       "score_nat": 0.056, "couleur": "#CC0000"},
            "Jean-Pierre Chevènement": {"parti": "MDC", "score_nat": 0.052, "couleur": "#8B0000"},
            "Autres":              {"parti": "Divers",   "score_nat": 0.295, "couleur": "#999"},
        },
        "tour2": {
            "Jacques Chirac":    {"parti": "RPR/UMP", "score_nat": 0.820, "couleur": "#1E90FF"},
            "Jean-Marie Le Pen": {"parti": "FN",      "score_nat": 0.180, "couleur": "#1a1a2e"},
        },
        "participation_t1": 0.718, "participation_t2": 0.800,
    },
    2007: {
        "tour1": {
            "Nicolas Sarkozy":   {"parti": "UMP",    "score_nat": 0.312, "couleur": "#1E90FF"},
            "Ségolène Royal":    {"parti": "PS",     "score_nat": 0.259, "couleur": "#FF69B4"},
            "François Bayrou":   {"parti": "UDF/MoDem","score_nat": 0.187,"couleur": "#FF8C00"},
            "Jean-Marie Le Pen": {"parti": "FN",     "score_nat": 0.104, "couleur": "#1a1a2e"},
            "Olivier Besancenot":{"parti": "LCR",    "score_nat": 0.041, "couleur": "#CC0000"},
            "Autres":            {"parti": "Divers", "score_nat": 0.097, "couleur": "#999"},
        },
        "tour2": {
            "Nicolas Sarkozy":   {"parti": "UMP", "score_nat": 0.531, "couleur": "#1E90FF"},
            "Ségolène Royal":    {"parti": "PS",  "score_nat": 0.469, "couleur": "#FF69B4"},
        },
        "participation_t1": 0.837, "participation_t2": 0.840,
    },
    2012: {
        "tour1": {
            "François Hollande":  {"parti": "PS",      "score_nat": 0.286, "couleur": "#FF69B4"},
            "Nicolas Sarkozy":    {"parti": "UMP",     "score_nat": 0.274, "couleur": "#1E90FF"},
            "Marine Le Pen":      {"parti": "FN",      "score_nat": 0.179, "couleur": "#1a1a2e"},
            "Jean-Luc Mélenchon": {"parti": "FG",      "score_nat": 0.114, "couleur": "#CC0000"},
            "François Bayrou":    {"parti": "MoDem",   "score_nat": 0.090, "couleur": "#FF8C00"},
            "Autres":             {"parti": "Divers",  "score_nat": 0.057, "couleur": "#999"},
        },
        "tour2": {
            "François Hollande":  {"parti": "PS",  "score_nat": 0.516, "couleur": "#FF69B4"},
            "Nicolas Sarkozy":    {"parti": "UMP", "score_nat": 0.484, "couleur": "#1E90FF"},
        },
        "participation_t1": 0.797, "participation_t2": 0.806,
    },
    2017: {
        "tour1": {
            "Emmanuel Macron":    {"parti": "LREM",    "score_nat": 0.241, "couleur": "#FFCC00"},
            "Marine Le Pen":      {"parti": "FN/RN",   "score_nat": 0.214, "couleur": "#1a1a2e"},
            "François Fillon":    {"parti": "LR",      "score_nat": 0.200, "couleur": "#1E90FF"},
            "Jean-Luc Mélenchon": {"parti": "FI",      "score_nat": 0.198, "couleur": "#CC0000"},
            "Benoît Hamon":       {"parti": "PS",      "score_nat": 0.062, "couleur": "#FF69B4"},
            "Autres":             {"parti": "Divers",  "score_nat": 0.085, "couleur": "#999"},
        },
        "tour2": {
            "Emmanuel Macron":    {"parti": "LREM",   "score_nat": 0.661, "couleur": "#FFCC00"},
            "Marine Le Pen":      {"parti": "FN/RN",  "score_nat": 0.339, "couleur": "#1a1a2e"},
        },
        "participation_t1": 0.778, "participation_t2": 0.742,
    },
    2022: {
        "tour1": {
            "Emmanuel Macron":    {"parti": "LREM/RE", "score_nat": 0.277, "couleur": "#FFCC00"},
            "Marine Le Pen":      {"parti": "RN",      "score_nat": 0.231, "couleur": "#1a1a2e"},
            "Jean-Luc Mélenchon": {"parti": "NUPES",   "score_nat": 0.220, "couleur": "#CC0000"},
            "Éric Zemmour":       {"parti": "Reconquête","score_nat": 0.071,"couleur": "#2F4F4F"},
            "Valérie Pécresse":   {"parti": "LR",      "score_nat": 0.049, "couleur": "#1E90FF"},
            "Yannick Jadot":      {"parti": "EELV",    "score_nat": 0.047, "couleur": "#2E8B57"},
            "Autres":             {"parti": "Divers",  "score_nat": 0.105, "couleur": "#999"},
        },
        "tour2": {
            "Emmanuel Macron":    {"parti": "RE",  "score_nat": 0.584, "couleur": "#FFCC00"},
            "Marine Le Pen":      {"parti": "RN",  "score_nat": 0.416, "couleur": "#1a1a2e"},
        },
        "participation_t1": 0.737, "participation_t2": 0.719,
    },
}


def gen_departement_scores(annee, tour, dept_code, candidats_data, participation_nat):
    profil = PROFILS_REGIONAUX.get(dept_code, (0, 0, 0, 0))
    g_bonus, d_bonus, ed_bonus, abst_bonus = profil
    participation = max(0.45, min(0.95, participation_nat + abst_bonus + np.random.normal(0, 0.025)))
    scores = {}
    total = 0
    for candidat, info in candidats_data.items():
        score_base = info["score_nat"]
        parti = info["parti"]
        if any(p in parti for p in ["PS", "FI", "FG", "NUPES", "LO", "LCR"]):
            score_base += g_bonus
        elif any(p in parti for p in ["UMP", "LR", "RPR", "UDF", "MoDem"]):
            score_base += d_bonus
        elif any(p in parti for p in ["FN", "RN", "Reconquête"]):
            score_base += ed_bonus
        score_base = max(0.001, score_base + np.random.normal(0, 0.01))
        scores[candidat] = score_base
        total += score_base
    scores = {k: v/total for k, v in scores.items()}
    inscrits = int(np.random.uniform(50000, 450000))
    votants = int(inscrits * participation)
    blancs_nuls = int(votants * np.random.uniform(0.01, 0.03))
    exprimes = votants - blancs_nuls
    rows = []
    for candidat, pct in scores.items():
        voix = int(exprimes * pct)
        rows.append({
            "annee": annee, "tour": tour, "dept_code": dept_code,
            "dept_nom": DEPARTEMENTS[dept_code], "candidat": candidat,
            "parti": candidats_data[candidat]["parti"], "voix": voix,
            "pct_exprimes": round(pct * 100, 2), "inscrits": inscrits,
            "votants": votants, "participation": round(participation * 100, 2),
            "abstentions": inscrits - votants, "blancs_nuls": blancs_nuls,
            "exprimes": exprimes,
        })
    return rows


def generate_elections_dataset():
    print("  Génération des résultats électoraux 2002-2022...")
    all_rows = []
    for annee, data in ELECTIONS_HISTORIQUE.items():
        for tour_key in ["tour1", "tour2"]:
            tour_num = 1 if tour_key == "tour1" else 2
            part_key = f"participation_t{tour_num}"
            candidats = data[tour_key]
            participation = data[part_key]
            for dept_code in DEPARTEMENTS:
                rows = gen_departement_scores(annee, tour_num, dept_code, candidats, participation)
                all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    print(f"  {len(df):,} lignes générées")
    return df


def generate_insee_dataset():
    print("  Génération des données INSEE socio-démographiques...")
    rows = []
    for annee in [2002, 2007, 2012, 2017, 2022]:
        for dept_code, dept_nom in DEPARTEMENTS.items():
            profil = PROFILS_REGIONAUX.get(dept_code, (0, 0, 0, 0))
            g_b = profil[0]
            pop_base = {"75": 2200000, "13": 2050000, "59": 2620000, "69": 1870000,
                "62": 1470000, "93": 1680000, "92": 1600000, "33": 1600000,
                "31": 1380000, "06": 1100000}.get(dept_code, int(np.random.uniform(150000, 750000)))
            croissance = 1 + (annee - 2002) * 0.003
            population = int(pop_base * croissance * np.random.uniform(0.97, 1.03))
            chom_base = 0.10 - g_b * 0.3 + profil[2] * 0.4
            chom_annuel = {2002: 0.090, 2007: 0.082, 2012: 0.098, 2017: 0.093, 2022: 0.073}
            chomage = max(0.04, chom_base * chom_annuel[annee] / 0.09 + np.random.normal(0, 0.015))
            rev_base = 21000 + g_b * 1000 + profil[1] * 2500 - profil[2] * 1500
            rev_annuel = {2002: 18000, 2007: 19500, 2012: 20800, 2017: 21500, 2022: 23000}
            revenu_median = int(rev_base * rev_annuel[annee] / 21000 + np.random.normal(0, 1200))
            diplomés_sup = max(0.10, min(0.60, 0.28 + g_b * 0.15 + profil[1] * 0.05 + (annee - 2002) * 0.004 + np.random.normal(0, 0.03)))
            age_median = max(30, min(52, 41 - g_b * 5 + np.random.normal(0, 2)))
            dens_ref = {"75": 21000, "92": 9500, "93": 6800, "69": 560, "13": 390}.get(dept_code, 80)
            densite = max(10, dens_ref * np.random.uniform(0.95, 1.05))
            part_rurale = max(0.01, min(0.90, 0.45 - np.log10(densite) * 0.12 + np.random.normal(0, 0.05)))
            rows.append({
                "annee": annee, "dept_code": dept_code, "dept_nom": dept_nom,
                "population": population, "taux_chomage": round(chomage * 100, 2),
                "revenu_median": revenu_median, "part_diplomes_sup": round(diplomés_sup * 100, 2),
                "age_median": round(age_median, 1), "densite_hab_km2": round(densite, 1),
                "part_rurale": round(part_rurale * 100, 2),
                "part_emploi_public": round(max(0.12, min(0.45, 0.25 + g_b * 0.1 + np.random.normal(0, 0.03))) * 100, 2),
                "part_ouvriers": round(max(0.05, min(0.40, 0.22 - g_b * 0.05 + profil[2] * 0.08 + np.random.normal(0, 0.03))) * 100, 2),
                "part_cadres": round(max(0.05, min(0.45, 0.18 + g_b * 0.08 + profil[1] * 0.06 + np.random.normal(0, 0.02))) * 100, 2),
                "taux_pauvrete": round(max(0.05, min(0.35, 0.14 - g_b * 0.04 + profil[2] * 0.06 + np.random.normal(0, 0.02))) * 100, 2),
            })
    df = pd.DataFrame(rows)
    print(f"  {len(df):,} lignes INSEE générées")
    return df


def generate_candidats_2027():
    candidats = [
        {"candidat": "Emmanuel Macron (ou successeur RE)", "parti": "Renaissance", "couleur": "#FFCC00", "orientation": "Centre", "score_sondages_t1": 0.22},
        {"candidat": "Marine Le Pen", "parti": "RN", "couleur": "#1a1a2e", "orientation": "Extrême droite", "score_sondages_t1": 0.28},
        {"candidat": "Jean-Luc Mélenchon (ou successeur FI)", "parti": "LFI/NUPES", "couleur": "#CC0000", "orientation": "Gauche radicale", "score_sondages_t1": 0.18},
        {"candidat": "Candidat LR", "parti": "Les Républicains", "couleur": "#1E90FF", "orientation": "Droite", "score_sondages_t1": 0.10},
        {"candidat": "Candidat PS/Gauche", "parti": "PS", "couleur": "#FF69B4", "orientation": "Gauche", "score_sondages_t1": 0.09},
        {"candidat": "Candidat EELV", "parti": "Les Verts", "couleur": "#2E8B57", "orientation": "Gauche écolo", "score_sondages_t1": 0.06},
        {"candidat": "Autres", "parti": "Divers", "couleur": "#999", "orientation": "Divers", "score_sondages_t1": 0.07},
    ]
    return pd.DataFrame(candidats)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(out, exist_ok=True)
    df_elections = generate_elections_dataset()
    df_elections.to_csv(f"{out}/elections_presidentielles_2002_2022.csv", index=False, encoding="utf-8")
    print(f"  Sauvegardé : elections_presidentielles_2002_2022.csv")
    df_insee = generate_insee_dataset()
    df_insee.to_csv(f"{out}/insee_sociodemographie_departements.csv", index=False, encoding="utf-8")
    print(f"  Sauvegardé : insee_sociodemographie_departements.csv")
    df_cand = generate_candidats_2027()
    df_cand.to_csv(f"{out}/candidats_2027_hypotheses.csv", index=False, encoding="utf-8")
    print(f"  Sauvegardé : candidats_2027_hypotheses.csv")
    print("\nDatasets générés avec succès.")
