# app_streamlit.py - Interface CardioRisk
# Lancer :  streamlit run app_streamlit.py
import os

import joblib
import pandas as pd
import streamlit as st

DOSSIER = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="CardioRisk", page_icon="💗")

# ---------------------------------------------------------------
# 1. COULEURS (le texte est forcé en foncé pour rester lisible,
#    même si le navigateur est en mode sombre)
# ---------------------------------------------------------------
FOND = "#FFF5F7"         # fond principal
FOND2 = "#FFE4EB"        # fond secondaire (formulaire)
ENTETE = "#F9CBD5"       # en-tête
TITRES = "#A8324A"       # titres
BOUTON = "#D93652"       # bouton
SURVOL = "#B92340"       # bouton au survol
CARTE = "#FFFAFB"        # cartes
TEXTE = "#3B1219"        # texte normal (foncé)

st.markdown(f"""
<style>
/* Fonds */
.stApp {{ background-color: {FOND}; }}
[data-testid="stHeader"] {{ background-color: {ENTETE}; }}
[data-testid="stForm"] {{ background-color: {FOND2}; border: 1px solid {ENTETE}; }}

/* Barre latérale (à gauche) et carte SIGNIFICATION */
[data-testid="stSidebar"] {{ background-color: {FOND2}; }}
[data-testid="stExpander"] {{ background-color: {CARTE}; border: 1px solid {ENTETE}; border-radius: 12px; }}
[data-testid="stExpander"] summary {{ background-color: {CARTE}; }}
[data-testid="stExpander"] summary p {{ font-weight: 700; color: {TITRES} !important; }}
[data-testid="stExpander"] svg {{ fill: {TEXTE}; }}

/* Texte : tout en foncé */
.stApp, [data-testid="stSidebar"] *, .stApp p, .stApp label, .stApp span, .stApp li,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stCaptionContainer"], [data-testid="stTable"] * {{
    color: {TEXTE};
}}
h1, h2, h3, h1 *, h2 *, h3 * {{ color: {TITRES} !important; }}

/* Champs de saisie : fond blanc, texte foncé */
input, [data-baseweb="select"] > div, [data-baseweb="input"],
[data-baseweb="base-input"] {{
    background-color: white !important;
    color: {TEXTE} !important;
}}
[data-baseweb="select"] *, [data-baseweb="input"] * {{ color: {TEXTE} !important; }}

/* Messages de résultat : fonds clairs */
[data-testid="stAlert"], [data-testid="stAlert"] div {{ background-color: {CARTE} !important; }}
[data-testid="stAlert"] {{ border: 1px solid {ENTETE}; border-radius: 8px; }}
[data-testid="stAlert"] * {{ color: {TEXTE} !important; }}
[data-baseweb="select"] svg {{ fill: {TEXTE}; }}
[data-baseweb="select"] > div > div:last-child, [data-baseweb="input"] button {{
    background-color: white !important;
}}

/* Bouton */
[data-testid="stFormSubmitButton"] button {{
    background-color: {BOUTON};
    border: none;
}}
[data-testid="stFormSubmitButton"] button * {{ color: white !important; }}
[data-testid="stFormSubmitButton"] button:hover {{ background-color: {SURVOL}; }}

/* Barre de probabilité */
[data-testid="stProgressBarTrack"] {{ background-color: {ENTETE}; }}
[data-testid="stProgressBarTrack"] > div {{ background-color: {BOUTON}; }}

/* Cartes */
.carte {{
    background-color: {CARTE};
    border: 1px solid {ENTETE};
    border-radius: 12px;
    padding: 14px 20px;
}}
.entete {{ background-color: {ENTETE}; border-radius: 14px; padding: 16px 24px; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 2. CHARGEMENT du modèle
# ---------------------------------------------------------------
try:
    contenu = joblib.load(os.path.join(DOSSIER, "modele.joblib"))
except FileNotFoundError:
    st.error("modele.joblib est introuvable. Lancez d'abord : python train_model.py")
    st.stop()

modele = contenu["modele"]      # le pipeline (standardisation + régression logistique)
colonnes = contenu["colonnes"]  # les 13 noms de variables, dans le bon ordre


# ---------------------------------------------------------------
# 3. EN-TÊTE
# ---------------------------------------------------------------
st.markdown("""
<div class="entete">
  <h1 style="margin:0">💗 CardioRisk</h1>
  <p style="margin:4px 0 0 0">Estimation du risque cardiaque.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 3b. SIGNIFICATION : carte déroulante à gauche (barre latérale)
# ---------------------------------------------------------------
SIGNIFICATION = {
    "Âge": "Âge du patient, en années.",
    "Sexe": "Femme ou Homme.",
    "Type de douleur thoracique": "Nature de la douleur dans la poitrine (0 à 3).",
    "Tension au repos": "Pression artérielle au repos, en mm Hg.",
    "Cholestérol": "Taux de cholestérol dans le sang, en mg/dl.",
    "Glycémie à jeun": "Oui si le taux de sucre à jeun dépasse 120 mg/dl.",
    "ECG au repos": "Résultat de l'électrocardiogramme au repos (normal, anomalie ST-T, hypertrophie).",
    "Fréquence cardiaque max": "Plus haut nombre de battements par minute atteint à l'effort.",
    "Angine provoquée par l'effort": "Oui si l'effort déclenche une douleur thoracique.",
    "Dépression ST (oldpeak)": "Baisse du segment ST à l'effort par rapport au repos.",
    "Pente du segment ST": "Direction du segment ST au pic de l'effort (descendante, plate, montante).",
    "Nombre de vaisseaux colorés": "Nombre de gros vaisseaux (0 à 4) visibles à la fluoroscopie.",
    "Thalassémie": "Résultat du test au thallium (non précisé, défaut fixe, normal, défaut réversible).",
}

with st.sidebar:
    with st.expander("SIGNIFICATION", expanded=False):
        for nom, texte in SIGNIFICATION.items():
            st.markdown(f"**{nom}** : {texte}")

# ---------------------------------------------------------------
# 4. FORMULAIRE : une ligne par variable
#    - nombre : (libellé, min, max, valeur par défaut, pas)
#    - choix  : (libellé, {valeur: texte affiché})
# ---------------------------------------------------------------
OUI_NON = {0: "Non", 1: "Oui"}

NOMBRES = {
    "age":      ("Âge (années)", 20, 100, 55, 1),
    "trestbps": ("Tension au repos (mm Hg)", 80, 220, 130, 1),
    "chol":     ("Cholestérol (mg/dl)", 100, 600, 240, 1),
    "thalach":  ("Fréquence cardiaque max", 60, 220, 150, 1),
    "oldpeak":  ("Dépression ST (oldpeak)", 0.0, 7.0, 1.0, 0.1),
}

CHOIX = {
    "sex":     ("Sexe", {0: "Femme", 1: "Homme"}),
    "cp":      ("Type de douleur thoracique",
                {0: "0 - Angine typique", 1: "1 - Angine atypique",
                 2: "2 - Douleur non angineuse", 3: "3 - Asymptomatique"}),
    "fbs":     ("Glycémie à jeun > 120 mg/dl", OUI_NON),
    "restecg": ("ECG au repos",
                {0: "0 - Normal", 1: "1 - Anomalie ST-T", 2: "2 - Hypertrophie"}),
    "exang":   ("Angine provoquée par l'effort", OUI_NON),
    "slope":   ("Pente du segment ST",
                {0: "0 - Descendante", 1: "1 - Plate", 2: "2 - Montante"}),
    "ca":      ("Nombre de vaisseaux colorés", {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}),
    "thal":    ("Thalassémie",
                {0: "0 - Non précisé", 1: "1 - Défaut fixe",
                 2: "2 - Normal", 3: "3 - Défaut réversible"}),
}

st.header("Données du patient")

with st.form("formulaire"):
    valeurs = {}
    for nom in colonnes:
        if nom in NOMBRES:
            libelle, mini, maxi, defaut, pas = NOMBRES[nom]
            valeurs[nom] = st.number_input(libelle, min_value=mini, max_value=maxi,
                                           value=defaut, step=pas)
        else:
            libelle, options = CHOIX[nom]
            valeurs[nom] = st.selectbox(libelle, list(options), format_func=options.get)

    bouton = st.form_submit_button("Prédire le risque", use_container_width=True)

# ---------------------------------------------------------------
# 5. PRÉDICTION (après le clic sur le bouton)
# ---------------------------------------------------------------
if bouton:
    patient = pd.DataFrame([valeurs], columns=colonnes)
    proba = modele.predict_proba(patient)[0][1]   # probabilité d'être "à risque"

    st.header("Probabilité estimée")
    st.markdown(f'<div class="carte"><b>Probabilité d\'être à risque : {proba:.1%}</b></div>',
                unsafe_allow_html=True)
    st.progress(float(proba))

    st.header("Résultat")
    if proba >= 0.5:
        st.error("Catégorie prédite : Risque élevé")
    else:
        st.success("Catégorie prédite : Risque faible")
    st.caption("Outil pédagogique : ce n'est pas un diagnostic médical.")
