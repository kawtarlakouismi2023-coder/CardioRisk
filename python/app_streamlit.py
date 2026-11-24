
#Usage : streamlit run app_streamlit.py

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

DOSSIER = Path(__file__).parent

st.set_page_config(
    page_title="CardioRisk",
    layout="centered",
    initial_sidebar_state="auto",
)
# couleurs utilisées dans les cartes
ROSE = "#B83256"
ROSE_FONCE = "#7A1F3A"
VERT = "#2F7D6B"

# mise en forme (CSS) : titre, cartes, barre de probabilité
st.markdown(
    f"""
<style>
.block-container {{ max-width: 900px; padding-top: 2.5rem; }}

.titre {{ border-bottom: 2px solid {ROSE}; padding-bottom: 0.7rem; margin-bottom: 1.4rem; text-align: center; }}
.titre h1 {{ margin: 0; padding: 0; font-size: 2rem; color: {ROSE_FONCE}; }}
.titre p {{ margin: 0.25rem 0 0 0; color: #6B5A60; }}

.carte {{
    background: #FFFFFF;
    border: 1px solid #EBD3DA;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}}
.carte .etiquette {{ font-size: 0.85rem; color: #6B5A60; text-transform: uppercase; letter-spacing: 0.04em; }}
.carte .valeur {{ font-size: 2.8rem; font-weight: 700; color: {ROSE_FONCE}; line-height: 1.2; }}

.barre {{ position: relative; height: 12px; background: #F3E3E8; border-radius: 6px; margin: 0.8rem 0 0.3rem 0; }}
.barre .rempli {{ height: 100%; background: {ROSE}; border-radius: 6px; }}
.barre .seuil {{ position: absolute; left: 50%; top: -4px; width: 2px; height: 20px; background: #2B1B20; }}
.legende {{ display: flex; justify-content: space-between; font-size: 0.8rem; color: #6B5A60; }}

.verdict {{ display: inline-block; margin-top: 0.9rem; padding: 0.3rem 0.8rem; border-radius: 6px; font-weight: 600; }}
.verdict.eleve {{ background: #FBE3EA; color: {ROSE_FONCE}; }}
.verdict.faible {{ background: #E1F1EC; color: #1E5A4C; }}

.facteur {{ display: flex; align-items: center; margin: 0.35rem 0; font-size: 0.92rem; }}
.facteur .nom {{ width: 46%; }}
.facteur .nom small {{ color: #6B5A60; }}
.facteur .piste {{ width: 54%; background: #F7EEF1; border-radius: 4px; height: 14px; }}
.facteur .trait {{ height: 100%; border-radius: 4px; }}

.definition {{ background: #FFFFFF; border: 1px solid #EBD3DA; border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.92rem; }}
.definition b {{ color: {ROSE_FONCE}; }}

.note {{ font-size: 0.82rem; color: #6B5A60; margin-top: 1.5rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# chargement du modèle (une seule fois grâce au cache)
@st.cache_resource
def charger_modele():
    contenu = joblib.load(DOSSIER / "modele.joblib")
    return contenu["modele"], contenu["colonnes"]


try:
    modele, colonnes = charger_modele()
except FileNotFoundError:
    st.error("Modèle introuvable. Lancez d'abord : python train_model.py")
    st.stop()

# titre de la page
st.markdown(
    """
<div class="titre">
  <h1>CardioRisk</h1>
  <p>Estimation du risque cardiaque</p>
</div>
""",
    unsafe_allow_html=True,
)

OUI_NON = {0: "Non", 1: "Oui"}

# variables numériques : (libellé, min, max, valeur par défaut, pas)
NOMBRES = {
    "age": ("Âge (années)", 20, 100, 55, 1),
    "trestbps": ("Tension au repos (mm Hg)", 80, 220, 130, 1),
    "chol": ("Cholestérol (mg/dl)", 100, 600, 240, 1),
    "thalach": ("Fréquence cardiaque max", 60, 220, 150, 1),
    "oldpeak": ("Dépression ST (oldpeak)", 0.0, 7.0, 1.0, 0.1),
}

# variables à choix : (libellé, {valeur: texte affiché})
CHOIX = {
    "sex": ("Sexe", {0: "Femme", 1: "Homme"}),
    "cp": (
        "Type de douleur thoracique",
        {0: "0 - Angine typique", 1: "1 - Angine atypique",
         2: "2 - Douleur non angineuse", 3: "3 - Asymptomatique"},
    ),
    "fbs": ("Glycémie à jeun > 120 mg/dl", OUI_NON),
    "restecg": (
        "ECG au repos",
        {0: "0 - Normal", 1: "1 - Anomalie ST-T", 2: "2 - Hypertrophie"},
    ),
    "exang": ("Angine provoquée par l'effort", OUI_NON),
    "slope": (
        "Pente du segment ST",
        {0: "0 - Descendante", 1: "1 - Plate", 2: "2 - Montante"},
    ),
    "ca": ("Nombre de vaisseaux colorés", {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}),
    "thal": (
        "Thalassémie",
        {0: "0 - Non précisé", 1: "1 - Défaut fixe",
         2: "2 - Normal", 3: "3 - Défaut réversible"},
    ),
}

# explication de chaque variable (affichée dans la barre latérale)
SIGNIFICATION = {
    "age": "Âge du patient, en années.",
    "sex": "Sexe du patient.",
    "cp": "Nature de la douleur dans la poitrine.",
    "trestbps": "Pression artérielle au repos, en mm Hg.",
    "chol": "Taux de cholestérol dans le sang, en mg/dl.",
    "fbs": "Oui si le taux de sucre à jeun dépasse 120 mg/dl.",
    "restecg": "Résultat de l'électrocardiogramme au repos.",
    "thalach": "Plus haut nombre de battements par minute atteint à l'effort.",
    "exang": "Oui si l'effort déclenche une douleur thoracique.",
    "oldpeak": "Baisse du segment ST à l'effort par rapport au repos.",
    "slope": "Direction du segment ST au pic de l'effort.",
    "ca": "Nombre de gros vaisseaux (0 à 4) visibles à la fluoroscopie.",
    "thal": "Résultat du test au thallium.",
}


def libelle(nom):
    # nom affiché d'une variable
    return NOMBRES[nom][0] if nom in NOMBRES else CHOIX[nom][0]


def texte_valeur(nom, valeur):
    # valeur saisie, écrite en toutes lettres
    if nom in CHOIX:
        return CHOIX[nom][1][valeur]
    return f"{valeur:g}"


def pourcentage(p):
    # 0.734 devient "73,4 %"
    return f"{p * 100:.1f}".replace(".", ",") + " %"


def contributions(patient):
    # poids de chaque variable dans le score : coefficient x valeur standardisée
    scaler = modele.named_steps["standardscaler"]
    reg = modele.named_steps["logisticregression"]
    z = scaler.transform(patient)[0]
    return pd.Series(z * reg.coef_[0], index=colonnes)


# barre latérale : on choisit une variable et sa signification s'affiche
with st.sidebar:
    st.header("Signification des données")
    choisie = st.selectbox("Choisir une variable", colonnes, format_func=libelle)

    st.markdown(
        f'<div class="definition"><b>{libelle(choisie)}</b><br>{SIGNIFICATION[choisie]}</div>',
        unsafe_allow_html=True,
    )

    # pour les variables à choix, on montre aussi la liste des valeurs possibles
    if choisie in CHOIX:
        st.write("Valeurs possibles :")
        for texte in CHOIX[choisie][1].values():
            st.markdown(f"- {texte}")
    else:
        _, mini, maxi, _, _ = NOMBRES[choisie]
        st.write(f"Valeurs acceptées : de {mini:g} à {maxi:g}.")

# formulaire de saisie, sur trois colonnes
st.subheader("Données du patient")

with st.form("formulaire"):
    valeurs = {}
    cols = st.columns(3)
    for i, nom in enumerate(colonnes):
        with cols[i % 3]:
            if nom in NOMBRES:
                texte, mini, maxi, defaut, pas = NOMBRES[nom]
                valeurs[nom] = st.number_input(
                    texte, min_value=mini, max_value=maxi, value=defaut, step=pas
                )
            else:
                texte, options = CHOIX[nom]
                valeurs[nom] = st.selectbox(
                    texte, list(options), format_func=options.get
                )
    envoye = st.form_submit_button(
        "Calculer le risque", type="primary", width="stretch"
    )

# calcul et affichage du résultat après le clic
if envoye:
    patient = pd.DataFrame([valeurs], columns=colonnes)
    proba = float(modele.predict_proba(patient)[0][1])
    eleve = proba >= 0.5  # seuil de décision

    st.subheader("Résultat")
    classe = "eleve" if eleve else "faible"
    verdict = "Risque élevé" if eleve else "Risque faible"
    st.markdown(
        f"""
<div class="carte">
  <div class="etiquette">Probabilité estimée d'être à risque</div>
  <div class="valeur">{pourcentage(proba)}</div>
  <div class="barre"><div class="rempli" style="width:{proba * 100:.1f}%"></div><div class="seuil"></div></div>
  <div class="legende"><span>0 %</span><span>seuil de décision : 50 %</span><span>100 %</span></div>
  <div class="verdict {classe}">{verdict}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # les 6 variables qui ont le plus d'influence sur ce patient
    st.subheader("Variables qui pèsent le plus")
    contrib = contributions(patient)
    principales = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:6]
    maxi = principales.abs().max() or 1
    lignes = []
    for nom, c in principales.items():
        couleur = ROSE if c > 0 else VERT  # rose = augmente, vert = diminue
        largeur = abs(c) / maxi * 100
        sens = "augmente le risque" if c > 0 else "diminue le risque"
        lignes.append(
            f'<div class="facteur"><div class="nom">{libelle(nom)} : {texte_valeur(nom, valeurs[nom])}'
            f'<br><small>{sens}</small></div>'
            f'<div class="piste"><div class="trait" style="width:{largeur:.0f}%;background:{couleur}"></div></div></div>'
        )
    st.markdown(f'<div class="carte">{"".join(lignes)}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="note">Outil pédagogique basé sur un petit jeu de données. '
        ".</div>",
        unsafe_allow_html=True,
    )