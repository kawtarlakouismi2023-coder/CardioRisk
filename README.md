# CardioRisk

Petite application Streamlit qui estime le risque cardiaque d'un patient avec une régression logistique entraînée sur le jeu de données Heart Disease (302 patients après suppression des doublons).

## Installation

    pip install -r requirements.txt

## Utilisation

    python train_model.py
    streamlit run app_streamlit.py

`train_model.py` lit `data/DataSet_CardioRisk.csv`, entraîne le modèle et écrit `modele.joblib` et `metriques.json`. L'application charge ensuite ces deux fichiers.

## Contenu

- `train_model.py` : préparation des données, entraînement, évaluation
- `app_streamlit.py` : interface (estimation, performance du modèle, glossaire)
- `.streamlit/config.toml` : thème de l'application
- `data/` : jeu de données

## Limites

Le jeu de données est petit (61 patients en test) : les métriques varient beaucoup d'un découpage à l'autre. L'outil est pédagogique et ne remplace pas un avis médical.
