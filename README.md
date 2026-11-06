# 💗 CardioRisk – Prédiction du risque cardiaque

Application web qui estime le **risque cardiaque** d'un patient à partir de **13 mesures médicales**, grâce à une **régression logistique**.

> Outil pédagogique : le résultat n'est **pas** un diagnostic médical.

**Application en ligne :** *(collez ici votre lien `https://xxxx.streamlit.app`)*

## 1. Comment ça marche

```
DataSet_CardioRisk.csv ──► train_model.py ──► modele.joblib + metriques.json
                                                      │
                  Navigateur ◄──► app_streamlit.py ◄──┘
                  (formulaire + résultat)
```

1. `train_model.py` lit le CSV, entraîne la régression logistique et sauvegarde le modèle.
2. `app_streamlit.py` charge ce modèle et affiche le formulaire.
3. L'utilisateur remplit les 13 champs et clique sur **Prédire le risque**.
4. L'application affiche la **probabilité d'être à risque** et la **catégorie prédite**.

Tout tourne dans **un seul programme Python** : pas besoin de serveur Java ni d'API séparée.

## 2. Les fichiers

| Fichier | Rôle en une phrase |
|---|---|
| `data/DataSet_CardioRisk.csv` | Les données : 1 ligne = 1 patient |
| `python/train_model.py` | Entraîne et teste le modèle, crée `modele.joblib` et `metriques.json` |
| `python/modele.joblib` | Le modèle entraîné (pipeline : standardisation + régression logistique) |
| `python/metriques.json` | Résultats du test : accuracy, précision, rappel, matrice de confusion |
| `python/app_streamlit.py` | L'application web (interface + prédiction) |
| `python/requirements.txt` | Les bibliothèques à installer (scikit-learn est fixé en 1.9.1) |
| `.streamlit/config.toml` | Thème clair aux couleurs du projet (à la racine du dépôt) |

## 3. L'interface

- **En-tête** CardioRisk.
- **Carte SIGNIFICATION** (à gauche, déroulante) : explique chacun des 13 éléments à saisir.
- **Formulaire** des 13 variables (champs numériques et listes déroulantes).
- **Bouton rouge « Prédire le risque ».**
- **Probabilité estimée** : pourcentage et barre de progression.
- **Résultat** : *Risque faible* ou *Risque élevé*.

Couleurs : fond `#FFF5F7`, formulaire `#FFE4EB`, en-tête `#F9CBD5`, titres `#A8324A`, bouton `#D93652` (survol `#B92340`), cartes `#FFFAFB`.

## 4. Les 13 variables

| Variable | Signification |
|---|---|
| `age` | Âge du patient, en années |
| `sex` | Sexe (0 = femme, 1 = homme) |
| `cp` | Type de douleur thoracique (0 à 3) |
| `trestbps` | Tension artérielle au repos (mm Hg) |
| `chol` | Cholestérol (mg/dl) |
| `fbs` | Glycémie à jeun > 120 mg/dl (0 = non, 1 = oui) |
| `restecg` | ECG au repos (0 normal, 1 anomalie ST-T, 2 hypertrophie) |
| `thalach` | Fréquence cardiaque maximale atteinte |
| `exang` | Angine provoquée par l'effort (0 = non, 1 = oui) |
| `oldpeak` | Dépression du segment ST à l'effort |
| `slope` | Pente du segment ST (0 descendante, 1 plate, 2 montante) |
| `ca` | Nombre de gros vaisseaux colorés (0 à 4) |
| `thal` | Résultat du test au thallium (0 à 3) |

## 5. La régression logistique expliquée simplement

C'est un modèle qui répond à une question **oui / non** (« le patient est-il à risque ? ») en donnant une **probabilité**.

1. **Standardiser** chaque mesure (le cholestérol vaut ~250 et l'âge ~55 : sans cela, les grands nombres dominent).
2. **Calculer un score** : chaque mesure a un poids (coefficient) appris pendant l'entraînement, on additionne.
3. **Appliquer la sigmoïde** `p = 1 / (1 + e^(−score))` : elle transforme le score en probabilité entre 0 et 1.

Si `p ≥ 0,5` → **risque élevé**, sinon → **risque faible**.
Dans le code : `make_pipeline(StandardScaler(), LogisticRegression())`.

## 6. Les résultats du test

- 302 patients (1 doublon supprimé) : 241 pour apprendre, 61 pour tester (jamais vus par le modèle).
- **Accuracy : 82 %** (50 patients bien classés sur 61), **précision : 79 %**, **rappel : 82 %**.
- Matrice de confusion : 27 vrais négatifs, 6 faux positifs, 5 faux négatifs, 23 vrais positifs.

Ces résultats sont dans `metriques.json` (ils ne sont plus affichés dans l'application).

## 7. Lancer le projet sur votre PC

Dans le dossier `python` :

```
pip install -r requirements.txt
python train_model.py        (facultatif : modele.joblib est déjà fourni)
streamlit run app_streamlit.py
```

L'application s'ouvre sur **http://localhost:8501**.

## 8. Mettre l'application en ligne

1. Envoyer le dépôt sur GitHub (avec `python/modele.joblib`).
2. Sur **share.streamlit.io**, cliquer sur *Create app*.
3. Choisir le dépôt, la branche `main` et le fichier **`python/app_streamlit.py`**.
4. Cliquer sur *Deploy* : un lien public est créé.

Si le modèle ne s'ouvre pas, vérifier que `scikit-learn==1.9.1` est bien dans `requirements.txt` (version utilisée pour créer `modele.joblib`).

## 9. Point important : la signification de `target`

Dans ce dataset, les patients `target = 1` ont une fréquence cardiaque maximale plus haute et moins d'angine d'effort, de dépression ST et de vaisseaux atteints : ce sont cliniquement les patients **sans maladie**. Le projet considère donc **« à risque » = `target 0`**.

Si votre enseignant utilise la convention inverse, changez `VALEUR_A_RISQUE = 0` en `1` au début de `train_model.py`, puis relancez `train_model.py`.

## 10. Présenter le projet (3 minutes)

1. **Le problème** : estimer le risque cardiaque d'un patient à partir de 13 mesures médicales.
2. **Les données** : 303 patients, 13 variables, une étiquette (à risque ou non).
3. **Le modèle** : régression logistique entraînée sur 80 % des données, testée sur 20 % (82 % de bonnes réponses).
4. **L'application** : un formulaire Streamlit qui charge le modèle et affiche la probabilité et la catégorie.
5. **La mise en ligne** : l'application est publiée gratuitement et accessible avec un simple lien.
6. **Les limites** : petit dataset, pas de validation clinique : outil pédagogique, pas un diagnostic.

**Questions possibles du jury**
- *Pourquoi la régression logistique et pas linéaire ?* Pour obtenir une probabilité entre 0 et 1 pour une réponse oui/non.
- *Pourquoi un jeu de test ?* Pour mesurer la performance sur des données jamais vues.
- *Pourquoi standardiser ?* Pour que les variables à grande échelle n'écrasent pas les autres.
- *Pourquoi Streamlit ?* Une seule technologie Python pour l'interface et le modèle : simple à développer, à expliquer et à publier.
- *Quelles améliorations ?* Plus de données, comparer d'autres modèles (KNN, arbre de décision), enregistrer les prédictions dans une base SQL, ajouter une API et une interface Java JEE.
