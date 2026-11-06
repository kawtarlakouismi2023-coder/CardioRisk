
import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# REGLAGE IMPORTANT : quelle valeur de "target" signifie "a risque" ?
# Dans ce dataset, les patients target = 0 ont plus d'angine d'effort,
# plus de vaisseaux atteints et une frequence cardiaque max plus basse :
# ce sont donc eux les patients a risque. Si votre enseignant utilise
# l'autre convention, mettez 1 ici et relancez le script.
VALEUR_A_RISQUE = 0
# ---------------------------------------------------------------

DOSSIER = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(DOSSIER, "..", "data", "DataSet_CardioRisk.csv")
FICHIER_MODELE = os.path.join(DOSSIER, "modele.joblib")
FICHIER_METRIQUES = os.path.join(DOSSIER, "metriques.json")

# 1. Lecture des donnees (on retire le doublon)
donnees = pd.read_csv(CSV).drop_duplicates()
print("Nombre de patients :", len(donnees))

colonnes = [c for c in donnees.columns if c != "target"]   # les 13 variables
X = donnees[colonnes]
y = (donnees["target"] == VALEUR_A_RISQUE).astype(int)      # 1 = a risque, 0 = pas a risque

# 2. Separation apprentissage / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Entrainement : standardisation puis regression logistique, enchainees dans un "pipeline"
modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modele.fit(X_train, y_train)

# 4. Test sur les donnees jamais vues
predictions = modele.predict(X_test)
metriques = {
    "precision_globale": round(accuracy_score(y_test, predictions), 3),
    "precision_positifs": round(precision_score(y_test, predictions), 3),
    "rappel": round(recall_score(y_test, predictions), 3),
    "matrice_confusion": confusion_matrix(y_test, predictions).tolist(),
    "patients_entrainement": len(X_train),
    "patients_test": len(X_test),
    "valeur_a_risque": VALEUR_A_RISQUE,
}
print("Accuracy (bonnes reponses) :", round(metriques["precision_globale"] * 100, 1), "%")
print("Precision :", metriques["precision_positifs"], "| Rappel :", metriques["rappel"])
print("Matrice de confusion [[VN, FP], [FN, VP]] :")
print(confusion_matrix(y_test, predictions))

# 5. Sauvegarde
joblib.dump({"modele": modele, "colonnes": colonnes}, FICHIER_MODELE)
with open(FICHIER_METRIQUES, "w") as f:
    json.dump(metriques, f, indent=2)
print("Modele sauvegarde :", FICHIER_MODELE)
