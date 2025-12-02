# %%
import pandas as pd
from pathlib import Path
import openpyxl as oxl
import numpy as np
import pkg_resources
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder


# %%
# --- Import des données ---
df = pd.read_excel(Path("data") / "credit.xlsx")
df.info()
# 1000 lignes, 21 colonnes
# 12 colonnes numériques (int64), 9 non numériques (object)

# %%
# Encodage de la variable cible : 1 -> 0, 2 -> 1 puis afficher
df["Cible"] = df["Cible"].map({1: 0, 2: 1})
df["Cible"].value_counts()

# %%
# --- Statistiques générales ---
ProfileReport(df, title="Rapport de données - Crédit")
# Pas de valeurs manquantes
# Garanties -> imbalanced
# Distributions semblent fortement assymétriques

# Corrélations fortes :
# entre Biens et Statut_domicile
# Montant_credit et Duree_credit

# A quoi correspondent les valeurs de :
# Historique_credit, Objet_credit, Situation_familiale, Garanties, Biens, Autres_credits, Statut_domicile, Type_emploi, Telephone

# %%
# Sélection des variables numériques et variables catégorielles
# Cle à exclure (car identification de l'observation)
# Variables numériques : Age, Montant_credit, Duree_credit
# Variables catégorielles : toutes les autres
var_num = df[["Age", "Montant_credit", "Duree_credit"]]
var_cat = df.drop(columns=["Cle", "Age", "Montant_credit", "Duree_credit"])

# %%
# --- Statistiques descriptives - Variables numériques ---
print("Statistiques descriptives variables numériques :")
var_num.describe()
# %%
# Distributions
for col in var_num.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(var_num[col], kde=True)
    plt.title(f"Distribution de {col}")
    plt.show()
# Asymétrie pour Age, Montant_credit, et distribution irrégulière et asymétrique pour Duree_credit
# %%
# Boxplot
for col in var_num.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=var_num[col])
    plt.title(f"Boxplot de {col}")
    plt.show()
# 13 valeurs potentiellements atypiques dans Age
# Beaucoup dans Montant_credit
# 10 dans Duree_credit
# %%
# Corrélations
plt.figure(figsize=(10, 8))
corr_matrix = var_num.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation - Variables numériques")
plt.show()
# Corrélation de 0.62 entre Montant_credit et Duree_credit
# %%
# --- Statistiques descriptives - Variables catégorielles ---
print("Statistiques descriptives variables catégorielles :")
var_cat.describe()
# %%
# Barplot
for col in var_cat.columns:
    plt.figure(figsize=(8, 4))
    var_cat[col].value_counts().plot(kind="bar")
    plt.title(f"Répartition de {col}")
    plt.xlabel(col)
    plt.ylabel("Fréquence")
    plt.show()


# %%
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


cat_cols = var_cat.columns
cramers = pd.DataFrame(
    np.zeros((len(cat_cols), len(cat_cols))), index=cat_cols, columns=cat_cols
)
for col1 in cat_cols:
    for col2 in cat_cols:
        cramers.loc[col1, col2] = cramers_v(var_cat[col1], var_cat[col2])
plt.figure(figsize=(10, 8))
sns.heatmap(cramers, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corrélation Cramer's V - Variables catégorielles")
plt.show()
# Corrélation de 0.55 entre Statut_domicile et Biens

# %%
# --- Exploration visuelle des données ---
# Par rapport à la variable Cible (en enlevant les variables continues numériques)
# Barplots croisés
for col in df.drop(
    columns=["Cible", "Cle", "Montant_credit", "Duree_credit", "Age"]
).columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue="Cible", data=df)
    plt.title(f"{col} vs Cible")
    plt.show()
# %%
# Histogrammes empilés
for col in df.drop(
    columns=["Cible", "Cle", "Montant_credit", "Duree_credit", "Age"]
).columns:
    cross_tab = pd.crosstab(df[col], df["Cible"], normalize="index")
    cross_tab.plot(kind="bar", stacked=True, figsize=(6, 4))
    plt.title(f"Proportion de Cible par {col}")
    plt.ylabel("Proportion")
    plt.show()

# %%
# --- Représentation variable Cible ---
prop = df["Cible"].value_counts(normalize=True) * 100
prop.plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Proportion des classes de la variable Cible")
plt.xlabel("Cible")
plt.ylabel("Pourcentage (%)")
plt.show()

# %%

for col in df.select_dtypes("object").columns:
    df[col] = df[col].astype("category")

df.info()

# %%
quali = df.select_dtypes(include=["category"])  # Je sélectionne mes variables quali

# je met cible que mtn car j'en veux pas dans les quali
df["Cible"] = df["Cible"].astype("category")
# %%

le = LabelEncoder()

for var in quali.columns:
    df[var] = le.fit_transform(df[var])

# %%
print(df.head())
# %%

# ===========================================================================
# 1.                                 LOGIT
# ===========================================================================
print("=== LOGIT ===")

# Modèle Logit avec statsmodels
X_train_sm = sm.add_constant(X_train)  # ajout de la constante
logit_model = sm.Logit(y_train, X_train_sm)
logit_result = logit_model.fit(disp=False)

# Prédiction sur test
X_test_sm = sm.add_constant(X_test)
y_pred_prob_logit = logit_result.predict(X_test_sm)
y_pred_logit = (y_pred_prob_logit >= 0.5).astype(int)

# ----------------------------- Indicateurs -------------------------------
# Indice de Gini
gini_logit = 2 * roc_auc_score(y_test, y_pred_prob_logit) - 1
print("Indice de Gini:", gini_logit)

# AUC
auc_logit = roc_auc_score(y_test, y_pred_prob_logit)
print("AUC:", auc_logit)

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_logit)
plt.figure()
plt.plot(fpr, tpr, label=f"Logit (AUC = {auc_logit:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Logit")
plt.legend()
plt.show()

# Matrice de confusion
conf_matrix_logit = confusion_matrix(y_test, y_pred_logit)
print("Matrice de confusion:\n", conf_matrix_logit)

# F-score
fscore_logit = f1_score(y_test, y_pred_logit)
print("F-score:", fscore_logit)

# F-statistique (likelihood ratio test)
fstat_logit = logit_result.llr
print("F-statistique:", fstat_logit)

# ===========================================================================
# 2.                                 PROBIT
# ===========================================================================
print("\n=== PROBIT ===")

probit_model = sm.Probit(y_train, X_train_sm)
probit_result = probit_model.fit(disp=False)

y_pred_prob_probit = probit_result.predict(X_test_sm)
y_pred_probit = (y_pred_prob_probit >= 0.5).astype(int)

# ----------------------------- Indicateurs -------------------------------

# Indice de Gini
gini_probit = 2 * roc_auc_score(y_test, y_pred_prob_probit) - 1
print("Indice de Gini:", gini_probit)

# AUC
auc_probit = roc_auc_score(y_test, y_pred_prob_probit)
print("AUC:", auc_probit)

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_probit)
plt.figure()
plt.plot(fpr, tpr, label=f"Probit (AUC = {auc_probit:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Probit")
plt.legend()
plt.show()

# Matrice de confusion
conf_matrix_probit = confusion_matrix(y_test, y_pred_probit)
print("Matrice de confusion:\n", conf_matrix_probit)

# F-score
fscore_probit = f1_score(y_test, y_pred_probit)
print("F-score:", fscore_probit)

# F-statistique
fstat_probit = probit_result.llr
print("F-statistique:", fstat_probit)

# ===========================================================================
# 3.                                 CATBOOST
# ===========================================================================
print("\n=== CATBOOST ===")

cat_model = CatBoostClassifier(verbose=0, random_state=42)
cat_model.fit(X_train, y_train)

y_pred_prob_cat = cat_model.predict_proba(X_test)[:, 1]
y_pred_cat = (y_pred_prob_cat >= 0.5).astype(int)

# ----------------------------- Indicateurs -------------------------------

# Indice de Gini
gini_cat = 2 * roc_auc_score(y_test, y_pred_prob_cat) - 1
print("Indice de Gini:", gini_cat)

# AUC
auc_cat = roc_auc_score(y_test, y_pred_prob_cat)
print("AUC:", auc_cat)

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_cat)
plt.figure()
plt.plot(fpr, tpr, label=f"CatBoost (AUC = {auc_cat:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CatBoost")
plt.legend()
plt.show()

# Matrice de confusion
conf_matrix_cat = confusion_matrix(y_test, y_pred_cat)
print("Matrice de confusion:\n", conf_matrix_cat)

# F-score
fscore_cat = f1_score(y_test, y_pred_cat)
print("F-score:", fscore_cat)

# Pas de F-statistique classique pour CatBoost, on affiche None
print("F-statistique: None (non applicable pour CatBoost)")

# ===========================================================================
# 4.                            FORÊT ALÉATOIRE
# ===========================================================================
print("\n=== FORÊT ALÉATOIRE ===")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = (y_pred_prob_rf >= 0.5).astype(int)

# ----------------------------- Indicateurs -------------------------------

# Indice de Gini
gini_rf = 2 * roc_auc_score(y_test, y_pred_prob_rf) - 1
print("Indice de Gini:", gini_rf)

# AUC
auc_rf = roc_auc_score(y_test, y_pred_prob_rf)
print("AUC:", auc_rf)

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_rf)
plt.figure()
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc_rf:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Random Forest")
plt.legend()
plt.show()

# Matrice de confusion
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Matrice de confusion:\n", conf_matrix_rf)

# F-score
fscore_rf = f1_score(y_test, y_pred_rf)
print("F-score:", fscore_rf)

# Pas de F-statistique classique pour Random Forest
print("F-statistique: None (non applicable pour Random Forest)")
