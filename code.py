#%%
import pandas as pd
from pathlib import Path
import openpyxl as oxl
import numpy as np 
import pkg_resources
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

#%%
df = pd.read_excel(Path("data") / "credit.xlsx")
df.info()
# 1000 lignes, 21 colonnes
# 12 colonnes numériques (int64), 9 catégorielles (object)

#%%
# Sélection des variables numériques et variables catégorielles
var_num = df.select_dtypes(include=['int64', 'float64'])
var_cat = df.select_dtypes(include=['object'])

#%%
# Aperçu des statistiques générales
ProfileReport(df, title = "Rapport de données - Crédit")
# Pas de valeurs manquantes
# Garanties -> imbalanced
# Distributions semblent fortement assymétriques

# Corrélations fortes :  
# entre Biens et Statut_domicile
# Montant_credit et Duree_credit

# A quoi correspondent les valeurs de : 
# Historique_credit, Objet_credit, Situation_familiale, Garanties, Biens, Autres_credits, Statut_domicile, Type_emploi, Telephone

#%%
# --- Statistiques descriptives - Variables numériques ---
print("Statistiques descriptives variables numériques :")
var_num.describe()
# %%
# Distributions
for col in var_num.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(var_num[col], kde=True)
    plt.title(f"Distribution de {col}")
    plt.show()
#%%
# Boxplot
for col in var_num.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df_num[col])
    plt.title(f"Boxplot de {col}")
    plt.show()
#%%
# Corrélations
plt.figure(figsize=(10,8))
corr_matrix = var_num.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation - Variables numériques")
plt.show()

#%%
# --- Statistiques descriptives - Variables catégorielles ---
print("Statistiques descriptives variables catégorielles :")
var_cat.describe()
#%%
# Barplot
for col in var_cat.columns:
    plt.figure(figsize=(8,4))
    var_cat[col].value_counts().plot(kind='bar')
    plt.title(f"Répartition de {col}")
    plt.xlabel(col)
    plt.ylabel("Fréquence")
    plt.show()
#%%
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))
cat_cols = var_cat.columns
cramers = pd.DataFrame(np.zeros((len(cat_cols), len(cat_cols))), 
                       index=cat_cols, columns=cat_cols)
for col1 in cat_cols:
    for col2 in cat_cols:
        cramers.loc[col1, col2] = cramers_v(var_cat[col1], var_cat[col2])
plt.figure(figsize=(10,8))
sns.heatmap(cramers, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corrélation Cramer's V - Variables catégorielles")
plt.show()