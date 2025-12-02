#%%
import pandas as pd
from pathlib import Path
import numpy as np 
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


#%%
# --- Import data ---
df = pd.read_excel(Path("data") / "credit.xlsx")

# %%
# Encodage de la variable cible : 1 -> 0, 2 -> 1 puis afficher
df["Cible"] = df["Cible"].map({1: 0, 2: 1})
df["Cible"].value_counts()

# %%
var_num = df[["Age", "Montant_credit", "Duree_credit"]]
var_cat = df.drop(columns=["Cle", "Age", "Montant_credit", "Duree_credit"])


# Obj_credit plus de 5 catégories (9), on fais Woe UNIQUEMENT sur cette variables cat
# Pas les autres

# %%

# test sur la variable Age

# ---------------------------
# 1. Fine classing (exemple sur Age)
# ---------------------------

# 10 bacs par quantiles (modifiable)
df["Age_bin_fine"] = pd.qcut(df["Age"], q=10, duplicates='drop')

# ---------------------------
# 2. Calcul WOE et IV sur les bacs fins
# ---------------------------

def woe_iv(data, feature, target):

    tmp = data.groupby(feature, observed=False)[target].agg(['count', 'sum'])
    tmp.columns = ['total', 'bad']
    tmp['good'] = tmp['total'] - tmp['bad']

    # Proportions globales
    total_good = tmp['good'].sum()
    total_bad = tmp['bad'].sum()

    tmp['pct_good'] = tmp['good'] / total_good
    tmp['pct_bad'] = tmp['bad'] / total_bad

    tmp['WOE'] = np.log(tmp['pct_good'] / tmp['pct_bad'])
    tmp['IV'] = (tmp['pct_good'] - tmp['pct_bad']) * tmp['WOE']

    return tmp, tmp['IV'].sum()

woe_table_fine, iv_fine = woe_iv(df, "Age_bin_fine", "Cible")

print("IV fin =", iv_fine)
print(woe_table_fine)

# ---------------------------
# 3. Coarse classing : regroupement automatique selon le WOE
# ---------------------------

# Ici : on regroupe en fusionnant les bacs proches en WOE (méthode simple)
woe_table_fine = woe_table_fine.sort_values("WOE")
woe_table_fine["WOE_shift"] = woe_table_fine["WOE"].shift()

# seuil de fusion : différence de WOE trop faible
threshold = 0.15
woe_table_fine["merge_group"] = (abs(woe_table_fine["WOE"] - woe_table_fine["WOE_shift"]) > threshold).cumsum()

# Création des bacs grossiers
mapping = woe_table_fine["merge_group"].to_dict()

df["Age_bin_coarse"] = df["Age_bin_fine"].map(mapping)

# ---------------------------
# 4. WOE et IV sur les bacs grossiers
# ---------------------------

woe_table_coarse, iv_coarse = woe_iv(df, "Age_bin_coarse", "Cible")

print("IV coarse =", iv_coarse)
print(woe_table_coarse)

# La stabilité des classes, min 5 %

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['Cle']),
    df['Cle'],
    test_size=0.2,
    random_state=42
)

# %%
# Code automatisé
# Paramètres
n_bins = 10           # Nombre de bacs pour fine classing
threshold = 0.15      # Seuil de fusion pour coarse classing

# Dictionnaire pour stocker les résultats
woe_iv_results = {}

# Boucle sur chaque variable numérique
for var in var_num:
    print(f"\n--- Traitement de {var} ---")
    
    # 1. Fine classing
    bin_fine_col = f"{var}_bin_fine"
    X_train[bin_fine_col] = pd.qcut(X_train[var], q=n_bins, duplicates='drop')
    
    # 2. Calcul WOE/IV sur bacs fins
    woe_table_fine, iv_fine = woe_iv(X_train, bin_fine_col, "Cible")
    print(f"IV fin ({var}) = {iv_fine:.4f}")
    print(woe_table_fine)
    
    # 3. Coarse classing
    woe_table_fine = woe_table_fine.sort_values("WOE")
    woe_table_fine["WOE_shift"] = woe_table_fine["WOE"].shift()
    woe_table_fine["merge_group"] = (abs(woe_table_fine["WOE"] - woe_table_fine["WOE_shift"]) > threshold).cumsum()
    
    # Mapping coarse bins
    mapping = woe_table_fine["merge_group"].to_dict()
    bin_coarse_col = f"{var}_bin_coarse"
    X_train[bin_coarse_col] = X_train[bin_fine_col].map(mapping)
    
    # 4. WOE/IV sur bacs grossiers
    woe_table_coarse, iv_coarse = woe_iv(X_train, bin_coarse_col, "Cible")
    print(f"IV coarse ({var}) = {iv_coarse:.4f}")
    print(woe_table_coarse)
    
    # Stocker les résultats
    woe_iv_results[var] = {
        "fine": (woe_table_fine, iv_fine),
        "coarse": (woe_table_coarse, iv_coarse)
    }

#%%

# -------------------------------------------------------------------
# Ajouter les variables coarse à X_test en utilisant les mêmes bins
# -------------------------------------------------------------------

for var in var_num:
    bin_fine_col = f"{var}_bin_fine"
    bin_coarse_col = f"{var}_bin_coarse"

    # 1) Reproduire les quantiles sur X_test avec les mêmes bornes
    X_test[bin_fine_col] = pd.cut(
        X_test[var],
        bins=X_train[bin_fine_col].cat.categories,
        include_lowest=True
    )

    # 2) Appliquer exactement le même mapping coarse
    mapping = woe_iv_results[var]["fine"][0]["merge_group"].to_dict()

    X_test[bin_coarse_col] = X_test[bin_fine_col].map(mapping)

# %%
print(X_test.head())
# %%
