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

# %% Code automatisé pour les variables numériques
n_bins = 10       # Nombre de bacs pour fine classing
threshold = 0.15  # Seuil de fusion pour coarse classing

def woe_iv(data, feature, target):
    tmp = data.groupby(feature, observed=False)[target].agg(['count', 'sum'])
    tmp.columns = ['total', 'bad']
    tmp['good'] = tmp['total'] - tmp['bad']

    total_good = tmp['good'].sum()
    total_bad = tmp['bad'].sum()

    tmp['pct_good'] = tmp['good'] / total_good
    tmp['pct_bad'] = tmp['bad'] / total_bad

    # Ajouter un petit epsilon pour éviter log(0)
    tmp['WOE'] = np.log((tmp['pct_good'] + 1e-10) / (tmp['pct_bad'] + 1e-10))
    tmp['IV'] = (tmp['pct_good'] - tmp['pct_bad']) * tmp['WOE']

    return tmp, tmp['IV'].sum()

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
    
    # 3. Coarse classing : fusion des bacs proches selon WOE
    woe_table_fine = woe_table_fine.sort_values("WOE").reset_index()
    merge_group = 0
    groups = []
    for pos, row in woe_table_fine.iterrows():
        if pos == 0:
            merge_group += 1
        else:
            prev_woe = woe_table_fine.loc[pos-1, "WOE"]
            # Fusion si différence WOE > seuil
            if abs(row["WOE"] - prev_woe) > threshold:
                merge_group += 1
        groups.append(merge_group)
    woe_table_fine["merge_group"] = groups
    
    # Mapping coarse bins directement dans X_train
    mapping = dict(zip(woe_table_fine[bin_fine_col], woe_table_fine["merge_group"]))
    bin_coarse_col = f"{var}_bin_coarse"
    X_train[bin_coarse_col] = X_train[bin_fine_col].map(mapping)
    
    # 4. WOE/IV sur bacs grossiers
    woe_table_coarse, iv_coarse = woe_iv(X_train, bin_coarse_col, "Cible")
    print(f"IV coarse ({var}) = {iv_coarse:.4f}")
    print(woe_table_coarse)

#%%
print(X_train.head())

#%%
# Même chose pour Obj_credit (variable catégorielle)

cat_var = "Objet_credit"
min_pct = 0.05  # Seuil minimal de représentation pour good et bad

# 1. Fine classing : chaque catégorie = bac fin
bin_fine_col = f"{cat_var}_bin_fine"
X_train[bin_fine_col] = X_train[cat_var]  # chaque catégorie est déjà un bac fin

# 2. Calcul WOE/IV sur les bacs fins
woe_table_fine, iv_fine = woe_iv(X_train, bin_fine_col, "Cible")
print(f"IV fin ({cat_var}) = {iv_fine:.4f}")
print(woe_table_fine)

# 3. Coarse classing : fusion selon WOE + seuil minimal
woe_table_fine = woe_table_fine.sort_values("WOE")
woe_table_fine["WOE_shift"] = woe_table_fine["WOE"].shift()

# Initialisation du groupe
merge_group = 0
groups = []

for i, row in woe_table_fine.iterrows():
    # Vérifier si la différence de WOE dépasse le seuil OU si la taille relative est suffisante
    if (i == woe_table_fine.index[0]) or \
       (abs(row["WOE"] - woe_table_fine.loc[woe_table_fine.index[i-1], "WOE"]) > 0.15) or \
       (row["pct_good"] < min_pct or row["pct_bad"] < min_pct):
        merge_group += 1
    groups.append(merge_group)

woe_table_fine["merge_group"] = groups

# Mapping coarse bins
mapping = woe_table_fine["merge_group"].to_dict()
bin_coarse_col = f"{cat_var}_bin_coarse"
X_train[bin_coarse_col] = X_train[bin_fine_col].map(mapping)

# 4. WOE/IV sur les bacs grossiers
woe_table_coarse, iv_coarse = woe_iv(X_train, bin_coarse_col, "Cible")
print(f"IV coarse ({cat_var}) = {iv_coarse:.4f}")
print(woe_table_coarse)


# %%
