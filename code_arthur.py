#%%
import pandas as pd
from pathlib import Path
import numpy as np 
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

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

# %%

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
