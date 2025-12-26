import pandas as pd
import numpy as np

# ===== Réglages (en cm) =====
INPUT_XLSX  = "verite_terrain_calais.xlsx"      # ton fichier
SHEET_NAME  = 0
POS_COL     = "Km"               # nom de la colonne position (en cm)
STEP_CM     = 10                 # 10 cm
RADIUS_CM   = 200                # ± 200 cm
OUTPUT_XLSX = "positions_interpolees_cm.xlsx"

# ===== Lecture & préparation =====
df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
if POS_COL not in df.columns:
    raise ValueError(f"Colonne position '{POS_COL}' introuvable.")

df = df.copy()
df[POS_COL] = pd.to_numeric(df[POS_COL], errors="coerce")
df = df.dropna(subset=[POS_COL]).sort_values(POS_COL).reset_index(drop=True)

# Colonnes numériques vs non numériques
num_cols, non_num_cols = [], []
for c in df.columns:
    if c == POS_COL:
        continue
    s = pd.to_numeric(df[c], errors="coerce")
    (num_cols if s.notna().mean() >= 0.5 else non_num_cols).append(c)

# ===== Grille globale + interpolation numérique (index = cm) =====
pos = df[POS_COL].to_numpy()
pos_min, pos_max = pos.min() - RADIUS_CM, pos.max() + RADIUS_CM
grid = np.arange(pos_min, pos_max + STEP_CM//2, STEP_CM, dtype=float)

df_num = df[[POS_COL] + num_cols].set_index(POS_COL).reindex(grid)
df_num = df_num.interpolate(method="index", limit_direction="both")

# ===== Conserver uniquement ±200 cm autour d’un point d’origine =====
sorted_pos = np.sort(pos)
idx = np.searchsorted(sorted_pos, grid)
left  = np.clip(idx - 1, 0, len(sorted_pos) - 1)
right = np.clip(idx,     0, len(sorted_pos) - 1)
nearest = np.where(np.abs(grid - sorted_pos[left]) <= np.abs(grid - sorted_pos[right]),
                   sorted_pos[left], sorted_pos[right])
keep_mask = np.abs(grid - nearest) <= RADIUS_CM

grid_kept = grid[keep_mask]
nearest_kept = nearest[keep_mask]

df_num = df_num.loc[grid_kept].reset_index().rename(columns={"index": POS_COL})
df_num["nearest_origin"] = nearest_kept

# ===== Colonnes non numériques : dupliquer la valeur du point d’origine le plus proche =====
df_cat_src = df.set_index(POS_COL)[non_num_cols]

def copy_nearest(row):
    return df_cat_src.loc[row["nearest_origin"]]

cats = df_num.apply(copy_nearest, axis=1)
df_out = pd.concat([df_num.drop(columns=["nearest_origin"]), cats.reset_index(drop=True)], axis=1)

# Ordonner les colonnes
cols = [POS_COL] + num_cols + non_num_cols
df_out = df_out[cols]

# ===== Sauvegarde =====
with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
    df_out.to_excel(writer, index=False, sheet_name="interpole_±200cm_pas10cm")

print(f"OK : {len(df_out)} lignes écrites dans '{OUTPUT_XLSX}'.")
