#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import math
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Utils
# ---------------------------

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for c in df.columns:
            if cand.lower() == c.lower():
                return c
        # contains?
        for low, orig in cols_lower.items():
            if cand.lower() in low:
                return orig
    return None


def _clean_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    # Remplacer '-' par NaN, retirer '*', strings vides -> NaN, puis to_numeric
    out[cols] = (
        out[cols]
        .replace('-', np.nan)
        .astype(str)
        .replace(r'\*', '', regex=True)
        .replace('', np.nan)
        .apply(pd.to_numeric, errors='coerce')
    )
    return out


def _km_to_cm(km_series: pd.Series) -> pd.Series:
    return pd.to_numeric(km_series, errors='coerce') * 100_000.0


def _ensure_sorted_unique_by_km(df: pd.DataFrame, km_col: str) -> pd.DataFrame:
    # Tri + supprimer doublons sur Km en gardant la 1re occurrence
    out = df.copy()
    out = out.sort_values(km_col, kind="mergesort")
    out = out.drop_duplicates(subset=[km_col], keep="first")
    return out


# ---------------------------
# 1) Nettoyage + Interpolation locale (±2 m, pas 10 cm)
# ---------------------------

def clean_and_interpolate_local(
    infile: str,
    out_xlsx: str,
    sheet: Optional[str] = None,
    position_filter: Optional[str] = "O",
    km_col_candidates: Optional[List[str]] = None,
    step_cm: int = 10,
    half_window_cm: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge un Excel, filtre (optionnel) par 'Position', nettoie les colonnes numériques
    et réalise pour chaque point une interpolation locale ±half_window_cm avec pas 'step_cm'.
    Sauvegarde le résultat dans 'out_xlsx'.
    """
    km_col_candidates = km_col_candidates or ["Km", "KM", "kilometre", "kilometer", "pk", "position_km"]

    # 1) Lecture
    df = pd.read_excel(infile, sheet_name=sheet, engine="openpyxl")

    # 2) Optionnel: filtrer Position == 'O'
    if position_filter is not None:
        pos_col = _find_col(df, ["Position"])
        if pos_col and position_filter in df[pos_col].astype(str).unique().tolist():
            df = df[df[pos_col].astype(str) == position_filter].copy()

    # 3) Trouver Km
    km_col = _find_col(df, km_col_candidates)
    if not km_col:
        raise ValueError(f"Colonne Km introuvable parmi {km_col_candidates}. Colonnes: {list(df.columns)}")

    # 4) Identifier colonnes numériques candidates: tout ce qui est float/int après nettoyage
    #    On nettoie d'abord tout sauf Km et colonnes textuelles probables
    textish = {"position", "id", "name", "nom", "type", "observations", "obs"}
    skip = {km_col}
    num_guess = [c for c in df.columns if c not in skip]
    # Nettoyage large puis sélection des colonnes devenues numériques
    df_clean_try = _clean_numeric(df, num_guess)
    numeric_cols = [c for c in num_guess if pd.api.types.is_numeric_dtype(df_clean_try[c])]

    # 5) Nettoyage final des colonnes numériques
    df = _clean_numeric(df, numeric_cols)

    # 6) Trie/unique sur Km
    df = _ensure_sorted_unique_by_km(df, km_col)

    # 7) Préparer vecteur x en cm
    x_cm = _km_to_cm(df[km_col])
    xmin, xmax = np.nanmin(x_cm.values), np.nanmax(x_cm.values)

    # 8) Préparer (xv,yv) une seule fois par colonne pour accélérer
    interp_bases = {}
    valid_x_mask = x_cm.notna().values
    x_base = x_cm.values[valid_x_mask]
    if x_base.size < 2:
        raise ValueError("Pas assez de points valides sur Km pour interpoler (>=2 requis).")

    sort_idx = np.argsort(x_base)
    x_base = x_base[sort_idx]

    for col in numeric_cols:
        y = pd.to_numeric(df[col], errors="coerce").values
        yb = y[valid_x_mask][sort_idx]
        interp_bases[col] = (x_base, yb)

    # 9) Fenêtre locale par point d’ancrage
    frames = []
    for i, xi in enumerate(x_cm.values):
        if math.isnan(xi):
            continue
        start = max(xmin, xi - half_window_cm)
        end = min(xmax, xi + half_window_cm)
        if end < start:
            continue

        x_win = np.arange(start, end + 0.1, step_cm)  # +0.1 pour inclure la borne
        if x_win.size == 0:
            continue

        # DataFrame de la fenêtre
        df_win = pd.DataFrame({
            km_col: x_win / 100_000.0,
            "anchor_idx": i,
            "anchor_Km": xi / 100_000.0
        })

        # Interpolation pour chaque colonne numérique
        for col in numeric_cols:
            xv, yv = interp_bases[col]
            if xv.size >= 2:
                df_win[col] = np.interp(x_win, xv, yv)
            else:
                df_win[col] = np.nan

        # Recopie des autres colonnes (valeur du point d’ancrage)
        for col in df.columns:
            if col in (numeric_cols + [km_col]):
                continue
            df_win[col] = df.iloc[i][col]

        frames.append(df_win)

    if not frames:
        raise ValueError("Aucune fenêtre construite. Vérifie Km / données.")

    interpolated = pd.concat(frames, ignore_index=True)

    # 10) Colonnes en ordre: Km, anchor_*, numériques, puis le reste
    ordered_cols = [km_col, "anchor_idx", "anchor_Km"] + numeric_cols + \
                   [c for c in df.columns if c not in numeric_cols + [km_col]]
    interpolated = interpolated[ordered_cols]

    # 11) Export
    interpolated.to_excel(out_xlsx, index=False)
    return df, interpolated


# ---------------------------
# 2) Plot stratigraphie (type: lignes + remplissage, y inversé)
# ---------------------------

def plot_stratigraphy(
    infile: str,
    km_col_candidates: Optional[List[str]] = None,
    numeric_cols_slice: Optional[slice] = None,
    title: str = "Stratigraphie interpolée",
    figsize=(14, 6)
):
    km_col_candidates = km_col_candidates or ["Km", "KM", "kilometre", "kilometer", "pk"]
    df = pd.read_excel(infile, engine="openpyxl")
    km_col = _find_col(df, km_col_candidates)
    if not km_col:
        raise ValueError(f"Colonne Km introuvable dans {list(df.columns)}")

    x = df[km_col]
    # Choix par défaut: prendre 5–6 premières colonnes numériques (hors Km)
    num_cols = [c for c in df.columns if c != km_col and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols_slice is None:
        cols = num_cols[:6]
    else:
        cols = num_cols[numeric_cols_slice]

    y_interp = df[cols].apply(pd.to_numeric, errors='coerce') * 100.0

    plt.figure(figsize=figsize)
    for col in y_interp.columns:
        plt.plot(x, y_interp[col], label=col)

    # Remplissage entre couches adjacentes
    for i in range(len(cols) - 1):
        y1 = y_interp[cols[i]]
        y2 = y_interp[cols[i + 1]]
        plt.fill_between(x, y1, y2, alpha=0.3, label=f"{cols[i]} → {cols[i + 1]}")

    plt.xlabel("Position (km)")
    plt.ylabel("Cote (cm)")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------
# 3) ε1 / ε2 assignment (règles + aléa reproductible)
# ---------------------------

def _stable_random_01(*values) -> float:
    # Pseudo-aléatoire stable basé sur un hash du contenu de la ligne
    h = hashlib.sha256(("|".join(map(lambda v: "" if pd.isna(v) else str(v), values))).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def get_eps1(humidite_bs, obs, jitter: float = 0.2) -> float:
    """
    Règles issues de ton script (argileuse/glaiseuse/pollution diffuse/BS),
    avec un petit bruit stable (reproductible).  :contentReference[oaicite:7]{index=7}
    """
    obs_s = str(obs).lower().strip()
    hum = str(humidite_bs).lower().strip()
    r = _stable_random_01(humidite_bs, obs)

    if any(kw in obs_s for kw in ["glaiseuse", "argileuse", "remontées argileuses"]) and "bs" in obs_s:
        base = 9.0
        return round(base + (r - 0.5) * jitter, 1)

    if "pollution diffuse" in obs_s and "bs" in obs_s:
        base = 4.7
        return round(base + (r - 0.5) * jitter, 1)

    if "bs" not in obs_s:
        if hum == "hum":
            base = 4.7
        elif hum == "sat":
            base = 12.0
        else:
            base = 4.0
        return round(base + (r - 0.5) * jitter, 1)

    return round(5.0 + (r - 0.5) * jitter, 1)


def get_eps2(humidite_bc, colmatage, obs, jitter: float = 0.2) -> float:
    """
    Règles issues de ton script (argileuse/glaiseuse, pollution diffuse BC, colmatage GL/N). :contentReference[oaicite:8]{index=8}
    """
    obs_s = str(obs).lower().strip()
    hum = str(humidite_bc).lower().strip()
    colm = str(colmatage).lower().strip()
    r = _stable_random_01(humidite_bc, colmatage, obs)

    if any(kw in obs_s for kw in ["glaiseuse", "argileuse", "remontées argileuses"]) and (
        "bc" in obs_s or "bs" in obs_s or ("bc" not in obs_s and "bs" not in obs_s)
    ):
        base = 12.0
        return round(base + (r - 0.5) * jitter, 1)

    if "pollution diffuse" in obs_s and "bc" in obs_s and not any(kw in obs_s for kw in ["glaiseuse", "argileuse", "remontées argileuses"]):
        if hum == "sec":
            base = 6.7
        elif hum == "hum":
            base = 8.3
        elif hum == "sat":
            base = 12.0
        else:
            base = 6.7
        return round(base + (r - 0.5) * jitter, 1)

    if colm == "gl":
        base = 12.0
    elif colm == "n" and hum == "sec":
        base = 6.7
    elif colm == "n" and hum == "hum":
        base = 9.0
    else:
        base = 8.0

    return round(base + (r - 0.5) * jitter, 1)


def eps_assign(
    infile: str,
    out_xlsx: str,
    col_hum_bs_candidates=("Humidité BS", "Humidite BS", "humidité bs", "humidite bs", "BS hum"),
    col_hum_bc_candidates=("Humidité BC", "Humidite BC", "humidité bc", "humidite bc", "BC hum"),
    col_colmatage_candidates=("Colmatage", "colmatage", "CL", "colm"),
    col_obs_candidates=("Observations", "Observation", "Obs", "obs", "Remarques")
) -> pd.DataFrame:
    """
    Calcule ε1 et ε2 à partir des colonnes d'humidité BS/BC, Colmatage et Observations,
    puis insère ε1/ε2 après la 8e colonne (comme dans ton script) et exporte vers out_xlsx.  :contentReference[oaicite:9]{index=9}
    """
    df = pd.read_excel(infile, engine="openpyxl")
    c_hbs = _find_col(df, list(col_hum_bs_candidates))
    c_hbc = _find_col(df, list(col_hum_bc_candidates))
    c_col = _find_col(df, list(col_colmatage_candidates))
    c_obs = _find_col(df, list(col_obs_candidates))

    missing = [name for name, found in [
        ("Humidité BS", c_hbs),
        ("Humidité BC", c_hbc),
        ("Colmatage", c_col),
        ("Observations", c_obs),
    ] if not found]
    if missing:
        raise ValueError(f"Colonnes introuvables: {missing}. Colonnes={list(df.columns)}")

    df = df.copy()
    df["Eps 1"] = df.apply(lambda r: get_eps1(r[c_hbs], r[c_obs]), axis=1)
    df["Eps 2"] = df.apply(lambda r: get_eps2(r[c_hbc], r[c_col], r[c_obs]), axis=1)

    # Insérer après la 8e colonne (ou à la fin si moins de 8)
    insert_index = 8 if len(df.columns) >= 8 else len(df.columns)
    eps1 = df.pop("Eps 1")
    eps2 = df.pop("Eps 2")
    df.insert(min(insert_index + 1, len(df.columns)), "Eps 1", eps1)
    df.insert(min(insert_index + 2, len(df.columns)), "Eps 2", eps2)

    df.to_excel(out_xlsx, index=False)
    return df


# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Mini-outil data: nettoyage, interpolation locale, plot, eps-assign")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("clean-interp", help="Nettoyer + interpoler localement (±2m, pas 10cm) puis exporter Excel")
    p1.add_argument("--in", dest="infile", required=True)
    p1.add_argument("--out", dest="out_xlsx", default="tableau_interpole_local_2m.xlsx")
    p1.add_argument("--sheet", dest="sheet", default=None)
    p1.add_argument("--position", dest="position_filter", default="O", help="Filtre Position (ex: O); vide pour désactiver")
    p1.add_argument("--step-cm", type=int, default=10)
    p1.add_argument("--half-window-cm", type=int, default=200)

    p2 = sub.add_parser("plot", help="Tracer stratigraphie depuis un Excel (axes inversés, remplissage)")
    p2.add_argument("--in", dest="infile", required=True)
    p2.add_argument("--title", dest="title", default="Stratigraphie interpolée")

    p3 = sub.add_parser("eps-assign", help="Calculer ε1/ε2 et insérer dans le fichier")
    p3.add_argument("--in", dest="infile", required=True)
    p3.add_argument("--out", dest="out_xlsx", default="resultat_avec_eps.xlsx")

    args = p.parse_args()

    if args.cmd == "clean-interp":
        _, interpolated = clean_and_interpolate_local(
            infile=args.infile,
            out_xlsx=args.out_xlsx,
            sheet=args.sheet,
            position_filter=(args.position_filter if args.position_filter != "" else None),
            step_cm=args.step_cm,
            half_window_cm=args.half_window_cm,
        )
        print(f"✅ Export: {args.out_xlsx} [{len(interpolated)} lignes]")

    elif args.cmd == "plot":
        plot_stratigraphy(infile=args.infile, title=args.title)

    elif args.cmd == "eps-assign":
        df = eps_assign(infile=args.infile, out_xlsx=args.out_xlsx)
        print(f"✅ Export: {args.out_xlsx} [{len(df)} lignes]")


if __name__ == "__main__":
    main()
