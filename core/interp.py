import numpy as np
import pandas as pd

from .utils import clean_numeric, ensure_sorted_unique_by_km, km_to_cm
from core.indicateur import get_eps1, get_eps2, sample_eps1_with_indicator, sample_eps2_with_indicator


def clean_and_interpolate_local(
    df: pd.DataFrame,
    km_col: str,
    numeric_cols,
    position_filter_value,
    position_col,
    drop_cols=None,
    step_cm: int = 10,
    half_window_cm: int = 200,
):
    # Supprimer colonnes demandées (ne jamais supprimer km/position)
    drop_cols = drop_cols or []
    essentials = {km_col}
    if position_col:
        essentials.add(position_col)
    to_drop_final = [c for c in drop_cols if c in df.columns and c not in essentials]
    if to_drop_final:
        df = df.drop(columns=to_drop_final)

    # Filtre position (optionnel)
    if position_col and position_filter_value is not None and position_filter_value != "— Aucun filtre —":
        df = df[df[position_col].astype(str) == str(position_filter_value)].copy()

    # Nettoyage + tri
    numeric_cols_eff = [c for c in numeric_cols if c in df.columns and c != km_col and c not in to_drop_final]
    df = clean_numeric(df, numeric_cols_eff)
    df = ensure_sorted_unique_by_km(df, km_col)

    # Bases d'interpolation
    x_cm = km_to_cm(df[km_col])
    if x_cm.isna().all():
        return None, to_drop_final
    xmin, xmax = np.nanmin(x_cm.values), np.nanmax(x_cm.values)

    valid = x_cm.notna().values
    x_base = x_cm.values[valid]
    if x_base.size < 2:
        return None, to_drop_final
    sort_idx = np.argsort(x_base)
    x_base = x_base[sort_idx]

    interp_bases = {}
    for col in numeric_cols_eff:
        y = pd.to_numeric(df[col], errors="coerce").values
        yb = y[valid][sort_idx]
        interp_bases[col] = (x_base, yb)

    frames = []
    for i, xi in enumerate(x_cm.values):
        if np.isnan(xi):
            continue
        start = max(xmin, xi - half_window_cm)
        end = min(xmax, xi + half_window_cm)
        if end < start:
            continue
        x_win = np.arange(start, end + 0.1, step_cm)
        if x_win.size == 0:
            continue

        df_win = pd.DataFrame({
            km_col: x_win / 100_000.0,
            "anchor_idx": i,
            "anchor_Km": xi / 100_000.0
        })

        # Interpolation linéaire pour colonnes numériques
        for col in numeric_cols_eff:
            xv, yv = interp_bases[col]
            df_win[col] = np.interp(x_win, xv, yv) if xv.size >= 2 else np.nan

        # Colonnes autres (non numériques) : recopie sauf Eps/Ind recalculés
        others = [c for c in df.columns if c not in numeric_cols_eff + [km_col]]
        for col in others:
            if col == "Eps_1":
                obs = df.iloc[i].get("obs") or df.iloc[i].get("observation")
                hum_bs = df.iloc[i].get("humidite_bs")
                eps, ind = sample_eps1_with_indicator(hum_bs, obs)
                df_win["Eps_1"] = eps
                df_win["Ind_1"] = ind
            elif col == "Eps_2":
                obs = df.iloc[i].get("obs") or df.iloc[i].get("observation")
                hum_bc = df.iloc[i].get("humidite_bc")
                colmat = df.iloc[i].get("colmatage")
                eps, ind = sample_eps2_with_indicator(hum_bc, colmat, obs)
                df_win["Eps_2"] = eps
                df_win["Ind_2"] = ind
            else:
                df_win[col] = df.iloc[i][col]

        frames.append(df_win)

    if not frames:
        return None, to_drop_final
    return pd.concat(frames, ignore_index=True), to_drop_final
