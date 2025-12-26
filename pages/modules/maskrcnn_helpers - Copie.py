# pages/modules/maskrcnn_helpers.py

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple

import os
import numpy as np
import pandas as pd
import streamlit as st


# ===================== Utilitaires simples ===================== #

def _safe_float(val, default: float = 0.0) -> float:
    """Convertit en float sans lever d'exception."""
    try:
        if val is None:
            return float(default)
        if isinstance(val, (int, float, np.floating)):
            v = float(val)
            return float(default) if np.isnan(v) else v
        s = str(val).strip().replace(",", ".")
        return float(s) if s not in ("", "None", "nan", "NaN") else float(default)
    except Exception:
        return float(default)


def _xcm_minmax(df: pd.DataFrame) -> Tuple[float, float] | None:
    """Retourne (xmin, xmax) pour la sélection UI."""
    if df is None or df.empty:
        return None
    if "x_m" in df.columns:
        s = pd.to_numeric(df["x_m"], errors="coerce").dropna()
    elif "x_cm" in df.columns:
        s = pd.to_numeric(df["x_cm"], errors="coerce").dropna()
    else:
        return None
    if s.empty:
        return None
    return float(s.min()), float(s.max())


def _filter_troncon(df: pd.DataFrame, xmin: float, xmax: float) -> pd.DataFrame:
    """Filtre le DataFrame au tronçon [xmin, xmax]."""
    if df is None or df.empty:
        return df
    xcol = "x_m" if "x_m" in df.columns else ("x_cm" if "x_cm" in df.columns else None)
    if xcol is None:
        return df
    xv = pd.to_numeric(df[xcol], errors="coerce")
    mask = (xv >= float(xmin)) & (xv <= float(xmax))
    return df.loc[mask].copy()


def _rename_interfaces_for_unit(df: pd.DataFrame, unit_val: str) -> pd.DataFrame:
    """Ajoute le suffixe d’unité aux colonnes d’interfaces."""
    unit = str(unit_val).lower()
    out = df.copy()
    ren = {c: f"{c}_{unit}" for c in out.columns if str(c).startswith("interface_")}
    return out.rename(columns=ren)


def _export_excel(df: pd.DataFrame, unit_val: str, out_path: str, meta: Optional[Dict[str, Any]] = None):
    """Écrit le DataFrame (sheet 'stratigraphie') + meta éventuelle (sheet 'meta')."""
    if df is None:
        df = pd.DataFrame()
    dirn = os.path.dirname(out_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="stratigraphie", index=False)
        if meta:
            pd.DataFrame([meta]).to_excel(writer, sheet_name="meta", index=False)


# ===================== Conversion d'unités ===================== #

def _convert_from_raw(df_raw: pd.DataFrame, unit: str, total_time_ns, real_height_px):
    """Convertit les profondeurs selon l’unité choisie."""
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df = df_raw.copy(deep=True)

    if str(unit).lower() != "ns":
        return df

    T = _safe_float(total_time_ns, None)
    if T is None or T <= 0:
        st.warning("⚠️ Conversion en ns impossible : longueur image (ns) invalide.")
        return df

    interf_cols = [c for c in df.columns if str(c).startswith("interface_")]
    if not interf_cols:
        return df

    Hreal = _safe_float(real_height_px, None)
    if Hreal and Hreal > 0:
        factor = T / Hreal
        for c in interf_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce") * factor
        return df

    if "height_px" not in df.columns:
        st.warning("⚠️ Conversion en ns impossible : height_px absent et hauteur réelle non fournie.")
        return df

    H = pd.to_numeric(df["height_px"], errors="coerce").mask(lambda x: (x <= 0))
    if H.isna().all():
        st.warning("⚠️ Conversion en ns impossible : height_px invalide partout.")
        return df

    for c in interf_cols:
        df[c] = (pd.to_numeric(df[c], errors="coerce") / H) * T

    return df


# ===================== Tendance / Binning ===================== #

def _rolling_trend(df: pd.DataFrame, x_col: str, ycols: List[str], window_m: int = 5) -> pd.DataFrame:
    """Calcule une tendance médiane glissante sur une fenêtre en mètres."""
    if df is None or df.empty or x_col not in df.columns or not ycols:
        return pd.DataFrame()
    try:
        w = float(window_m)
        if w <= 0:
            w = 5.0
    except Exception:
        w = 5.0

    x_raw = pd.to_numeric(df[x_col], errors="coerce")
    if x_raw.isna().all():
        return pd.DataFrame()
    x_m = x_raw / 100.0 if x_col == "x_cm" else x_raw
    x_bin_left = np.floor(x_m / w) * w
    x_bin_center = x_bin_left + w * 0.5
    out = pd.DataFrame({"x_m": x_bin_center.astype(float)})
    for c in ycols:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    g = out.groupby("x_m", as_index=False).median(numeric_only=True)
    return g.sort_values("x_m").reset_index(drop=True)


# ===================== Tracé ===================== #

def _plot_stratigraphy(
    df: pd.DataFrame,
    title: str,
    unit_label: str,
    total_time_ns,
    style_choice: str,
    x_col: str = "x_cm",
    x_axis_label: str = None,
    selected_interfaces=None,
    trend_window_m: int = 5
):
    """
    Affiche la stratigraphie avec options de style et tendance.
    """
    import plotly.express as px
    import plotly.graph_objects as go

    if selected_interfaces is None:
        ycols = [c for c in df.columns if str(c).startswith("interface_")]
    else:
        ycols = [c for c in selected_interfaces if c in df.columns]

    if not ycols or df.empty or x_col not in df.columns:
        st.info("Aucune donnée à tracer.")
        return

    # ---- Points / Continu ----
    if style_choice in ("Points", "Continu"):
        df_bin = _rolling_trend(df, x_col=x_col, ycols=ycols, window_m=1)
        if df_bin.empty:
            st.info("Aucune donnée après agrégation.")
            return
        if x_col == "x_cm":
            x_plot = df_bin["x_m"] * 100.0
            xlab = x_axis_label or "Position (cm)"
        else:
            x_plot = df_bin["x_m"]
            xlab = x_axis_label or "Position (m)"
        df_plot = df_bin.copy()
        df_plot["__x_plot__"] = x_plot
        fig = px.scatter(df_plot, x="__x_plot__", y=ycols, title=title) if style_choice == "Points" else px.line(df_plot, x="__x_plot__", y=ycols, title=title)

    # ---- Tendance ----
    elif "Tendance" in style_choice:
        if style_choice == "Tendance (avec points)":
            fig = px.scatter(df, x=x_col, y=ycols, title=title, opacity=0.4)
        else:
            fig = go.Figure(layout_title_text=title)
        for c in ycols:
            tr = _rolling_trend(df, x_col=x_col, ycols=[c], window_m=int(trend_window_m))
            if tr.empty:
                continue
            x_tr = tr["x_m"] * 100.0 if x_col == "x_cm" else tr["x_m"]
            fig.add_scatter(x=x_tr, y=tr[c], mode="lines", name=f"Tendance {c}")
        xlab = x_axis_label or ("Position (cm)" if x_col == "x_cm" else "Position (m)")

    # ---- Fallback ----
    else:
        fig = px.scatter(df, x=x_col, y=ycols, title=title)
        xlab = x_axis_label or ("Position (cm)" if x_col == "x_cm" else "Position (m)")

    # === Axes ===
    fig.update_xaxes(title=xlab)
    if str(unit_label).lower() == "ns" and total_time_ns and _safe_float(total_time_ns) > 0:
        fig.update_yaxes(title="Profondeur (ns)", autorange=False, range=[float(_safe_float(total_time_ns)), 0.0])
    else:
        fig.update_yaxes(title=f"Profondeur ({unit_label})", autorange="reversed")

    # === Correction : bornes pour reset ===
    x_range = [float(df[x_col].min()), float(df[x_col].max())]
    y_min, y_max = None, None
    for c in ycols:
        vals = pd.to_numeric(df[c], errors="coerce")
        if not vals.empty:
            vmin, vmax = vals.min(), vals.max()
            if y_min is None or vmin < y_min:
                y_min = vmin
            if y_max is None or vmax > y_max:
                y_max = vmax
    fig.update_xaxes(range=x_range)
    if y_min is not None and y_max is not None:
        fig.update_yaxes(range=[y_max, y_min])

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False, "doubleClick": "reset+autosize"},
    )
