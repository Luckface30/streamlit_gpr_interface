import os
import numpy as np
import pandas as pd
import streamlit as st

def _safe_float(val, default=None):
    try:
        if val is None:
            return default
        v = float(val)
        return default if np.isnan(v) else v
    except Exception:
        return default

def _xcm_minmax(df: pd.DataFrame):
    if df is None or df.empty or "x_cm" not in df.columns:
        return None
    xcm = pd.to_numeric(df["x_cm"], errors="coerce").dropna()
    if xcm.empty:
        return None
    return float(np.nanmin(xcm.values)), float(np.nanmax(xcm.values))

def _filter_troncon(df: pd.DataFrame, xmin: float, xmax: float):
    if df is None or df.empty or "x_cm" not in df.columns:
        return pd.DataFrame()
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    xcm = pd.to_numeric(df["x_cm"], errors="coerce")
    return df[(xcm >= xmin) & (xcm <= xmax)].copy()

def _rename_interfaces_for_unit(df: pd.DataFrame, unit: str):
    df = df.copy()
    for c in list(df.columns):
        if c.startswith("interface_"):
            df.rename(columns={c: f"{c}_{unit}"}, inplace=True)
    return df

def _export_excel(df: pd.DataFrame, unit: str, out_path: str, meta: dict = None):
    dirn = os.path.dirname(out_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="stratigraphie", index=False)
        if meta:
            pd.DataFrame([meta]).to_excel(writer, sheet_name="meta", index=False)

def _convert_from_raw(df_raw: pd.DataFrame, unit: str, total_time_ns, real_height_px):
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df = df_raw.copy(deep=True)
    if unit != "ns":
        return df
    T = _safe_float(total_time_ns, None)
    if T is None:
        return df
    interf_cols = [c for c in df.columns if c.startswith("interface_")]
    if not interf_cols:
        return df
    Hreal = _safe_float(real_height_px, None)
    if Hreal and Hreal > 0:
        factor = T / Hreal
        for c in interf_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce") * factor
        return df
    if "height_px" not in df.columns:
        return df
    H = pd.to_numeric(df["height_px"], errors="coerce").mask(lambda x: x <= 0, np.nan)
    for c in interf_cols:
        df[c] = (pd.to_numeric(df[c], errors="coerce") / H) * T
    return df

def _avg_by_meter_for_plot(df: pd.DataFrame):
    if df is None or df.empty or "x_cm" not in df.columns:
        return pd.DataFrame()
    ycols = [c for c in df.columns if c.startswith("interface_")]
    if not ycols:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["x_cm"] = pd.to_numeric(tmp["x_cm"], errors="coerce").dropna()
    tmp["x_m"] = (tmp["x_cm"] // 100).astype(int)
    g = tmp.groupby("x_m", as_index=False)[ycols].mean(numeric_only=True)
    return g.sort_values("x_m").reset_index(drop=True)

def _plot_stratigraphy(df: pd.DataFrame, title: str, unit_label: str, total_time_ns, style_choice: str,
                       x_col: str = "x_cm", x_axis_label: str = None):
    ycols = [c for c in df.columns if c.startswith("interface_")]
    if not ycols or x_col not in df.columns or df.empty:
        st.info("Aucune donnée à tracer."); return
    try:
        import plotly.express as px
        fig = px.scatter(df, x=x_col, y=ycols, title=title) if style_choice == "Points" else px.line(df, x=x_col, y=ycols, title=title)
        fig.update_xaxes(title=(x_axis_label or ("Position (cm)" if x_col == "x_cm" else "Position (m)")))
        if unit_label == "ns" and total_time_ns and total_time_ns > 0:
            fig.update_yaxes(title="Profondeur (ns)", autorange=False, range=[float(total_time_ns), 0.0])
        else:
            fig.update_yaxes(title=f"Profondeur ({unit_label})", autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.dataframe(df.head(200), use_container_width=True)
