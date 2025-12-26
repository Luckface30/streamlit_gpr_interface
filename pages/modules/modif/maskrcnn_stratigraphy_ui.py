import os
import numpy as np
import pandas as pd
import streamlit as st

from core.mrcnn_infer import aggregate_excels_to_stratigraphy
from .maskrcnn_helpers import (
    _safe_float, _xcm_minmax, _filter_troncon,
    _rename_interfaces_for_unit, _export_excel,
    _convert_from_raw, _plot_stratigraphy
)

def stratigraphy_block():
    st.markdown("### 4) Représentation & export de la stratigraphie (par catégorie)")

    # --- helper robuste : coerce en float avec fallback
    def _as_float(val, default):
        if val is None:
            return float(default)
        try:
            v = float(val)
            if np.isnan(v):
                return float(default)
            return v
        except Exception:
            return float(default)

    @st.cache_data(show_spinner=False)
    def _load_strat(excels_dir: str, order: str):
        return aggregate_excels_to_stratigraphy(excels_dir, sort_order=order)

    def _ensure_raw_loaded(category: str):
        scan = st.session_state.get("_scan")
        if not scan:
            return None
        if category == "ASC":
            if getattr(st.session_state, "df_asc_raw", None) is None:
                excels_dir = os.path.join(scan["output_root"], "ASC", "excels")
                res = _load_strat(excels_dir, "ASC")
                st.session_state.df_asc_raw = res if res is not None else pd.DataFrame()
            return st.session_state.df_asc_raw
        else:
            if getattr(st.session_state, "df_desc_raw", None) is None:
                excels_dir = os.path.join(scan["output_root"], "DESC", "excels")
                res = _load_strat(excels_dir, "DESC")
                st.session_state.df_desc_raw = res if res is not None else pd.DataFrame()
            return st.session_state.df_desc_raw

    def _strato_block(category: str):
        scan = st.session_state.get("_scan")
        if not scan:
            st.info("Commence par ‘Scanner le dossier’."); return

        df_raw = _ensure_raw_loaded(category)
        if df_raw is None or df_raw.empty:
            st.error("Aucun Excel d’interfaces pour cette catégorie. Lance l’inférence d’abord.")
            return

        st.success(f"{len(df_raw)} lignes chargées (RAW px).")

        # --- clés par catégorie
        if category == "ASC":
            unit_key, t_key, h_key   = "unit_asc", "t_ns_asc", "hreal_asc"
            xmin_key, xmax_key       = "xmin_asc", "xmax_asc"
            style_key, trend_key     = "style_asc", "trend_ASC"
            out_dir, title           = os.path.join(st.session_state._scan["output_root"], "ASC"), "Stratigraphie (ASC)"
        else:
            unit_key, t_key, h_key   = "unit_desc", "t_ns_desc", "hreal_desc"
            xmin_key, xmax_key       = "xmin_desc", "xmax_desc"
            style_key, trend_key     = "style_desc", "trend_DESC"
            out_dir, title           = os.path.join(st.session_state._scan["output_root"], "DESC"), "Stratigraphie (DESC)"

        # --- init défauts (force 24/249 si absent)
        if unit_key not in st.session_state:  st.session_state[unit_key]  = "px"
        if t_key    not in st.session_state:  st.session_state[t_key]     = 24.0
        if h_key    not in st.session_state:  st.session_state[h_key]     = 249.0
        if style_key not in st.session_state: st.session_state[style_key] = "Points"
        if trend_key not in st.session_state: st.session_state[trend_key] = 5

        # --- contrôles
        cols = st.columns(5)
        with cols[0]:
            st.radio("Unité", ["px", "ns"], key=unit_key, horizontal=True)
        with cols[1]:
            t_val = st.number_input(
                f"Longueur image (ns) {category}",
                min_value=0.0, step=1.0,
                value=_as_float(st.session_state.get(t_key), 24.0),
                key=f"{t_key}_input"
            )
            st.session_state[t_key] = t_val
        with cols[2]:
            h_val = st.number_input(
                f"Hauteur réelle (px) {category}",
                min_value=0.0, step=1.0,
                value=_as_float(st.session_state.get(h_key), 249.0),
                key=f"{h_key}_input"
            )
            st.session_state[h_key] = h_val
        with cols[3]:
            st.radio(
                "Style",
                ["Points", "Continu", "Tendance (avec points)", "Tendance (sans points)"],
                key=style_key, horizontal=True
            )
        with cols[4]:
            st.slider("Fenêtre de lissage (m)", 1, 15, step=1, key=trend_key)

        # --- interfaces à afficher
        all_interfaces = [c for c in df_raw.columns if c.startswith("interface_")]
        selected_interfaces = st.multiselect(
            f"Choisis les interfaces à afficher ({category}) :",
            all_interfaces,
            default=all_interfaces,
            key=f"interfaces_{category}"
        )

        # --- bornes min / max
        mm = _xcm_minmax(df_raw)
        if mm is None:
            xmin_all, xmax_all = 0.0, 100.0   # fallback si x_cm absent
            st.warning("⚠️ Colonne x_cm absente ou vide → bornes par défaut [0, 100].")
        else:
            xmin_all, xmax_all = mm

        b1, b2 = st.columns(2)
        with b1:
            xmin_sel = st.number_input(
                f"Borne min x_cm ({category})",
                min_value=float(xmin_all), max_value=float(xmax_all),
                step=10.0,
                value=_as_float(st.session_state.get(xmin_key), xmin_all),
                key=f"{xmin_key}_input"
            )
            st.session_state[xmin_key] = xmin_sel
        with b2:
            xmax_sel = st.number_input(
                f"Borne max x_cm ({category})",
                min_value=float(xmin_all), max_value=float(xmax_all),
                step=10.0,
                value=_as_float(st.session_state.get(xmax_key), xmax_all),
                key=f"{xmax_key}_input"
            )
            st.session_state[xmax_key] = xmax_sel

        # --- conversion + tronçon
        unit_val  = st.session_state[unit_key]
        T_ns_val  = _safe_float(st.session_state.get(t_key), None) if unit_val == "ns" else None
        H_real_val= _safe_float(st.session_state.get(h_key), None)
        style     = st.session_state.get(style_key, "Points")
        trend_win = int(st.session_state.get(trend_key, 5))  # toujours en mètres

        df_raw_view = _filter_troncon(df_raw, float(xmin_sel), float(xmax_sel))
        df_view     = _convert_from_raw(df_raw_view, unit_val, T_ns_val, H_real_val)

        # --- garder x_m réels si x_cm dispo
        if "x_cm" in df_view.columns:
            df_view["x_m"] = df_view["x_cm"] / 100.0
        df_view_mean = df_view.copy()

        st.dataframe(df_view_mean.head(100), use_container_width=True)

        # --- tracé avec fenêtre en mètres
        _plot_stratigraphy(
            df_view_mean,
            title=title,
            unit_label=unit_val,
            total_time_ns=T_ns_val,
            style_choice=style,
            x_col="x_m" if "x_m" in df_view_mean.columns else "x_cm",
            x_axis_label="Position (m)" if "x_m" in df_view_mean.columns else "Position (cm)",
            selected_interfaces=selected_interfaces,
            trend_window_m=trend_win
        )

        # --- exports
        st.markdown("#### 💾 Exporter en Excel")
        e1, e2 = st.columns(2)
        with e1:
            if st.button("Exporter ce tronçon", key=f"exp_troncon_{category}"):
                df_to_save = _rename_interfaces_for_unit(df_view_mean, unit_val)
                out_path = os.path.join(out_dir, f"stratigraphy_{category}_troncon_{int(xmin_sel)}_{int(xmax_sel)}_{unit_val}.xlsx")
                meta = {"unit": unit_val, "total_time_ns": T_ns_val, "real_height_px": H_real_val,
                        "xmin_cm": xmin_sel, "xmax_cm": xmax_sel}
                _export_excel(df_to_save, unit_val, out_path, meta)
                st.success(f"Export ok: {out_path}")
        with e2:
            if st.button("Exporter toute la stratigraphie", key=f"exp_full_{category}"):
                df_all_u   = _convert_from_raw(df_raw, unit_val, T_ns_val, H_real_val)
                if "x_cm" in df_all_u.columns:
                    df_all_u["x_m"] = df_all_u["x_cm"] / 100.0
                df_to_save = _rename_interfaces_for_unit(df_all_u, unit_val)
                out_path   = os.path.join(out_dir, f"stratigraphy_{category}_full_{unit_val}.xlsx")
                meta       = {"unit": unit_val, "total_time_ns": T_ns_val, "real_height_px": H_real_val}
                _export_excel(df_to_save, unit_val, out_path, meta)

    # affichage côte à côte
    left, right = st.columns(2)
    with left:
        st.subheader("ASC")
        _strato_block("ASC")
    with right:
        st.subheader("DESC")
        _strato_block("DESC")
