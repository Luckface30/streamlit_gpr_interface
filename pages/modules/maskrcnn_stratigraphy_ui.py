# pages/modules/maskrcnn_stratigraphy_ui.py

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

    # --- helpers robustes ---
    def _as_float(val, default):
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
        """Charge et met en cache df_asc_raw / df_desc_raw (ASC tri ASC, DESC tri DESC)."""
        scan = st.session_state.get("_scan")
        if not scan:
            return None
        out_root = scan.get("output_root")
        if not out_root:
            return None

        if category == "ASC":
            if getattr(st.session_state, "df_asc_raw", None) is None:
                excels_dir = os.path.join(out_root, "ASC", "excels")
                res = _load_strat(excels_dir, "ASC")
                st.session_state.df_asc_raw = res if isinstance(res, pd.DataFrame) else pd.DataFrame()
            return st.session_state.df_asc_raw
        else:
            if getattr(st.session_state, "df_desc_raw", None) is None:
                excels_dir = os.path.join(out_root, "DESC", "excels")
                res = _load_strat(excels_dir, "DESC")
                st.session_state.df_desc_raw = res if isinstance(res, pd.DataFrame) else pd.DataFrame()
            return st.session_state.df_desc_raw

    def _ensure_xm(df: pd.DataFrame) -> pd.DataFrame:
        """Garantit une colonne x_m (en mètres)."""
        out = df.copy()
        if "x_m" not in out.columns:
            if "x_cm" in out.columns:
                out["x_m"] = out["x_cm"]
            else:
                x_candidates = [c for c in out.columns if c.startswith("x_")]
                if x_candidates:
                    out["x_m"] = out[x_candidates[0]]
                else:
                    raise ValueError("Aucune colonne d'abscisse trouvée (x_m / x_cm).")
        return out.sort_values("x_m").reset_index(drop=True)

    def _clamp(v, lo, hi):
        try:
            v = float(v)
        except Exception:
            return float(lo)
        if np.isnan(v):
            return float(lo)
        return float(min(max(v, lo), hi))

    def _ensure_bound_in_session(key: str, default_val: float, lo: float, hi: float) -> float:
        v = st.session_state.get(key)
        if v is None:
            v = default_val
        v = _clamp(v, lo, hi)
        st.session_state[key] = v
        return v

    # -----------------------------------------------------------------
    # Bloc ASC / DESC
    # -----------------------------------------------------------------
    def _strato_block(category: str):
        scan = st.session_state.get("_scan")
        if not scan:
            st.info("Commence par ‘Scanner le dossier’.")
            return

        df_raw = _ensure_raw_loaded(category)
        if df_raw is None or df_raw.empty:
            st.error("Aucun Excel d’interfaces pour cette catégorie.")
            return

        st.success(f"{len(df_raw)} lignes chargées (RAW px).")

        # --- Variables spécifiques ---
        if category == "ASC":
            unit_key, t_key, h_key = "unit_asc", "t_ns_asc", "hreal_asc"
            xmin_key, xmax_key = "xmin_asc", "xmax_asc"
            style_key, trend_key = "style_asc", "trend_ASC"
            title = "Stratigraphie (ASC)"
        else:
            unit_key, t_key, h_key = "unit_desc", "t_ns_desc", "hreal_desc"
            xmin_key, xmax_key = "xmin_desc", "xmax_desc"
            style_key, trend_key = "style_desc", "trend_DESC"
            title = "Stratigraphie (DESC)"

        # --- Valeurs par défaut (jamais None) ---
        if unit_key not in st.session_state:  st.session_state[unit_key]  = "px"
        if t_key    not in st.session_state:  st.session_state[t_key]     = 24.0
        if h_key    not in st.session_state:  st.session_state[h_key]     = 249.0
        if style_key not in st.session_state: st.session_state[style_key] = "Points"
        if trend_key not in st.session_state: st.session_state[trend_key] = 5

        # --- UI paramétrage ---
        cols = st.columns(5)
        with cols[0]:
            st.radio("Unité", ["px", "ns"], key=unit_key, horizontal=True)
        with cols[1]:
            st.number_input(
                f"Longueur image (ns) {category}",
                min_value=0.0, step=1.0,
                key=f"{t_key}_input",
                value=_as_float(st.session_state.get(t_key), 24.0),
            )
            st.session_state[t_key] = _as_float(st.session_state.get(f"{t_key}_input"), 24.0)
        with cols[2]:
            st.number_input(
                f"Hauteur réelle (px) {category}",
                min_value=0.0, step=1.0,
                key=f"{h_key}_input",
                value=_as_float(st.session_state.get(h_key), 249.0),
            )
            st.session_state[h_key] = _as_float(st.session_state.get(f"{h_key}_input"), 249.0)
        with cols[3]:
            st.radio(
                "Style",
                ["Points", "Continu", "Tendance (avec points)", "Tendance (sans points)"],
                key=style_key, horizontal=True,
            )
        with cols[4]:
            st.slider("Fenêtre de lissage (m)", 1, 15, step=1, key=trend_key)

        # --- Sélection interfaces ---
        all_interfaces = [c for c in df_raw.columns if c.startswith("interface_")]
        selected_interfaces = st.multiselect(
            f"Choisis les interfaces à afficher ({category}) :",
            all_interfaces,
            default=all_interfaces,
            key=f"interfaces_{category}",
        )

        # --- Bornes globales ---
        mm = _xcm_minmax(df_raw)  # (xmin_all, xmax_all) en mètres ou None
        if mm is None:
            xmin_all, xmax_all = 0.0, 100.0
        else:
            xmin_all, xmax_all = float(mm[0]), float(mm[1])
            if not np.isfinite(xmin_all) or not np.isfinite(xmax_all) or xmin_all >= xmax_all:
                xmin_all, xmax_all = 0.0, 100.0

        # Assurer des valeurs valides en session (écrase None historiques)
        xmin_session = _ensure_bound_in_session(xmin_key, xmin_all, xmin_all, xmax_all)
        xmax_session = _ensure_bound_in_session(xmax_key, xmax_all, xmin_all, xmax_all)
        if xmin_session > xmax_session:
            xmin_session, xmax_session = xmin_all, xmax_all
            st.session_state[xmin_key] = xmin_session
            st.session_state[xmax_key] = xmax_session

        # --- Bouton Reset AVANT les inputs : met à jour la session ET les valeurs à injecter ---
        reset_clicked = st.button(f"↺ Réinitialiser bornes {category}", key=f"reset_bounds_{category}")
        if reset_clicked:
            st.session_state[xmin_key] = float(xmin_all)
            st.session_state[xmax_key] = float(xmax_all)
            xmin_session, xmax_session = xmin_all, xmax_all  # pour injection immédiate

        # --- Champs bornes (injecte toujours la valeur de session clampée) ---
        c1, c2 = st.columns(2)
        with c1:
            xmin_sel = st.number_input(
                f"Borne min (m) {category}",
                min_value=float(xmin_all),
                max_value=float(xmax_all),
                step=1.0,
                key=f"{xmin_key}_input",
                value=float(xmin_session),
            )
        with c2:
            xmax_sel = st.number_input(
                f"Borne max (m) {category}",
                min_value=float(xmin_all),
                max_value=float(xmax_all),
                step=1.0,
                key=f"{xmax_key}_input",
                value=float(xmax_session),
            )

        # Clamp final + cohérence min<=max
        xmin_sel = _clamp(xmin_sel, xmin_all, xmax_all)
        xmax_sel = _clamp(xmax_sel, xmin_all, xmax_all)
        if xmin_sel > xmax_sel:
            # si l'utilisateur force une incohérence, on corrige doucement
            xmax_sel = xmin_sel

        st.session_state[xmin_key] = float(xmin_sel)
        st.session_state[xmax_key] = float(xmax_sel)

        # --- Conversion + affichage ---
        unit_val = st.session_state[unit_key]
        T_ns_val = _safe_float(st.session_state[t_key], None) if unit_val == "ns" else None
        H_real_val = _safe_float(st.session_state[h_key], None)
        style = st.session_state[style_key]
        trend_win = int(st.session_state[trend_key])

        df_raw_view = _filter_troncon(df_raw, float(xmin_sel), float(xmax_sel))
        df_view = _convert_from_raw(df_raw_view, unit_val, T_ns_val, H_real_val)
        df_view = _ensure_xm(df_view)

        st.dataframe(df_view.head(100), use_container_width=True)

        _plot_stratigraphy(
            df_view,
            title=title,
            unit_label=unit_val,
            total_time_ns=T_ns_val,
            style_choice=style,
            x_col="x_m",
            x_axis_label="Position (m)",
            selected_interfaces=selected_interfaces,
            trend_window_m=trend_win,
        )

        if category == "ASC":
            st.session_state["df_strat_ASC"] = df_view.copy()
        else:
            st.session_state["df_strat_DESC"] = df_view.copy()

    # --- Affichage côte à côte (ASC | DESC) ---
    left, right = st.columns(2)
    with left:
        _strato_block("ASC")
    with right:
        _strato_block("DESC")
