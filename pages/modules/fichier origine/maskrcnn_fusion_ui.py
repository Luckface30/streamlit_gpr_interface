# pages/modules/maskrcnn_fusion_ui.py
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


def fusion_block():
    st.markdown("### 5) Fusion ASC + DESC (moyenne alignée)")

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    def _as_float(val, default):
        """Convertit une valeur en float avec repli sur défaut."""
        try:
            v = float(val)
            return v if not np.isnan(v) else float(default)
        except Exception:
            return float(default)

    @st.cache_data(show_spinner=False)
    def _load_strat(excels_dir: str, order: str):
        """Charge les Excels et agrège les stratigraphies."""
        return aggregate_excels_to_stratigraphy(excels_dir, sort_order=order)

    def _ensure_raw_loaded(category: str):
        """Charge et met en cache les données brutes ASC/DESC."""
        scan = st.session_state.get("_scan")
        if not scan:
            return None
        out_root = scan.get("output_root")
        if not out_root:
            return None

        if category == "ASC":
            if getattr(st.session_state, "df_asc_raw", None) is None:
                path = os.path.join(out_root, "ASC", "excels")
                res = _load_strat(path, "ASC")
                st.session_state.df_asc_raw = res if res is not None else pd.DataFrame()
            return st.session_state.df_asc_raw
        else:
            if getattr(st.session_state, "df_desc_raw", None) is None:
                path = os.path.join(out_root, "DESC", "excels")
                res = _load_strat(path, "DESC")
                st.session_state.df_desc_raw = res if res is not None else pd.DataFrame()
            return st.session_state.df_desc_raw

    def _ensure_xm(df: pd.DataFrame) -> pd.DataFrame:
        """Garantir une colonne x_m (en mètres)."""
        out = df.copy()
        if "x_m" not in out.columns:
            if "x_cm" in out.columns:
                out["x_m"] = out["x_cm"]
            else:
                cands = [c for c in out.columns if c.startswith("x_")]
                if cands:
                    out["x_m"] = out[cands[0]]
                else:
                    raise ValueError("Aucune colonne d'abscisse trouvée.")
        return out.sort_values("x_m").reset_index(drop=True)

    def _clamp(v, lo, hi):
        try:
            v = float(v)
        except Exception:
            return float(lo)
        if np.isnan(v):
            return float(lo)
        return float(min(max(v, lo), hi))

    # -------------------------------------------------------------
    # Chargement ASC / DESC
    # -------------------------------------------------------------
    df_asc_raw = _ensure_raw_loaded("ASC")
    df_desc_raw = _ensure_raw_loaded("DESC")

    if df_asc_raw is None or df_desc_raw is None or df_asc_raw.empty or df_desc_raw.empty:
        st.info("Charge d'abord ASC et DESC (onglet Stratigraphie).")
        return

    # -------------------------------------------------------------
    # Interface utilisateur de base
    # -------------------------------------------------------------
    unit_key, t_key, h_key = "unit_fusion", "t_ns_fusion", "hreal_fusion"
    style_key, trend_key = "style_fusion", "trend_fusion"
    defaults = {
        unit_key: "px",
        t_key: 24.0,
        h_key: 249.0,
        style_key: "Points",
        trend_key: 5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    cols = st.columns(5)
    with cols[0]:
        st.radio("Unité", ["px", "ns"], key=unit_key, horizontal=True)
    with cols[1]:
        st.number_input(
            "Longueur image (ns) — Fusion",
            min_value=0.0, step=1.0,
            key=f"{t_key}_input",
            value=_as_float(st.session_state.get(t_key), 24.0),
        )
        # ✅ recopie dans la vraie clé (comme dans stratigraphy)
        st.session_state[t_key] = _as_float(st.session_state.get(f"{t_key}_input"), 24.0)
    with cols[2]:
        st.number_input(
            "Hauteur réelle (px) — Fusion",
            min_value=0.0, step=1.0,
            key=f"{h_key}_input",
            value=_as_float(st.session_state.get(h_key), 249.0),
        )
        # ✅ recopie dans la vraie clé (comme dans stratigraphy)
        st.session_state[h_key] = _as_float(st.session_state.get(f"{h_key}_input"), 249.0)
    with cols[3]:
        st.radio(
            "Style",
            ["Points", "Continu", "Tendance (avec points)", "Tendance (sans points)"],
            key=style_key,
            horizontal=True
        )
    with cols[4]:
        st.slider("Fenêtre de lissage (m)", 1, 15, step=1, key=trend_key)

    unit_val = st.session_state[unit_key]
    T_ns_val = _safe_float(st.session_state[t_key], None) if unit_val == "ns" else None
    H_real_val = _safe_float(st.session_state[h_key], None)
    style_val = st.session_state[style_key]
    trend_val = int(st.session_state[trend_key])

    # -------------------------------------------------------------
    # Conversion + Fusion
    # -------------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def _convert_and_fuse(dfA, dfD, unit_val, T_ns_val, H_real_val):
        """Convertit ASC/DESC et les fusionne."""
        A = _ensure_xm(_convert_from_raw(dfA, unit_val, T_ns_val, H_real_val))
        D = _ensure_xm(_convert_from_raw(dfD, unit_val, T_ns_val, H_real_val))

        iface_cols = sorted(
            [c for c in A.columns if c.startswith("interface_") and c in D.columns],
            key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0,
        )
        if not iface_cols:
            return pd.DataFrame(), []

        def _prep(df):
            g = df[["x_m"] + iface_cols].copy()
            g["x_key"] = g["x_m"].astype(float).round(3)
            agg = {c: "mean" for c in iface_cols}
            agg["x_m"] = "mean"
            return g.groupby("x_key", as_index=False).agg(agg)

        A1, D1 = _prep(A), _prep(D)
        x_union = pd.DataFrame({"x_key": sorted(set(A1["x_key"]) | set(D1["x_key"]))})
        Aj = pd.merge(x_union, A1, on="x_key", how="left")
        Dj = pd.merge(x_union, D1, on="x_key", how="left")

        fused = pd.DataFrame({"x_key": x_union["x_key"]})
        fused["x_m"] = np.nanmean(np.vstack([Aj["x_m"], Dj["x_m"]]), axis=0)
        for c in iface_cols:
            fused[c] = np.nanmean(np.vstack([Aj[c], Dj[c]]), axis=0)
        return fused.sort_values("x_m").reset_index(drop=True), iface_cols

    fused, iface_cols = _convert_and_fuse(df_asc_raw, df_desc_raw, unit_val, T_ns_val, H_real_val)
    if fused.empty:
        st.error("Aucune colonne d’interface commune entre ASC et DESC.")
        return

    # -------------------------------------------------------------
    # Bornes et tronçonnage
    # -------------------------------------------------------------
    xmin_all, xmax_all = _xcm_minmax(fused) or (0.0, 100.0)
    xmin_key, xmax_key = "xmin_fuse_m", "xmax_fuse_m"

    for k, v in [(xmin_key, xmin_all), (xmax_key, xmax_all)]:
        if k not in st.session_state:
            st.session_state[k] = float(v)

    # === bouton reset (même logique que stratigraphy) ===
    reset_clicked = st.button("↺ Réinitialiser bornes Fusion", key="reset_bounds_fusion")
    if reset_clicked:
        st.session_state[xmin_key] = float(xmin_all)
        st.session_state[xmax_key] = float(xmax_all)
        xmin_session, xmax_session = xmin_all, xmax_all
    else:
        xmin_session = st.session_state[xmin_key]
        xmax_session = st.session_state[xmax_key]

    c1, c2 = st.columns(2)
    with c1:
        xmin_sel = st.number_input(
            "Borne min (m) — Fusion",
            min_value=float(xmin_all), max_value=float(xmax_all),
            step=1.0,
            value=float(xmin_session),
            key=f"{xmin_key}_input"
        )
    with c2:
        xmax_sel = st.number_input(
            "Borne max (m) — Fusion",
            min_value=float(xmin_all), max_value=float(xmax_all),
            step=1.0,
            value=float(xmax_session),
            key=f"{xmax_key}_input"
        )

    xmin_sel = _clamp(xmin_sel, xmin_all, xmax_all)
    xmax_sel = _clamp(xmax_sel, xmin_all, xmax_all)
    if xmin_sel > xmax_sel:
        xmax_sel = xmin_sel

    st.session_state[xmin_key] = float(xmin_sel)
    st.session_state[xmax_key] = float(xmax_sel)

    fused_view = _filter_troncon(fused, float(xmin_sel), float(xmax_sel))

    # -------------------------------------------------------------
    # Choix d’interfaces à afficher
    # -------------------------------------------------------------
    selected = st.multiselect(
        "Choisis les interfaces à afficher (Fusion) :",
        iface_cols,
        default=iface_cols,
        key="interfaces_fusion"
    )

    # -------------------------------------------------------------
    # Tracé
    # -------------------------------------------------------------
    _plot_stratigraphy(
        df=fused_view,
        title="Fusion ASC + DESC (moyenne alignée)",
        unit_label=unit_val,
        total_time_ns=T_ns_val,
        style_choice=style_val,
        x_col="x_m",
        x_axis_label="Position (m)",
        trend_window_m=trend_val,
        selected_interfaces=selected
    )

    # -------------------------------------------------------------
    # Export Excel
    # -------------------------------------------------------------
    st.markdown("#### 💾 Exporter la FUSION en Excel")
    out_root = st.session_state.get("_scan", {}).get("output_root")
    if not out_root:
        st.error("Dossier de sortie introuvable (output_root).")
        return

    out_dir = os.path.join(out_root, "FUSION")
    os.makedirs(out_dir, exist_ok=True)

    c3, c4 = st.columns(2)
    with c3:
        if st.button("Exporter ce tronçon (FUSION)"):
            df_to_save = _rename_interfaces_for_unit(fused_view, unit_val)
            out_path = os.path.join(
                out_dir,
                f"stratigraphy_FUSION_troncon_{int(xmin_sel)}_{int(xmax_sel)}_{unit_val}.xlsx"
            )
            meta = {
                "unit": unit_val,
                "total_time_ns": T_ns_val,
                "real_height_px": H_real_val,
                "xmin_m": float(xmin_sel),
                "xmax_m": float(xmax_sel),
            }
            _export_excel(df_to_save, unit_val, out_path, meta)
            st.success(f"Export ok: {out_path}")

    with c4:
        if st.button("Exporter toute la stratigraphie (FUSION)"):
            df_all_to_save = _rename_interfaces_for_unit(fused, unit_val)
            out_path = os.path.join(out_dir, f"stratigraphy_FUSION_full_{unit_val}.xlsx")
            meta = {
                "unit": unit_val,
                "total_time_ns": T_ns_val,
                "real_height_px": H_real_val,
            }
            _export_excel(df_all_to_save, unit_val, out_path, meta)
            st.success(f"Export ok: {out_path}")

    # Sauvegarde dans la session
    st.session_state["df_strat_FUSION"] = fused_view.copy()
