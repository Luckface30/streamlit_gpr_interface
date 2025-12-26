import os
import numpy as np
import pandas as pd
import streamlit as st

from .maskrcnn_helpers import (
    _safe_float, _xcm_minmax, _filter_troncon,
    _rename_interfaces_for_unit, _export_excel,
    _convert_from_raw, _plot_stratigraphy
)

def fusion_block():
    st.markdown("### 5) Fusion ASC + DESC (moyenne)")

    def _fuse_asc_desc_from_raw(df_asc_raw, df_desc_raw, unit, total_time_ns, real_height_px):
        if df_asc_raw is None or df_desc_raw is None or df_asc_raw.empty or df_desc_raw.empty:
            return None
        A = _convert_from_raw(df_asc_raw, unit, total_time_ns, real_height_px)
        D = _convert_from_raw(df_desc_raw, unit, total_time_ns, real_height_px)

        a_cols = [c for c in A.columns if c.startswith("interface_")]
        d_cols = [c for c in D.columns if c.startswith("interface_")]
        all_if = sorted(set(a_cols).union(d_cols), key=lambda s: int(s.split("_")[-1]))

        A = A[["x_cm"] + [c for c in all_if if c in A.columns]].copy()
        D = D[["x_cm"] + [c for c in all_if if c in D.columns]].copy()
        A["x_cm"] = pd.to_numeric(A["x_cm"], errors="coerce")
        D["x_cm"] = pd.to_numeric(D["x_cm"], errors="coerce")
        A = A.dropna(subset=["x_cm"])
        D = D.dropna(subset=["x_cm"])

        # ✅ Correction : garder toutes les positions ASC ou DESC
        merged = pd.merge(A, D, on="x_cm", how="outer", suffixes=("_asc", "_desc"))
        if merged.empty:
            return None

        out = pd.DataFrame({"x_cm": merged["x_cm"]})
        for c in all_if:
            a_name = f"{c}_asc" if f"{c}_asc" in merged.columns else None
            d_name = f"{c}_desc" if f"{c}_desc" in merged.columns else None
            a_vals = pd.to_numeric(merged[a_name], errors="coerce").values if a_name else np.full(len(merged), np.nan)
            d_vals = pd.to_numeric(merged[d_name], errors="coerce").values if d_name else np.full(len(merged), np.nan)
            out[c] = np.nanmean(np.vstack([a_vals, d_vals]), axis=0)

        return out.sort_values("x_cm").reset_index(drop=True)

    def _fusion_block():
        scan = st.session_state.get("_scan")
        if not scan:
            st.info("Commence par ‘Scanner le dossier’."); return

        dfA_raw = st.session_state.get("df_asc_raw")
        dfD_raw = st.session_state.get("df_desc_raw")
        if dfA_raw is None or dfA_raw.empty or dfD_raw is None or dfD_raw.empty:
            st.error("Stratigraphie ASC et/ou DESC introuvable(s). Ouvre d’abord l’onglet précédent pour charger les données.")
            return

        # --- init défauts (une fois)
        if "unit_fuse" not in st.session_state:  st.session_state["unit_fuse"]  = "px"
        if "t_ns_fuse" not in st.session_state:  st.session_state["t_ns_fuse"] = 24.0
        if "hreal_fuse" not in st.session_state: st.session_state["hreal_fuse"] = 249.0
        if "style_fuse" not in st.session_state: st.session_state["style_fuse"] = "Points"
        if "trend_fuse" not in st.session_state: st.session_state["trend_fuse"] = 5

        # --- contrôles
        cols = st.columns(5)
        with cols[0]:
            st.radio("Unité fusion", ["px", "ns"], key="unit_fuse", horizontal=True)
        with cols[1]:
            t_val = st.number_input(
                "Longueur image (ns) — fusion",
                min_value=0.0, step=1.0,
                value=_safe_float(st.session_state.get("t_ns_fuse"), 24.0),
                key="t_ns_fuse_input"
            )
            st.session_state["t_ns_fuse"] = t_val
        with cols[2]:
            h_val = st.number_input(
                "Hauteur réelle (px) — fusion",
                min_value=0.0, step=1.0,
                value=_safe_float(st.session_state.get("hreal_fuse"), 249.0),
                key="hreal_fuse_input"
            )
            st.session_state["hreal_fuse"] = h_val
        with cols[3]:
            st.radio("Style",
                     ["Points", "Continu", "Tendance (avec points)", "Tendance (sans points)"],
                     key="style_fuse", horizontal=True)
        with cols[4]:
            st.slider("Fenêtre de lissage (m) — fusion", 1, 15, step=1, key="trend_fuse")

        unit_val  = st.session_state["unit_fuse"]
        T_ns_val  = _safe_float(st.session_state.get("t_ns_fuse"), None) if unit_val == "ns" else None
        H_real_val= _safe_float(st.session_state.get("hreal_fuse"), None)
        style     = st.session_state.get("style_fuse", "Points")
        trend_win = int(st.session_state.get("trend_fuse", 5))

        fused_all = _fuse_asc_desc_from_raw(dfA_raw, dfD_raw, unit_val, T_ns_val, H_real_val)
        if fused_all is None or fused_all.empty:
            st.error("La fusion n’a produit aucune ligne (x_cm non concordants ?).")
            return

        # --- bornes min / max (restaurées uniquement si absentes)
        mm = _xcm_minmax(fused_all)
        if mm is None:
            st.error("x_cm absent ou non numérique (fusion)."); return
        xmin_all, xmax_all = mm
        if st.session_state.get("xmin_fuse") is None:
            st.session_state["xmin_fuse"] = xmin_all
        if st.session_state.get("xmax_fuse") is None:
            st.session_state["xmax_fuse"] = xmax_all

        b1, b2 = st.columns(2)
        with b1:
            xmin_sel = st.number_input(
                "Borne min x_cm (fusion)",
                min_value=float(xmin_all), max_value=float(xmax_all),
                step=10.0,
                value=_safe_float(st.session_state.get("xmin_fuse"), xmin_all),
                key="xmin_fuse_input"
            )
            st.session_state["xmin_fuse"] = xmin_sel
        with b2:
            xmax_sel = st.number_input(
                "Borne max x_cm (fusion)",
                min_value=float(xmin_all), max_value=float(xmax_all),
                step=10.0,
                value=_safe_float(st.session_state.get("xmax_fuse"), xmax_all),
                key="xmax_fuse_input"
            )
            st.session_state["xmax_fuse"] = xmax_sel

        xmin_sel = _safe_float(st.session_state.get("xmin_fuse"), xmin_all) or xmin_all
        xmax_sel = _safe_float(st.session_state.get("xmax_fuse"), xmax_all) or xmax_all
        df_view  = _filter_troncon(fused_all, float(xmin_sel), float(xmax_sel))

        # --- garder x_m réels si x_cm dispo (pour cohérence avec ASC/DESC)
        if "x_cm" in df_view.columns:
            df_view["x_m"] = df_view["x_cm"] / 100.0

        # --- interfaces à afficher
        all_if = [c for c in df_view.columns if c.startswith("interface_")]
        selected_interfaces = st.multiselect("Choisis les interfaces à afficher (fusion) :", all_if, default=all_if)

        st.dataframe(df_view.head(100), use_container_width=True)

        _plot_stratigraphy(
            df_view,
            title="Stratigraphie fusionnée (ASC+DESC)",
            unit_label=unit_val,
            total_time_ns=T_ns_val,
            style_choice=style,
            x_col="x_m" if "x_m" in df_view.columns else "x_cm",
            x_axis_label="Position (m)" if "x_m" in df_view.columns else "Position (cm)",
            selected_interfaces=selected_interfaces,
            trend_window_m=trend_win
        )

        # --- exports
        st.markdown("#### 💾 Exporter la FUSION en Excel")
        out_dir = os.path.join(scan["output_root"], "FUSION")
        e1, e2 = st.columns(2)
        with e1:
            if st.button("Exporter ce tronçon (FUSION)"):
                df_to_save = _rename_interfaces_for_unit(df_view, unit_val)
                out_path   = os.path.join(out_dir, f"stratigraphy_FUSION_troncon_{int(xmin_sel)}_{int(xmax_sel)}_{unit_val}.xlsx")
                meta       = {"unit": unit_val, "total_time_ns": T_ns_val, "real_height_px": H_real_val,
                              "xmin_cm": xmin_sel, "xmax_cm": xmax_sel}
                _export_excel(df_to_save, unit_val, out_path, meta)
                st.success(f"Export ok: {out_path}")
        with e2:
            if st.button("Exporter toute la stratigraphie (FUSION)"):
                df_all_to_save = _rename_interfaces_for_unit(fused_all, unit_val)
                out_path       = os.path.join(out_dir, f"stratigraphy_FUSION_full_{unit_val}.xlsx")
                meta           = {"unit": unit_val, "total_time_ns": T_ns_val, "real_height_px": H_real_val}
                _export_excel(df_all_to_save, unit_val, out_path, meta)

    _fusion_block()
