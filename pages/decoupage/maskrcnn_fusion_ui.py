import os
import numpy as np
import pandas as pd
import streamlit as st

from .maskrcnn_helpers import (
    _safe_float, _xcm_minmax, _filter_troncon,
    _rename_interfaces_for_unit, _export_excel,
    _convert_from_raw, _avg_by_meter_for_plot,
    _plot_stratigraphy
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
        A["x_cm"] = pd.to_numeric(A["x_cm"], errors="coerce").dropna()
        D["x_cm"] = pd.to_numeric(D["x_cm"], errors="coerce").dropna()
        merged = pd.merge(A, D, on="x_cm", how="inner", suffixes=("_asc", "_desc"))
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

        dfA_raw = st.session_state.df_asc_raw
        dfD_raw = st.session_state.df_desc_raw
        if dfA_raw is None or dfA_raw.empty or dfD_raw is None or dfD_raw.empty:
            st.error("Stratigraphie ASC et/ou DESC introuvable(s). Lance l’inférence d’abord.")
            return

        cols = st.columns(5)
        with cols[0]:
            u_def = st.session_state.get("unit_fuse", "px")
            st.radio("Unité fusion", ["px", "ns"], index=(0 if u_def == "px" else 1), key="unit_fuse", horizontal=True)
        with cols[1]:
            t_def = float(st.session_state.get("t_ns_fuse", 100.0))
            st.number_input("Longueur image (ns) — fusion", min_value=0.0, value=t_def, step=1.0, key="t_ns_fuse")
        with cols[2]:
            default_hA = float(np.nanmedian(pd.to_numeric(dfA_raw.get("height_px", pd.Series(dtype=float)), errors="coerce"))) if "height_px" in dfA_raw.columns else np.nan
            default_hD = float(np.nanmedian(pd.to_numeric(dfD_raw.get("height_px", pd.Series(dtype=float)), errors="coerce"))) if "height_px" in dfD_raw.columns else np.nan
            default_h = float(np.nanmean([default_hA, default_hD])) if np.isfinite(default_hA) or np.isfinite(default_hD) else 0.0
            h_def = _safe_float(st.session_state.get("hreal_fuse", default_h), default_h) or default_h
            st.number_input("Hauteur réelle (px) — fusion", min_value=0.0, value=float(h_def), step=1.0, key="hreal_fuse")
        with cols[3]:
            s_def = st.session_state.get("style_fuse", "Points")
            st.radio("Style", ["Points", "Continu"], index=(0 if s_def == "Points" else 1), key="style_fuse", horizontal=True)
        with cols[4]:
            if st.session_state["unit_fuse"] == "ns" and st.session_state["hreal_fuse"] and st.session_state["hreal_fuse"] > 0:
                st.caption(f"Échelle fusion: 1 px ≈ {st.session_state['t_ns_fuse']/st.session_state['hreal_fuse']:.6g} ns/px")
            elif st.session_state["unit_fuse"] == "ns":
                st.caption("Échelle fusion: renseigne la hauteur réelle (px).")

        unit_val = st.session_state["unit_fuse"]
        T_ns_val = _safe_float(st.session_state.get("t_ns_fuse"), None) if unit_val == "ns" else None
        H_real_val = _safe_float(st.session_state.get("hreal_fuse"), None)
        style_choice = st.session_state.get("style_fuse", "Points")

        fused_all = _fuse_asc_desc_from_raw(dfA_raw, dfD_raw, unit_val, T_ns_val, H_real_val)
        if fused_all is None or fused_all.empty:
            st.error("La fusion n’a produit aucune ligne (vérifie que les x_cm coïncident).")
            return

        mm = _xcm_minmax(fused_all)
        if mm is None:
            st.error("x_cm absent ou non numérique (fusion)."); return
        xmin_all, xmax_all = mm

        if st.session_state.get("xmin_fuse") is None or not np.isfinite(_safe_float(st.session_state.get("xmin_fuse"))):
            st.session_state["xmin_fuse"] = xmin_all
        if st.session_state.get("xmax_fuse") is None or not np.isfinite(_safe_float(st.session_state.get("xmax_fuse"))):
            st.session_state["xmax_fuse"] = xmax_all

        b1, b2 = st.columns(2)
        with b1:
            xv_min = _safe_float(st.session_state.get("xmin_fuse"), xmin_all) or xmin_all
            st.number_input("Borne min x_cm (fusion)", value=float(xv_min),
                            min_value=float(xmin_all), max_value=float(xmax_all),
                            step=10.0, key="xmin_fuse")
        with b2:
            xv_max = _safe_float(st.session_state.get("xmax_fuse"), xmax_all) or xmax_all
            st.number_input("Borne max x_cm (fusion)", value=float(xv_max),
                            min_value=float(xmin_all), max_value=float(xmax_all),
                            step=10.0, key="xmax_fuse")

        xmin_sel = _safe_float(st.session_state.get("xmin_fuse"), xmin_all) or xmin_all
        xmax_sel = _safe_float(st.session_state.get("xmax_fuse"), xmax_all) or xmax_all

        df_view = _filter_troncon(fused_all, float(xmin_sel), float(xmax_sel))

        with st.expander("Aperçu des valeurs fusionnées (min/max)"):
            ycols = [c for c in df_view.columns if c.startswith("interface_")]
            if ycols:
                st.write({c: (np.nanmin(df_view[c].values), np.nanmax(df_view[c].values)) for c in ycols})

        st.dataframe(df_view.head(100), use_container_width=True)

        df_plot_m = _avg_by_meter_for_plot(df_view)
        if df_plot_m.empty:
            st.info("Pas de données à tracer après agrégation par mètre (fusion).")
        else:
            _plot_stratigraphy(df_plot_m, title="Stratigraphie fusionnée (ASC+DESC) — moyenne par mètre",
                               unit_label=unit_val, total_time_ns=T_ns_val,
                               style_choice=style_choice, x_col="x_m", x_axis_label="Position (m)")

        st.markdown("#### 💾 Exporter la FUSION en Excel")
        out_dir = os.path.join(scan["output_root"], "FUSION")
        e1, e2 = st.columns(2)
        with e1:
            if st.button("Exporter ce tronçon (FUSION)"):
                df_to_save = _rename_interfaces_for_unit(df_view, unit_val)
                out_path = os.path.join(out_dir, f"stratigraphy_FUSION_troncon_{int(xmin_sel)}_{int(xmax_sel)}_{unit_val}.xlsx")
                meta = {"unit": unit_val, "total_time_ns": T_ns_val, "real_height_px": H_real_val,
                        "xmin_cm": xmin_sel, "xmax_cm": xmax_sel}
                _export_excel(df_to_save, unit_val, out_path, meta)
                st.success(f"Export ok: {out_path}")
        with e2:
            if st.button("Exporter toute la stratigraphie (FUSION)"):
                df_all_to_save = _rename_interfaces_for_unit(fused_all, unit_val)
                out_path = os.path.join(out_dir, f"stratigraphy_FUSION_full_{unit_val}.xlsx")
                meta = {"unit": unit_val, "total_time_ns": T_ns_val, "real_height_px": H_real_val}
                _export_excel(df_all_to_save, unit_val, out_path, meta)

    _fusion_block()
