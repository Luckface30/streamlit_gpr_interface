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
    st.markdown("### 5) Fusion ASC + DESC (matching + moyenne si proche)")

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
    gap_key = "gap_thr_fusion_ns"

    defaults = {
        unit_key: "px",
        t_key: 24.0,
        h_key: 249.0,
        style_key: "Points",
        trend_key: 5,
        gap_key: 0.5,  # seuil en ns demandé
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    cols = st.columns(6)
    with cols[0]:
        st.radio("Unité", ["px", "ns"], key=unit_key, horizontal=True)
    with cols[1]:
        st.number_input(
            "Longueur image (ns) — Fusion",
            min_value=0.0, step=1.0,
            value=_as_float(st.session_state[t_key], 24.0),
            key=f"{t_key}_input"
        )
    with cols[2]:
        st.number_input(
            "Hauteur réelle (px) — Fusion",
            min_value=0.0, step=1.0,
            value=_as_float(st.session_state[h_key], 249.0),
            key=f"{h_key}_input"
        )
    with cols[3]:
        st.number_input(
            "Seuil gap (ns) — Fusion",
            min_value=0.0, step=0.1,
            value=_as_float(st.session_state[gap_key], 0.5),
            key=f"{gap_key}_input"
        )
    with cols[4]:
        st.radio(
            "Style",
            ["Points", "Continu", "Tendance (avec points)", "Tendance (sans points)"],
            key=style_key,
            horizontal=True
        )
    with cols[5]:
        st.slider("Fenêtre de lissage (m)", 1, 15, step=1, key=trend_key)

    # synchro des inputs -> clés principales
    st.session_state[t_key] = _as_float(st.session_state.get(f"{t_key}_input", st.session_state[t_key]), 24.0)
    st.session_state[h_key] = _as_float(st.session_state.get(f"{h_key}_input", st.session_state[h_key]), 249.0)
    st.session_state[gap_key] = _as_float(st.session_state.get(f"{gap_key}_input", st.session_state[gap_key]), 0.5)

    unit_val = st.session_state[unit_key]
    T_total_ns = _safe_float(st.session_state[t_key], 24.0)  # toujours utile pour convertir le seuil si unité=px
    T_ns_val = _safe_float(st.session_state[t_key], None) if unit_val == "ns" else None
    H_real_val = _safe_float(st.session_state[h_key], None)
    style_val = st.session_state[style_key]
    trend_val = int(st.session_state[trend_key])
    gap_thr_ns = _safe_float(st.session_state[gap_key], 0.5)

    # conversion seuil ns -> unité de travail (ns ou px)
    if unit_val == "ns":
        gap_thr_work = float(gap_thr_ns)
        gap_unit_label = "ns"
    else:
        # y(px) <-> ns via y_px = ns / T_total_ns * H_real_px
        if (T_total_ns is None) or (H_real_val is None) or (T_total_ns <= 0) or (H_real_val <= 0):
            gap_thr_work = float("nan")
        else:
            gap_thr_work = float(gap_thr_ns) / float(T_total_ns) * float(H_real_val)
        gap_unit_label = "px"

    if np.isnan(gap_thr_work):
        st.error("Seuil de gap non calculable (T_ns ou H_real invalide).")
        return

    # -------------------------------------------------------------
    # Conversion + Fusion (matching avec seuil, complétion + moyenne si proche)
    # -------------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def _convert_and_fuse(dfA, dfD, unit_val, T_ns_val, H_real_val, gap_thr_work):
        """
        Convertit ASC/DESC puis fusionne avec matching par proximité (seuil):
          - si |ASC - DESC| <= gap_thr_work : on moyenne (fusion)
          - sinon : on conserve les deux détections séparément (comme si l'autre côté = NaN).
        => Résultat: les "manquants" sont complétés par l'autre, et quand les deux sont proches on moyenne.
        Retour: fused_df, iface_cols_out, fusion_report_df
        """
        A = _ensure_xm(_convert_from_raw(dfA, unit_val, T_ns_val, H_real_val))
        D = _ensure_xm(_convert_from_raw(dfD, unit_val, T_ns_val, H_real_val))

        def _iface_id(c):
            try:
                return int(c.split("_")[-1])
            except Exception:
                return 0

        # Union des colonnes interface_* en entrée
        iface_cols_in = sorted(
            list(set([c for c in A.columns if c.startswith("interface_")] +
                     [c for c in D.columns if c.startswith("interface_")])),
            key=_iface_id
        )
        if not iface_cols_in:
            return pd.DataFrame(), [], pd.DataFrame()

        def _prep(df):
            g = df[["x_m"] + [c for c in iface_cols_in if c in df.columns]].copy()
            for c in iface_cols_in:
                if c not in g.columns:
                    g[c] = np.nan
            g["x_key"] = g["x_m"].astype(float).round(3)
            agg = {c: "mean" for c in iface_cols_in}
            agg["x_m"] = "mean"
            return g.groupby("x_key", as_index=False).agg(agg)

        A1, D1 = _prep(A), _prep(D)
        x_union = pd.DataFrame({"x_key": sorted(set(A1["x_key"]) | set(D1["x_key"]))})
        Aj = pd.merge(x_union, A1, on="x_key", how="left")
        Dj = pd.merge(x_union, D1, on="x_key", how="left")

        n = len(x_union)
        fused = pd.DataFrame({"x_key": x_union["x_key"]})
        fused["x_m"] = np.nanmean(np.vstack([Aj["x_m"].to_numpy(), Dj["x_m"].to_numpy()]), axis=0)

        A_mat = Aj[iface_cols_in].to_numpy(dtype=float)
        D_mat = Dj[iface_cols_in].to_numpy(dtype=float)

        # Nombre max de valeurs en sortie = (#ASC + #DESC) au pire (on conserve aussi les non-matchées)
        cntA = np.sum(~np.isnan(A_mat), axis=1)
        cntD = np.sum(~np.isnan(D_mat), axis=1)
        K = int(np.max(cntA + cntD)) if n > 0 else 0
        K = max(K, 1)

        iface_cols_out = [f"interface_{i}" for i in range(1, K + 1)]
        F_mat = np.full((n, K), np.nan, dtype=float)

        report_rows = []

        for r in range(n):
            a = A_mat[r, :]
            d = D_mat[r, :]

            A_idx = [i for i in range(a.shape[0]) if not np.isnan(a[i])]
            D_idx = [j for j in range(d.shape[0]) if not np.isnan(d[j])]

            used_A = set()
            used_D = set()

            # --- matching greedy : associer les plus proches d'abord, sous le seuil ---
            candidates = []
            for i in A_idx:
                for j in D_idx:
                    gap = abs(a[i] - d[j])
                    candidates.append((gap, i, j))
            candidates.sort(key=lambda t: t[0])

            matched_pairs = []
            for gap, i, j in candidates:
                if gap > gap_thr_work:
                    break
                if i in used_A or j in used_D:
                    continue
                used_A.add(i)
                used_D.add(j)
                matched_pairs.append((i, j, gap))

            # valeurs de sortie :
            # - moyennes pour paires matchées
            # - + toutes les non-matchées (ASC et DESC) gardées telles quelles
            fused_vals = []

            for i, j, gap in matched_pairs:
                fused_vals.append(0.5 * (a[i] + d[j]))
                report_rows.append({
                    "x_m": float(fused.loc[r, "x_m"]) if not np.isnan(fused.loc[r, "x_m"]) else np.nan,
                    "x_key": float(fused.loc[r, "x_key"]),
                    "type": "FUSED_OK",
                    "asc_val": float(a[i]),
                    "desc_val": float(d[j]),
                    "gap": float(gap)
                })

            for i in A_idx:
                if i not in used_A:
                    fused_vals.append(float(a[i]))
                    best_gap = float(np.min([abs(a[i] - d[j]) for j in D_idx])) if D_idx else np.nan
                    report_rows.append({
                        "x_m": float(fused.loc[r, "x_m"]) if not np.isnan(fused.loc[r, "x_m"]) else np.nan,
                        "x_key": float(fused.loc[r, "x_key"]),
                        "type": "ASC_ONLY",
                        "asc_val": float(a[i]),
                        "desc_val": np.nan,
                        "gap": best_gap
                    })

            for j in D_idx:
                if j not in used_D:
                    fused_vals.append(float(d[j]))
                    best_gap = float(np.min([abs(d[j] - a[i]) for i in A_idx])) if A_idx else np.nan
                    report_rows.append({
                        "x_m": float(fused.loc[r, "x_m"]) if not np.isnan(fused.loc[r, "x_m"]) else np.nan,
                        "x_key": float(fused.loc[r, "x_key"]),
                        "type": "DESC_ONLY",
                        "asc_val": np.nan,
                        "desc_val": float(d[j]),
                        "gap": best_gap
                    })

            # ordre vertical : on trie par valeur (temps/profondeur) puis on remplit interface_1..K
            fused_vals = sorted(fused_vals)
            for k in range(min(len(fused_vals), K)):
                F_mat[r, k] = fused_vals[k]

        for k, c in enumerate(iface_cols_out):
            fused[c] = F_mat[:, k]

        report = pd.DataFrame(report_rows)
        return fused.sort_values("x_m").reset_index(drop=True), iface_cols_out, report

    fused, iface_cols, fusion_report = _convert_and_fuse(
        df_asc_raw, df_desc_raw, unit_val, T_ns_val, H_real_val, float(gap_thr_work)
    )
    if fused.empty:
        st.error("Fusion impossible (données vides après conversion).")
        return

    st.caption(
        f"Règle : fusion (= moyenne) si |ASC - DESC| ≤ {gap_thr_ns:.2f} ns "
        f"(≈ {gap_thr_work:.3f} {gap_unit_label} en unité de travail)."
    )

    with st.expander("Infos supplémentaires (paires fusionnées / valeurs seules)"):
        if fusion_report is None or fusion_report.empty:
            st.write("Aucune info à afficher.")
        else:
            rr = fusion_report.copy()
            rr["gap_unit"] = gap_unit_label
            st.dataframe(rr, use_container_width=True)

    # -------------------------------------------------------------
    # Bornes et tronçonnage
    # -------------------------------------------------------------
    xmin_all, xmax_all = _xcm_minmax(fused) or (0.0, 100.0)
    xmin_key, xmax_key = "xmin_fuse_m", "xmax_fuse_m"

    for k, v in [(xmin_key, xmin_all), (xmax_key, xmax_all)]:
        if k not in st.session_state:
            st.session_state[k] = float(v)

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
        title="Fusion ASC + DESC (matching + moyenne si proche)",
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
                "total_time_ns": T_total_ns,
                "real_height_px": H_real_val,
                "xmin_m": float(xmin_sel),
                "xmax_m": float(xmax_sel),
                "gap_thr_ns": float(gap_thr_ns),
            }
            _export_excel(df_to_save, unit_val, out_path, meta)
            st.success(f"Export ok: {out_path}")

    with c4:
        if st.button("Exporter toute la stratigraphie (FUSION)"):
            df_all_to_save = _rename_interfaces_for_unit(fused, unit_val)
            out_path = os.path.join(out_dir, f"stratigraphy_FUSION_full_{unit_val}.xlsx")
            meta = {
                "unit": unit_val,
                "total_time_ns": T_total_ns,
                "real_height_px": H_real_val,
                "gap_thr_ns": float(gap_thr_ns),
            }
            _export_excel(df_all_to_save, unit_val, out_path, meta)
            st.success(f"Export ok: {out_path}")

    # Sauvegardes dans la session
    st.session_state["df_strat_FUSION"] = fused_view.copy()
    st.session_state["fusion_report"] = fusion_report.copy() if isinstance(fusion_report, pd.DataFrame) else pd.DataFrame()
