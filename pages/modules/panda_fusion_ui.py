import io
import numpy as np
import pandas as pd
import streamlit as st

from .panda_upsert_ui import _read_excel_with_sheet, _safe_selectbox, _coerce_numeric, _maybe_round_pos


def fusion_block():
    st.header("Fusion data Panda + GPR")
    st.markdown(
        "Sortie : `position_cm, interface_1, interface_2, Eps_1, Eps_2, Ind_1, Ind_2, temps_1, temps_2` (positions communes)."
    )

    # === A) Source Panda
    source = st.radio(
        "Source Panda",
        ["📂 Charger un fichier Panda", "⚡ Utiliser l’UPsert (onglet 1)"],
        index=(1 if "panda_upsert_df" in st.session_state else 0),
        horizontal=True
    )

    df_panda_fus, cols_panda_fus = pd.DataFrame(), []
    if source == "⚡ Utiliser l’UPsert (onglet 1)" and "panda_upsert_df" in st.session_state:
        df_panda_fus = st.session_state["panda_upsert_df"].copy()
        st.success(f"Panda (UPsert) : {len(df_panda_fus)} lignes • {len(df_panda_fus.columns)} colonnes")
        with st.expander("Aperçu Panda (UPsert)"):
            st.dataframe(df_panda_fus.head(200), use_container_width=True)
        cols_panda_fus = list(df_panda_fus.columns)
    else:
        st.subheader("A. Charger **Panda**")
        upl_panda2 = st.file_uploader("Excel Panda (.xlsx/.xls)", type=["xlsx", "xls"], key="upl_panda_fus")
        if upl_panda2 is not None:
            df_panda_fus = _read_excel_with_sheet(upl_panda2, "Panda_Fusion")
            if not df_panda_fus.empty:
                st.success(f"Panda: {len(df_panda_fus)} lignes • {len(df_panda_fus.columns)} colonnes")
                with st.expander("Aperçu Panda"):
                    st.dataframe(df_panda_fus.head(200), use_container_width=True)
                cols_panda_fus = list(df_panda_fus.columns)

    def _sel(label, opts, defs, key):
        return _safe_selectbox(label, opts, defs, key)

    # Colonnes Panda
    pos_p = _sel("position_cm (Panda)", cols_panda_fus, ["position_cm", "x_cm", "position", "pos_cm"], "pos_panda_f")
    optsP = [c for c in (cols_panda_fus or []) if c != pos_p]
    i1_p = _sel("interface_1 (Panda)", optsP, ["interface_1", "interface1"], "i1_p")
    i2_p = _sel("interface_2 (Panda)", optsP, ["interface_2", "interface2"], "i2_p")
    e1_p = _sel("Eps_1 (Panda)", optsP, ["Eps_1", "eps_1", "epsilon_1"], "e1_p")
    e2_p = _sel("Eps_2 (Panda)", optsP, ["Eps_2", "eps_2", "epsilon_2"], "e2_p")
    ind1_p = _sel("Ind_1 (Panda)", optsP, ["Ind_1", "ind_1", "indicator_1"], "ind1_p")
    ind2_p = _sel("Ind_2 (Panda)", optsP, ["Ind_2", "ind_2", "indicator_2"], "ind2_p")

    # === B) GPR
    st.markdown("---")
    st.subheader("B. Charger **GPR**")
    upl_gpr = st.file_uploader("Excel GPR (.xlsx/.xls)", type=["xlsx", "xls"], key="upl_gpr")
    dfG, colsG = pd.DataFrame(), []
    if upl_gpr is not None:
        dfG = _read_excel_with_sheet(upl_gpr, "GPR_Fusion")
        if not dfG.empty:
            st.success(f"GPR: {len(dfG)} lignes • {len(dfG.columns)} colonnes")
            with st.expander("Aperçu GPR"):
                st.dataframe(dfG.head(200), use_container_width=True)
            colsG = list(dfG.columns)

    pos_g = _sel("position_cm (GPR)", colsG, ["position_cm", "x_cm", "position", "pos_cm"], "pos_g")
    optsG = [c for c in (colsG or []) if c != pos_g]
    t1_g = _sel("temps_1 (GPR)", optsG, ["temps_1", "time_1", "t1"], "t1_g")
    t2_g = _sel("temps_2 (GPR)", optsG, ["temps_2", "time_2", "t2"], "t2_g")

    # === C) Fusion & Export
    st.markdown("---")
    st.subheader("C. Fusion & Export")

    def _merge_panda_gpr(P: pd.DataFrame,
                         G: pd.DataFrame,
                         pos_p: str, pos_g: str,
                         i1: str, i2: str,
                         e1: str, e2: str,
                         ind1: str, ind2: str,
                         t1: str, t2: str) -> pd.DataFrame:
        # Colonnes Panda
        base_cols = [pos_p, i1, i2]
        p = P[base_cols].copy()
        p["Eps_1"] = _coerce_numeric(P[e1]) if e1 and e1 in P.columns else np.nan
        p["Eps_2"] = _coerce_numeric(P[e2]) if e2 and e2 in P.columns else np.nan
        p["Ind_1"] = _coerce_numeric(P[ind1]) if ind1 and ind1 in P.columns else np.nan
        p["Ind_2"] = _coerce_numeric(P[ind2]) if ind2 and ind2 in P.columns else np.nan

        # Colonnes GPR
        g = G[[pos_g, t1, t2]].copy()

        # Normalisation noms
        p.columns = ["position_cm", "interface_1", "interface_2", "Eps_1", "Eps_2", "Ind_1", "Ind_2"]
        g.columns = ["position_cm", "temps_1", "temps_2"]

        # Typage numeric & nettoyage
        p["position_cm"] = _coerce_numeric(p["position_cm"])
        g["position_cm"] = _coerce_numeric(g["position_cm"])
        for c in ["interface_1", "interface_2", "Eps_1", "Eps_2", "Ind_1", "Ind_2"]:
            p[c] = _coerce_numeric(p[c])
        for c in ["temps_1", "temps_2"]:
            g[c] = _coerce_numeric(g[c])

        p = p.dropna(subset=["position_cm"])
        g = g.dropna(subset=["position_cm"])
        p["position_cm"] = _maybe_round_pos(p["position_cm"])
        g["position_cm"] = _maybe_round_pos(g["position_cm"])

        p = p.sort_values("position_cm").drop_duplicates("position_cm", keep="first")
        g = g.sort_values("position_cm").drop_duplicates("position_cm", keep="first")

        out = pd.merge(p, g, on="position_cm", how="inner")
        out = out[["position_cm", "interface_1", "interface_2", "Eps_1", "Eps_2", "Ind_1", "Ind_2", "temps_1", "temps_2"]]
        return out.sort_values("position_cm").reset_index(drop=True)

    if st.button("➡️ Fusionner (Panda ∩ GPR) et générer l'Excel", key="btn_fuse"):
        if df_panda_fus.empty or dfG.empty or not all([pos_p, i1_p, i2_p, pos_g, t1_g, t2_g]):
            st.error("Charge Panda & GPR et sélectionne toutes les colonnes requises (Eps/Indicateurs optionnels).")
        else:
            try:
                fused = _merge_panda_gpr(df_panda_fus, dfG, pos_p, pos_g, i1_p, i2_p, e1_p, e2_p, ind1_p, ind2_p, t1_g, t2_g)
            except Exception as e:
                st.error(f"Erreur fusion : {e}")
                fused = pd.DataFrame()

            if fused.empty:
                st.warning("Aucune position commune trouvée.")
            else:
                st.success(f"Fusion OK : {len(fused)} lignes.")
                with st.expander("Aperçu"):
                    st.dataframe(fused.head(200), use_container_width=True)

                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine="openpyxl") as w:
                    fused.to_excel(w, sheet_name="fusion_panda_gpr", index=False)
                bio.seek(0)
                st.download_button("💾 Télécharger l'Excel fusionné",
                                   data=bio,
                                   file_name="fusion_panda_gpr.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
