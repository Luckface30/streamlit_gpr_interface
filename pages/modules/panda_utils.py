import io
import numpy as np
import pandas as pd
import streamlit as st

from core.indicateur import (
    get_eps1, get_eps2,
    sample_eps1_with_indicator, sample_eps2_with_indicator
)
from .panda_upsert_ui import (
    _read_excel_with_sheet, _safe_selectbox,
    _coerce_numeric, _maybe_round_pos,
    best_match_col
)


def upsert_block():
    st.header("UPsert : mapping manuel/auto des colonnes (Cote touche, Eps, Indicateurs, etc.)")

    # === A) Panda
    st.subheader("A. Charger **Panda**")
    upl_panda = st.file_uploader("Excel Panda (.xlsx/.xls)", type=["xlsx", "xls"], key="upl_panda_upsert")
    dfP, colsP = pd.DataFrame(), []
    if upl_panda is not None:
        dfP = _read_excel_with_sheet(upl_panda, "Panda")
        if not dfP.empty:
            st.success(f"Panda: {len(dfP)} lignes • {len(dfP.columns)} colonnes")
            with st.expander("Aperçu Panda"):
                st.dataframe(dfP.head(200), use_container_width=True)
            colsP = list(dfP.columns)

    pos_p = _safe_selectbox("Colonne de **position (cm)** → Panda", colsP,
                            ["position_cm", "x_cm", "position", "pos_cm"], "pos_col_p")

    # === B) Vérités
    st.subheader("B. Charger **Vérités terrain**")
    upl_T = st.file_uploader("Excel Vérités (.xlsx/.xls)", type=["xlsx", "xls"], key="upl_truth_upsert")
    dfT, colsT = pd.DataFrame(), []
    if upl_T is not None:
        dfT = _read_excel_with_sheet(upl_T, "Vérités")
        if not dfT.empty:
            st.success(f"Vérités: {len(dfT)} lignes • {len(dfT.columns)} colonnes")
            with st.expander("Aperçu Vérités (brut)"):
                st.dataframe(dfT.head(200), use_container_width=True)
            colsT = list(dfT.columns)

    pos_t = _safe_selectbox("Colonne de **position (cm)** → Vérités", colsT,
                            [pos_p, "position_cm", "x_cm", "position", "pos_cm"], "pos_col_t")

    # === C) Epsilons + Indicateurs dans Vérités
    st.subheader("C. Epsilons + Indicateurs dans Vérités (si absents) + aperçu")
    optsT_no_pos = [c for c in (colsT or []) if c != pos_t]
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        hum_bs_col = _safe_selectbox("Humidité BS (Vérités)", optsT_no_pos,
                                     ["humidite_bs", "hum_bs", "bs_hum"], "hum_bs_col_v")
    with col2:
        hum_bc_col = _safe_selectbox("Humidité BC (Vérités)", optsT_no_pos,
                                     ["humidite_bc", "hum_bc", "bc_hum"], "hum_bc_col_v")
    with col3:
        colmat_col = _safe_selectbox("Colmatage (Vérités)", optsT_no_pos,
                                     ["colmatage", "colm", "natureducolmatage"], "colmat_col_v")
    with col4:
        obs_col = _safe_selectbox("Observations (Vérités)", optsT_no_pos,
                                  ["obs", "observation", "observations"], "obs_col_v")

    compute_eps = st.checkbox("Compléter `Eps_1`, `Eps_2`, `Ind_1`, `Ind_2` si absents", value=True)

    T_enriched = pd.DataFrame()
    if not dfT.empty and pos_t:
        T_enriched = dfT.copy()
        if compute_eps:
            def _val(r, c): return r.get(c) if (c and c in r.index) else None

            if "Eps_1" not in T_enriched.columns:
                T_enriched[["Eps_1", "Ind_1"]] = T_enriched.apply(
                    lambda r: pd.Series(sample_eps1_with_indicator(_val(r, hum_bs_col), _val(r, obs_col))),
                    axis=1
                )
            if "Eps_2" not in T_enriched.columns:
                T_enriched[["Eps_2", "Ind_2"]] = T_enriched.apply(
                    lambda r: pd.Series(sample_eps2_with_indicator(
                        _val(r, hum_bc_col), _val(r, colmat_col), _val(r, obs_col)
                    )),
                    axis=1
                )

        st.info("Aperçu — Vérités enrichies (avec Eps + Indicateurs) :")
        with st.expander("Voir les Vérités enrichies"):
            st.dataframe(T_enriched.head(200), use_container_width=True)

    # === D) Mapping colonnes Vérités -> Panda
    st.subheader("D. Correspondances de colonnes (Vérités ⟶ Panda)")
    mapped_cols = {}
    colsP_no_pos = [c for c in (colsP or []) if c != pos_p]
    colsT_enriched = list(T_enriched.columns) if not T_enriched.empty else (colsT or [])

    defaults_to_update = []
    for cand in ["Eps_1", "Eps_2", "Ind_1", "Ind_2",
                 "interface_1", "interface_2",
                 "Ballast Cote touche (m)", "Cote touche", "Cote touche (m)"]:
        if cand in colsP_no_pos:
            defaults_to_update.append(cand)
    commons = [c for c in colsP_no_pos if c in colsT_enriched]
    defaults_to_update += [c for c in commons if c not in defaults_to_update]

    cols_panda_to_update = st.multiselect(
        "Colonnes Panda à alimenter/remplacer depuis Vérités",
        options=colsP_no_pos,
        default=defaults_to_update,
    )

    if cols_panda_to_update:
        st.caption("Pour chaque colonne Panda sélectionnée, choisis la colonne correspondante côté Vérités.")
        truth_choices = [c for c in colsT_enriched if c != pos_t]
        for pc in cols_panda_to_update:
            auto = best_match_col(pc, truth_choices)
            options = ["<aucune>"] + truth_choices
            idx = options.index(auto) if auto in options else 0
            mapped_cols[pc] = st.selectbox(f"Vérités → {pc}", options=options, index=idx, key=f"map_{pc}")
        clean_map = {pc: vc for pc, vc in mapped_cols.items() if vc and vc != "<aucune>"}
        if clean_map:
            st.success(f"Mapping utilisé: {clean_map}")
        else:
            st.warning("Aucun mapping valide sélectionné pour l’instant.")

    # === E) Options d'UPsert
    st.markdown("---")
    st.subheader("E. Options d’UPsert")
    round_positions = st.checkbox("Arrondir les positions quasi entières avant la jointure", value=False)
    prefer_nulls = st.checkbox("Remplacer aussi par valeurs vides de Vérités (sinon garder Panda si NaN)", value=False)
    dedup_truth = st.selectbox("Si Vérités a plusieurs lignes à la même position :",
                               ["Dernière valeur non nulle", "Première valeur non nulle", "Moyenne numérique"], index=0)

    # === F) Lancer
    st.markdown("---")
    st.subheader("F. Exécuter l’UPsert (avec mapping)")
    if st.button("▶️ Lancer", key="btn_upsert"):
        if dfP.empty or dfT.empty or not pos_p or not pos_t:
            st.error("Charge Panda & Vérités et choisis les colonnes de position (Panda et Vérités).")
            st.stop()

        P = dfP.copy()
        T = (T_enriched if not T_enriched.empty else dfT).copy()

        # 1) Mapping → renommer Vérités vers Panda
        rename_map = {}
        for panda_col, verites_col in (mapped_cols or {}).items():
            if verites_col and verites_col != "<aucune>" and verites_col in T.columns:
                rename_map[verites_col] = panda_col
        if rename_map:
            T = T.rename(columns=rename_map)

        # 2) Clés numériques
        P["_key_"] = _maybe_round_pos(P[pos_p]) if round_positions else _coerce_numeric(P[pos_p])
        T["_key_"] = _maybe_round_pos(T[pos_t]) if round_positions else _coerce_numeric(T[pos_t])

        # 3) Dédup Vérités
        T = T.dropna(subset=["_key_"])

        def _last_valid(s):
            s2 = s.dropna()
            return s2.iloc[-1] if len(s2) else np.nan

        def _first_valid(s):
            s2 = s.dropna()
            return s2.iloc[0] if len(s2) else np.nan

        if dedup_truth == "Dernière valeur non nulle":
            T = T.groupby("_key_", as_index=False).agg(_last_valid)
        elif dedup_truth == "Première valeur non nulle":
            T = T.groupby("_key_", as_index=False).agg(_first_valid)
        else:
            num_cols = [c for c in T.columns if c != "_key_" and pd.api.types.is_numeric_dtype(T[c])]
            agg = {c: "mean" for c in num_cols}
            for c in T.columns:
                if c not in num_cols and c != "_key_":
                    agg[c] = _last_valid
            T = T.groupby("_key_", as_index=False).agg(agg)

        # 4) Forcer schéma Panda (pas d'union de colonnes)
        panda_schema = [c for c in P.columns if c != "_key_"]
        keep_T_cols = set(["_key_", pos_t]) | set(panda_schema)
        T = T[[c for c in T.columns if c in keep_T_cols]].copy()

        # 5) Index pour UPsert
        P.set_index("_key_", inplace=True)
        T.set_index("_key_", inplace=True)

        # 6) Colonnes à remplacer
        if cols_panda_to_update:
            cols_sel = [c for c in cols_panda_to_update if c in P.columns and c in T.columns and c != pos_p]
        else:
            cols_sel = [c for c in P.columns if c in T.columns and c != pos_p]

        # 7) Remplacement
        inter = P.index.intersection(T.index)
        if prefer_nulls:
            if cols_sel:
                P.loc[inter, cols_sel] = T.loc[inter, cols_sel]
        else:
            for c in cols_sel:
                mask = T.loc[inter, c].notna()
                idx = mask[mask].index
                if len(idx) > 0:
                    P.loc[idx, c] = T.loc[idx, c]

        # 8) Ajout nouvelles positions (alignées sur schéma Panda)
        only_T = T.index.difference(P.index)
        if len(only_T) > 0:
            new_rows = T.loc[only_T].copy()
            if pos_p not in new_rows.columns:
                new_rows[pos_p] = new_rows.index
            else:
                new_rows[pos_p] = new_rows[pos_p].where(~new_rows[pos_p].isna(), new_rows.index)
            new_rows = new_rows.reindex(columns=P.columns, fill_value=np.nan)
            P = pd.concat([P, new_rows], axis=0, ignore_index=False)

        # 9) Finalisation
        P.reset_index(drop=False, inplace=True)  # _key_
        if pos_p not in P.columns:
            P[pos_p] = P["_key_"]
        else:
            P[pos_p] = P[pos_p].where(~P[pos_p].isna(), P["_key_"])
        P.drop(columns=["_key_"], inplace=True, errors="ignore")

        panda_order = [c for c in dfP.columns if c in P.columns and c != pos_p]
        others = [c for c in P.columns if c not in panda_order and c != pos_p]
        P = P[[pos_p] + panda_order + others]

        try:
            P[pos_p] = _coerce_numeric(P[pos_p])
            P.sort_values(by=pos_p, inplace=True)
        except Exception:
            pass

        st.success("UPsert + mapping terminé ✅ — schéma strictement identique à Panda.")
        with st.expander("Aperçu résultat"):
            st.dataframe(P.head(200), use_container_width=True)

        st.session_state["panda_upsert_df"] = P

        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as w:
            P.to_excel(w, sheet_name="panda_upsert", index=False)
        bio.seek(0)
        st.download_button("💾 Télécharger l'Excel fusionné (UPsert + mapping)",
                           data=bio,
                           file_name="panda_upsert.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
