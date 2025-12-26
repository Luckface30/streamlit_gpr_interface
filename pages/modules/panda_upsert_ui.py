# pages/modules/panda_upsert_ui.py
import io
import difflib
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

# >>> Tous les calculs Eps/Ind passent par core.indicateur <<<
from core.indicateur import (
    sample_eps1_with_indicator,
    sample_eps2_with_indicator,
)

# ------------------------- utils lecture & base -------------------------
def _read_excel_with_sheet(upl, label_prefix: str) -> pd.DataFrame:
    try:
        xls = pd.ExcelFile(upl)
        sheet = st.selectbox(f"{label_prefix} • Feuille", xls.sheet_names, index=0, key=f"{label_prefix}_sheet")
        return pd.read_excel(xls, sheet_name=sheet)
    except Exception as e:
        st.error(f"Erreur de lecture Excel ({label_prefix}) : {e}")
        return pd.DataFrame()

def _coerce_numeric(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")

def _maybe_round_pos(s: pd.Series) -> pd.Series:
    s = _coerce_numeric(s)
    if s.empty:
        return s
    try:
        if np.nanmax(np.abs(s - np.rint(s))) < 1e-6:
            return np.rint(s).astype(float)
    except ValueError:
        pass
    return s.astype(float)

# ---------- conversions sûres (corrige pd.Index "truth value is ambiguous") ----------
def _to_list(options):
    if options is None:
        return []
    try:
        return list(options)
    except Exception:
        return []

def _safe_selectbox(label, options, defaults, key):
    opts_raw = _to_list(options)
    opts = [o for o in opts_raw if o not in (None, "")]
    if len(opts) == 0:
        st.selectbox(label, ["<aucune>"], index=0, key=key)
        return None
    def_norms = [str(d).lower() for d in _to_list(defaults) if d is not None]
    guess = next((c for c in opts if str(c).lower() in def_norms), opts[0])
    idx = opts.index(guess) if guess in opts else 0
    return st.selectbox(label, opts, index=idx, key=key)

def best_match_col(target, options):
    def strip_accents(t):
        return "".join(c for c in unicodedata.normalize("NFKD", str(t)) if not unicodedata.combining(c))
    def norm(t):
        return strip_accents(t).lower().replace("_", " ").strip()

    if target is None:
        return None
    opts = [o for o in _to_list(options) if o is not None and str(o).strip() != ""]
    if len(opts) == 0:
        return None

    n_target = norm(target)
    ratios = [(c, difflib.SequenceMatcher(None, n_target, norm(c)).ratio()) for c in opts]
    ratios.sort(key=lambda t: t[1], reverse=True)
    return ratios[0][0] if ratios else None

def _last_valid(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[-1] if not s2.empty else np.nan

# ------------------------- interpolation Vérités -------------------------
def interpolate_verites_windows(
    df: pd.DataFrame,
    pos_col: str,
    chosen_num_cols: list,
    chosen_non_cols: list,
    step: int = 10,
    half_window: int = 200,
) -> pd.DataFrame:
    """
    Grille = union des fenêtres [p-200, p+200] pour chaque position p (pas 10 cm).
    Numériques: interpolation linéaire dans la fenêtre.
    Textes: ffill/bfill dans la fenêtre.
    Recouvrements: moyenne (num) / dernière valeur (texte).
    Aucune extrapolation au-delà des fenêtres.
    """
    if df.empty or pos_col not in df.columns:
        return df

    pos_clean = pd.to_numeric(df[pos_col], errors="coerce").dropna()
    if pos_clean.empty:
        return df
    base_positions = np.unique(np.rint(pos_clean).astype(int))

    work = df.copy()
    work[pos_col] = pd.to_numeric(work[pos_col], errors="coerce")
    work = work.dropna(subset=[pos_col]).set_index(pos_col).sort_index()

    for c in chosen_num_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    frames = []
    for p in base_positions:
        window_index = np.arange(p - half_window, p + half_window + 1, step)

        if chosen_num_cols:
            w_num = work[chosen_num_cols].reindex(window_index).interpolate(
                method="index", limit_direction="both"
            )
        else:
            w_num = pd.DataFrame(index=window_index)

        if chosen_non_cols:
            w_txt = work[chosen_non_cols].reindex(window_index).ffill().bfill()
        else:
            w_txt = pd.DataFrame(index=window_index)

        frames.append(pd.concat([w_num, w_txt], axis=1))

    if not frames:
        return df

    union_df = pd.concat(frames, axis=0)

    agg = {c: "mean" for c in chosen_num_cols if c in union_df.columns}
    for c in chosen_non_cols:
        if c in union_df.columns:
            agg[c] = _last_valid

    out = union_df.groupby(level=0).agg(agg).reset_index().rename(columns={"index": pos_col, "level_0": pos_col})
    keep_order = [pos_col] + [c for c in chosen_num_cols if c in out.columns] + [c for c in chosen_non_cols if c in out.columns]
    return out.reindex(columns=keep_order)

# ------------------------- enrichissement Eps/Ind via core.indicateur -------------------------
def _compute_eps_ind_1(hum_bs, obs):
    try:
        eps, ind = sample_eps1_with_indicator(hum_bs, obs)
        return (np.nan if eps is None else eps, np.nan if ind is None else ind)
    except Exception:
        return (np.nan, np.nan)

def _compute_eps_ind_2(hum_bc, colm, obs):
    try:
        eps, ind = sample_eps2_with_indicator(hum_bc, colm, obs)
        return (np.nan if eps is None else eps, np.nan if ind is None else ind)
    except Exception:
        return (np.nan, np.nan)

def enrich_with_eps_ind(df: pd.DataFrame,
                        hum_bs_col: str, hum_bc_col: str, colm_col: str, obs_col: str,
                        recalc_mode: str = "auto") -> pd.DataFrame:
    """
    Complète/recalcule Eps/Ind uniquement via core.indicateur.
    recalc_mode:
      - "auto"  : calcule quand manquant; si Eps existe mais Ind manque, calcule Ind (et garde Eps existant).
      - "force" : recalcule Eps & Ind quand les entrées sont disponibles, même si des valeurs existent déjà.
      - "keep"  : ne touche à rien si Eps/Ind existent; ne calcule que les 2 absents.
    """
    df = df.copy()
    cols = df.columns

    def _v(row, col):
        return row.get(col) if (col and col in cols) else None

    has_eps1 = "Eps_1" in cols
    has_ind1 = "Ind_1" in cols
    has_eps2 = "Eps_2" in cols
    has_ind2 = "Ind_2" in cols

    out_eps1, out_ind1, out_eps2, out_ind2 = [], [], [], []

    for _, r in df.iterrows():
        hum_bs = _v(r, hum_bs_col)
        hum_bc = _v(r, hum_bc_col)
        colm   = _v(r, colm_col)
        obs    = _v(r, obs_col)

        c1_eps, c1_ind = _compute_eps_ind_1(hum_bs, obs)
        c2_eps, c2_ind = _compute_eps_ind_2(hum_bc, colm, obs)

        # Colonne 1
        cur_eps1 = r.get("Eps_1") if has_eps1 else np.nan
        cur_ind1 = r.get("Ind_1") if has_ind1 else np.nan

        if recalc_mode == "force":
            eps1 = c1_eps if not np.isnan(c1_eps) else cur_eps1
            ind1 = c1_ind if not np.isnan(c1_ind) else cur_ind1
        elif recalc_mode == "keep":
            eps1 = cur_eps1 if not pd.isna(cur_eps1) else c1_eps
            ind1 = cur_ind1 if not pd.isna(cur_ind1) else c1_ind
        else:  # auto
            if pd.isna(cur_eps1) and pd.isna(cur_ind1):
                eps1, ind1 = c1_eps, c1_ind
            elif pd.isna(cur_eps1) and not pd.isna(cur_ind1):
                eps1, ind1 = c1_eps, cur_ind1
            elif not pd.isna(cur_eps1) and pd.isna(cur_ind1):
                eps1, ind1 = cur_eps1, c1_ind
            else:
                eps1, ind1 = cur_eps1, cur_ind1

        out_eps1.append(eps1)
        out_ind1.append(ind1)

        # Colonne 2
        cur_eps2 = r.get("Eps_2") if has_eps2 else np.nan
        cur_ind2 = r.get("Ind_2") if has_ind2 else np.nan

        if recalc_mode == "force":
            eps2 = c2_eps if not np.isnan(c2_eps) else cur_eps2
            ind2 = c2_ind if not np.isnan(c2_ind) else cur_ind2
        elif recalc_mode == "keep":
            eps2 = cur_eps2 if not pd.isna(cur_eps2) else c2_eps
            ind2 = cur_ind2 if not pd.isna(cur_ind2) else c2_ind
        else:  # auto
            if pd.isna(cur_eps2) and pd.isna(cur_ind2):
                eps2, ind2 = c2_eps, c2_ind
            elif pd.isna(cur_eps2) and not pd.isna(cur_ind2):
                eps2, ind2 = c2_eps, cur_ind2
            elif not pd.isna(cur_eps2) and pd.isna(cur_ind2):
                eps2, ind2 = cur_eps2, c2_ind
            else:
                eps2, ind2 = cur_eps2, cur_ind2

        out_eps2.append(eps2)
        out_ind2.append(ind2)

    df["Eps_1"] = out_eps1
    df["Ind_1"] = out_ind1
    df["Eps_2"] = out_eps2
    df["Ind_2"] = out_ind2
    return df

# ------------------------- UPsert principal -------------------------
def upsert_block():
    st.header("UPsert : mapping Panda ↔ Vérités + calcul ε/Ind (via core.indicateur)")

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

    # === C) Interpolation + Eps/Ind via indicateur
    st.subheader("C. Interpolation des Vérités (10 cm, ±2 m) + ε / Ind (core.indicateur)")

    recalc_choice = st.radio(
        "Politique de calcul Eps/Ind (toujours via core.indicateur) :",
        ["Auto (compléter uniquement)", "Forcer le recalcul", "Conserver ce qui existe"],
        horizontal=True,
        index=0,
    )
    recalc_mode = {"Auto (compléter uniquement)": "auto",
                   "Forcer le recalcul": "force",
                   "Conserver ce qui existe": "keep"}[recalc_choice]

    T_enriched = pd.DataFrame()
    if not dfT.empty and pos_t:
        default_num = [c for c in dfT.columns if c != pos_t and pd.api.types.is_numeric_dtype(dfT[c])]
        with st.expander("⚙️ Choisir les colonnes NUMÉRIQUES à interpoler (pas 10 cm)"):
            chosen_num_cols = st.multiselect("Numériques", options=[c for c in dfT.columns if c != pos_t], default=default_num)
        with st.expander("⚙️ Choisir les colonnes TEXTE à dupliquer (ffill/bfill)"):
            chosen_non_cols = st.multiselect("Textes", options=[c for c in dfT.columns if c != pos_t],
                                             default=[c for c in dfT.columns if c != pos_t and c not in default_num])

        do_interp = st.checkbox("👉 Interpoler (pas 10 cm) sur ±2 m autour de CHAQUE point", value=True)
        if do_interp:
            T_interp = interpolate_verites_windows(dfT, pos_t, chosen_num_cols, chosen_non_cols)
        else:
            T_interp = dfT.copy()

        # Colonnes d'entrée pour indicateur
        hum_bs_col = best_match_col("humidite_bs", T_interp.columns)
        hum_bc_col = best_match_col("humidite_bc", T_interp.columns)
        colm_col   = best_match_col("colmatage",   T_interp.columns)
        obs_col    = best_match_col("obs",         T_interp.columns) or best_match_col("observation", T_interp.columns)

        # Calcul/complétion Eps/Ind EXCLUSIVEMENT via core.indicateur
        T_enriched = enrich_with_eps_ind(
            T_interp,
            hum_bs_col=hum_bs_col, hum_bc_col=hum_bc_col, colm_col=colm_col, obs_col=obs_col,
            recalc_mode=recalc_mode,
        )

        st.info("Aperçu — Vérités enrichies (ε/Ind ajoutés ou complétés) :")
        with st.expander("Voir"):
            st.dataframe(T_enriched.head(200), use_container_width=True)

    # === D) Mapping colonnes Vérités -> Panda
    st.subheader("D. Correspondances de colonnes (Vérités ⟶ Panda)")
    mapped_cols = {}
    colsP = list(dfP.columns) if not dfP.empty else []
    colsT_enriched = list(T_enriched.columns) if not T_enriched.empty else (list(dfT.columns) if not dfT.empty else [])

    pos_p = pos_p or "position_cm"
    colsP_no_pos = [c for c in colsP if c != pos_p]

    for target in ["Eps_1", "Eps_2", "Ind_1", "Ind_2",
                   "interface_1", "interface_2",
                   "Ballast Cote touche (m)", "Cote touche", "Cote touche (m)"]:
        src = best_match_col(target, colsT_enriched)
        if src and target in colsP_no_pos:
            mapped_cols[target] = src

    st.caption("Ajuste les correspondances si besoin.")
    for panda_col in colsP_no_pos:
        mapped_cols[panda_col] = st.selectbox(
            f"Vérités → {panda_col}",
            options=["<aucune>"] + colsT_enriched,
            index=(1 + colsT_enriched.index(mapped_cols[panda_col])) if mapped_cols.get(panda_col) in colsT_enriched else 0,
            key=f"map_{panda_col}"
        )

    # === E) Options d’UPsert
    st.subheader("E. Options d’UPsert")
    round_positions = st.checkbox("Arrondir les positions au cm entier pour faire les clés", value=True)
    dedup_truth = st.selectbox("Déduplication côté Vérités (même position)",
                               ["Dernière valeur non nulle", "Première valeur non nulle", "Moyenne"], index=0)
    prefer_nulls = st.checkbox("Remplacer aussi par des NaN (sinon on garde l’ancienne valeur Panda)", value=False)

    # === F) Exécuter
    st.subheader("F. Exécuter l’UPsert (avec mapping)")

    # Option pour ajouter les lignes Vérités absentes de Panda
    add_missing_truths = st.checkbox(
        "Ajouter les lignes des Vérités qui n'existent pas dans Panda",
        value=True,
        help="Si coché, les positions présentes dans Vérités mais absentes de Panda seront ajoutées."
    )

    if st.button("▶️ Lancer", key="btn_upsert"):
        if dfP.empty or dfT.empty or not pos_p or not pos_t:
            st.error("Charge Panda & Vérités et choisis les colonnes de position (Panda et Vérités).")
            st.stop()

        P = dfP.copy()
        # garde aussi les Vérités enrichies en mémoire pour d'autres onglets
        T = (T_enriched if not T_enriched.empty else dfT).copy()
        st.session_state["verites_enriched_df"] = T.copy()

        # 1) Mapping → renommer Vérités vers schéma Panda
        rename_map = {}
        for panda_col, verites_col in (mapped_cols or {}).items():
            if verites_col and verites_col != "<aucune>" and verites_col in T.columns:
                rename_map[verites_col] = panda_col
        if rename_map:
            T = T.rename(columns=rename_map)

        # 2) Clés numériques arrondies (ou non)
        P["_key_"] = _maybe_round_pos(P[pos_p]) if round_positions else _coerce_numeric(P[pos_p])
        T["_key_"] = _maybe_round_pos(T[pos_t]) if round_positions else _coerce_numeric(T[pos_t])

        # 3) Dédup Vérités (au niveau clé)
        T = T.dropna(subset=["_key_"])

        def _last_valid_group(s):
            s2 = s.dropna()
            return s2.iloc[-1] if not s2.empty else np.nan

        def _first_valid_group(s):
            s2 = s.dropna()
            return s2.iloc[0] if not s2.empty else np.nan

        if dedup_truth == "Dernière valeur non nulle":
            T = T.groupby("_key_", as_index=False).agg(_last_valid_group)
        elif dedup_truth == "Première valeur non nulle":
            T = T.groupby("_key_", as_index=False).agg(_first_valid_group)
        else:
            num_cols = [c for c in T.columns if c != "_key_" and pd.api.types.is_numeric_dtype(T[c])]
            agg = {c: "mean" for c in num_cols}
            for c in T.columns:
                if c not in num_cols and c != "_key_":
                    agg[c] = _last_valid
            T = T.groupby("_key_", as_index=False).agg(agg)

        # 4) Forcer schéma Panda (colonnes communes uniquement)
        panda_schema = [c for c in P.columns if c != "_key_"]
        keep_T_cols = set(["_key_"]) | set(panda_schema)
        T = T[[c for c in T.columns if c in keep_T_cols]].copy()

        # 5) Index pour UPsert
        P.set_index("_key_", inplace=True)
        T.set_index("_key_", inplace=True)

        # 6) Intersection & manquants
        inter = P.index.intersection(T.index)
        missing_in_panda = T.index.difference(P.index)   # positions présentes dans Vérités seulement
        missing_in_truth = P.index.difference(T.index)   # positions présentes dans Panda seulement

        # 7) Mise à jour des colonnes sur les positions communes (AUCUN ajout ici)
        cols_panda_to_update = [c for c in panda_schema if c in T.columns and c != pos_p]
        updated_counts = {}
        if len(inter) > 0 and cols_panda_to_update:
            if prefer_nulls:
                P.loc[inter, cols_panda_to_update] = T.loc[inter, cols_panda_to_update]
            else:
                for c in cols_panda_to_update:
                    src = pd.to_numeric(T[c], errors="coerce") if pd.api.types.is_numeric_dtype(P[c]) else T[c]
                    dst = P[c]
                    if pd.api.types.is_numeric_dtype(dst):
                        mask = dst.isna()
                        P.loc[inter, c] = np.where(mask.loc[inter], src.loc[inter], dst.loc[inter])
                    else:
                        mask = dst.isna() | (dst.astype(str).str.strip() == "")
                        P.loc[inter, c] = np.where(mask.loc[inter], src.loc[inter], dst.loc[inter])
            # stats
            for c in cols_panda_to_update:
                try:
                    updated_counts[c] = int((P.loc[inter, c] == T.loc[inter, c]).sum())
                except Exception:
                    updated_counts[c] = int(len(inter))

        # 8) Ajouter les lignes Vérités manquantes dans Panda (si demandé)
        added_rows = 0
        if add_missing_truths and len(missing_in_panda) > 0:
            new_rows = pd.DataFrame(index=missing_in_panda, columns=panda_schema, dtype=object)

            # position Panda = position Vérités correspondante
            if pos_p in panda_schema:
                pos_src = pos_p if pos_p in T.columns else pos_t
                if pos_src not in T.columns and pos_t in dfT.columns:
                    tmp_pos = dfT.copy()
                    tmp_pos["_key_"] = _maybe_round_pos(tmp_pos[pos_t]) if round_positions else _coerce_numeric(tmp_pos[pos_t])
                    tmp_pos = tmp_pos.set_index("_key_")
                    new_rows[pos_p] = tmp_pos.loc[missing_in_panda, pos_t].values
                else:
                    new_rows[pos_p] = T.loc[missing_in_panda, pos_src].values

            # remplir colonnes communes depuis T
            for c in cols_panda_to_update:
                if c in T.columns:
                    new_rows[c] = T.loc[missing_in_panda, c].values

            P = pd.concat([P, new_rows], axis=0)
            added_rows = len(new_rows)

        # 9) Confrontation STRICTE (positions identiques uniquement)
        P_reset = P.reset_index()
        T_reset = T.reset_index()
        conf_df = pd.merge(
            P_reset, T_reset,
            on="_key_", how="inner",
            suffixes=("_p", "_v"),
            copy=False
        ).sort_values("_key_")

        # 10) Finalisation et affichage
        P.reset_index(drop=True, inplace=True)
        st.session_state["panda_upsert_df"] = P.copy()
        st.session_state["confrontation_df"] = conf_df.copy()

        st.success(
            f"UPsert terminé.\n"
            f"- Lignes Panda finales: {len(P)} (+{added_rows} ajoutées depuis Vérités)\n"
            f"- Positions en coïncidence (strict): {len(inter)}\n"
            f"- Vérités hors Panda (avant ajout): {len(missing_in_panda)}\n"
            f"- Panda sans Vérités: {len(missing_in_truth)}"
        )

        with st.expander("Détails colonnes modifiées (sur coïncidences seulement)"):
            if updated_counts:
                st.write(updated_counts)
            else:
                st.write("Aucune colonne modifiée.")

        st.subheader("Confrontation (positions identiques uniquement)")
        st.caption("Jointure interne stricte sur la position (clé).")
        st.dataframe(conf_df.head(500), use_container_width=True)
