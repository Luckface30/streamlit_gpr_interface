# pages/03_Confrontation_Fusion_Panda_GPR.py
import io
import re
import unicodedata
import difflib
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st

from core.eps import get_eps1, get_eps2

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Confrontation & Fusion — Panda / Vérités / GPR",
    layout="wide"
)
st.title("🧪 Confrontation + Ajout (UPsert + mapping) — 🔗 Fusion — ➕ Ajout des signaux")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _read_excel_with_sheet(upl, label_prefix: str) -> pd.DataFrame:
    try:
        xls = pd.ExcelFile(upl)
        sheet = st.selectbox(
            f"{label_prefix} • Feuille",
            xls.sheet_names,
            index=0,
            key=f"{label_prefix}_sheet"
        )
        return pd.read_excel(xls, sheet_name=sheet)
    except Exception as e:
        st.error(f"Erreur de lecture Excel ({label_prefix}) : {e}")
        return pd.DataFrame()

def _norm(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace("_", "")

def _default_pick(cols, wanted):
    cols = list(cols or [])
    n = {c: _norm(c) for c in cols}
    for w in wanted:
        ww = _norm(w)
        for c, cn in n.items():
            if cn == ww:
                return c
        for c, cn in n.items():
            if ww in cn:
                return c
    return cols[0] if cols else None

def _coerce_numeric(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")

def _safe_selectbox(label: str, options: list, defaults: list, key: str):
    opts = [o for o in (options or []) if o not in (None, "")]
    if not opts:
        st.selectbox(label, ["<aucune>"], index=0, key=key)
        return None
    guess = _default_pick(opts, defaults)
    idx = opts.index(guess) if guess in opts else 0
    return st.selectbox(label, opts, index=idx, key=key)

# ------------------------------------------------------------
# Interpolation Vérités terrain (positions déjà en cm)
# ------------------------------------------------------------
def interpolate_truths_cm(
    df: pd.DataFrame,
    pos_col: str,
    numeric_cols: list[str],
    step_cm: int = 10,
    half_window_cm: int = 200,
) -> pd.DataFrame:
    if df.empty or pos_col not in df.columns:
        return pd.DataFrame()

    x = pd.to_numeric(df[pos_col], errors="coerce")
    xmin, xmax = float(x.min()), float(x.max())

    frames = []
    for i, xi in enumerate(x):
        if pd.isna(xi):
            continue
        start, end = max(xmin, xi - half_window_cm), min(xmax, xi + half_window_cm)
        x_win = np.arange(start, end + step_cm, step_cm)

        row_anchor = df.iloc[i]
        win = {pos_col: x_win}

        # interpolation colonnes numériques
        for col in numeric_cols:
            xv = pd.to_numeric(df[pos_col], errors="coerce")
            yv = pd.to_numeric(df[col], errors="coerce")
            mask = xv.notna() & yv.notna()
            if mask.sum() >= 2:
                win[col] = np.interp(x_win, xv[mask], yv[mask])
            else:
                win[col] = [np.nan] * len(x_win)

        # colonnes texte : duplication valeur ancre
        text_cols = [c for c in df.columns if c not in numeric_cols + [pos_col]]
        for col in text_cols:
            win[col] = [row_anchor.get(col, np.nan)] * len(x_win)

        # EPS recalculés
        obs    = row_anchor.get("obs") or row_anchor.get("observation")
        hum_bs = row_anchor.get("humidite_bs")
        hum_bc = row_anchor.get("humidite_bc")
        colmat = row_anchor.get("colmatage")
        win["Eps_1"] = [get_eps1(hum_bs, obs)] * len(x_win)
        win["Eps_2"] = [get_eps2(hum_bc, colmat, obs)] * len(x_win)

        frames.append(pd.DataFrame(win))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ------------------------------------------------------------
# Onglets
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "1) UPsert Panda ⇄ Vérités (mapping)",
    "2) Fusion Panda + GPR",
    "3) Ajout des signaux",
])

# ============================================================
# TAB 1 — UPsert Panda ⇄ Vérités
# ============================================================
with tab1:
    st.header("UPsert + Interpolation Vérités terrain")

    # Charger Panda
    upl_panda = st.file_uploader("Excel Panda (.xlsx/.xls)", type=["xlsx", "xls"], key="upl_panda_upsert")
    dfP, colsP = pd.DataFrame(), []
    if upl_panda is not None:
        dfP = _read_excel_with_sheet(upl_panda, "Panda")
        if not dfP.empty:
            st.success(f"Panda: {len(dfP)} lignes • {len(dfP.columns)} colonnes")
            st.dataframe(dfP.head(200), use_container_width=True)
            colsP = list(dfP.columns)

    pos_p = _safe_selectbox("Colonne position Panda (cm)", colsP, ["position_cm", "x_cm"], "pos_col_p")

    # Charger Vérités
    upl_T = st.file_uploader("Excel Vérités (.xlsx/.xls)", type=["xlsx", "xls"], key="upl_truth")
    dfT, colsT = pd.DataFrame(), []
    if upl_T is not None:
        dfT = _read_excel_with_sheet(upl_T, "Vérités")
        if not dfT.empty:
            st.success(f"Vérités: {len(dfT)} lignes • {len(dfT.columns)} colonnes")
            st.dataframe(dfT.head(200), use_container_width=True)
            colsT = list(dfT.columns)

    pos_t = _safe_selectbox("Colonne position Vérités (cm)", colsT, ["position_cm", "x_cm"], "pos_col_t")

    # Interpolation Vérités terrain
    interpT = pd.DataFrame()
    if not dfT.empty and pos_t:
        numeric_cols_T = st.multiselect(
            "Colonnes numériques à interpoler (Vérités terrain)",
            options=[c for c in dfT.columns if c != pos_t],
            default=[]
        )
        if numeric_cols_T and st.button("▶️ Interpoler Vérités terrain"):
            interpT = interpolate_truths_cm(dfT, pos_col=pos_t, numeric_cols=numeric_cols_T)
            if not interpT.empty:
                st.success(f"{len(interpT)} lignes interpolées ✅")
                st.dataframe(interpT.head(200), use_container_width=True)
                st.session_state["verites_interp_df"] = interpT

                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine="openpyxl") as w:
                    interpT.to_excel(w, sheet_name="verites_interp", index=False)
                bio.seek(0)
                st.download_button("💾 Télécharger Vérités interpolées",
                                   data=bio, file_name="verites_interp.xlsx")

# ============================================================
# TAB 2 — Fusion Panda + GPR
# ============================================================
with tab2:
    st.header("Fusion Panda + GPR")

    source = st.radio("Source Panda",
                      ["📂 Charger un fichier Panda", "⚡ Utiliser UPsert (onglet 1)"],
                      index=(1 if "panda_upsert_df" in st.session_state else 0),
                      horizontal=True)

    df_panda_fus = pd.DataFrame()
    if source == "⚡ Utiliser UPsert (onglet 1)" and "panda_upsert_df" in st.session_state:
        df_panda_fus = st.session_state["panda_upsert_df"].copy()
        st.success(f"Panda (UPsert) : {len(df_panda_fus)} lignes")
        st.dataframe(df_panda_fus.head(200), use_container_width=True)

    # Utilisation des Vérités interpolées si dispo
    df_verites = st.session_state.get("verites_interp_df", pd.DataFrame())
    if not df_verites.empty:
        st.info("⚡ Fusion avec Vérités interpolées")
        st.dataframe(df_verites.head(100), use_container_width=True)

    # Charger GPR
    upl_gpr = st.file_uploader("Excel GPR (.xlsx/.xls)", type=["xlsx", "xls"], key="upl_gpr")
    dfG = pd.DataFrame()
    if upl_gpr is not None:
        dfG = _read_excel_with_sheet(upl_gpr, "GPR")
        if not dfG.empty:
            st.success(f"GPR: {len(dfG)} lignes")
            st.dataframe(dfG.head(200), use_container_width=True)

    if st.button("➡️ Fusionner Panda + GPR"):
        if df_panda_fus.empty or dfG.empty:
            st.error("Charge Panda et GPR d’abord")
        else:
            fused = pd.merge(
                df_panda_fus,
                dfG,
                left_on=pos_p,
                right_on=dfG.columns[0],  # simplif: première colonne GPR est position
                how="inner"
            )
            st.success(f"Fusion OK : {len(fused)} lignes")
            st.dataframe(fused.head(200), use_container_width=True)
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as w:
                fused.to_excel(w, sheet_name="fusion", index=False)
            bio.seek(0)
            st.download_button("💾 Télécharger Excel fusionné",
                               data=bio, file_name="fusion_panda_gpr.xlsx")

# ============================================================
# TAB 3 — Ajout des signaux
# ============================================================
with tab3:
    st.header("Ajout des signaux (inchangé)")
    st.info("➡️ Conserve ton code existant ici pour enrichir la fusion avec les fichiers ANT...")


# ============================================================
# TAB 3 — Ajout des signaux ANT
# ============================================================
with tab3:
    st.header("Ajout des signaux à partir des fichiers ANT_..._start_end.xlsx")

    def _parse_interval_from_name(fname: str):
        base = str(fname).split("/")[-1]
        m = re.search(r"_([0-9]{4,})_([0-9]{4,})\.xlsx$", base)
        if not m:
            return None, None
        try:
            a, b = int(m.group(1)), int(m.group(2))
            return (a, b) if a <= b else (b, a)
        except Exception:
            return None, None

    def _find_position_column_in_ant(df: pd.DataFrame, start: int, end: int) -> Optional[str]:
        if df.empty:
            return None
        first_col = df.columns[0]
        s = pd.to_numeric(df[first_col], errors="coerce")
        if s.notna().any():
            return first_col
        candidates = list(df.columns[:5])
        for c in candidates:
            cn = str(c).strip().lower()
            if cn in ["position", "position_cm", "x_cm", "pos", "km", "pk", "positioncm"]:
                return c
        for c in candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return c
        return None

    def _collect_traces_for_positions(fusion_df: pd.DataFrame, pos_col_fusion: str, ant_files, read_all_sheets: bool):
        fx = fusion_df.copy()
        fx[pos_col_fusion] = pd.to_numeric(fx[pos_col_fusion], errors="coerce")
        fx = fx.dropna(subset=[pos_col_fusion])
        fusion_meta_by_pos = fx.drop_duplicates(pos_col_fusion, keep="first").set_index(pos_col_fusion)
        target_positions = set(fusion_meta_by_pos.index.astype(int).tolist())
        rows_out = []
        max_sig_len = 0
        for upl in (ant_files or []):
            src_name = getattr(upl, "name", None) or str(upl)
            start, end = _parse_interval_from_name(src_name)
            if start is None or end is None:
                continue
            xls = pd.ExcelFile(upl)
            sheets_to_read = xls.sheet_names if read_all_sheets else xls.sheet_names[:1]
            for sheet in sheets_to_read:
                df = pd.read_excel(xls, sheet_name=sheet)
                if df.empty:
                    continue
                pos_col_ant = _find_position_column_in_ant(df, start, end)
                if not pos_col_ant:
                    continue
                df[pos_col_ant] = pd.to_numeric(df[pos_col_ant], errors="coerce")
                df = df.dropna(subset=[pos_col_ant])
                df["__P__"] = df[pos_col_ant].astype(int)
                df = df[df["__P__"].isin(target_positions)]
                if df.empty:
                    continue
                if df.shape[1] <= 5:
                    continue
                sig_df = df.iloc[:, 5:].apply(pd.to_numeric, errors="coerce")
                for ridx, (p_int, sig_vals) in enumerate(zip(df["__P__"], sig_df.values)):
                    meta = fusion_meta_by_pos.loc[p_int]
                    meta_list = [meta[c] if c in meta.index else np.nan for c in fx.columns]
                    sig_list = list(sig_vals.tolist())
                    max_sig_len = max(max_sig_len, len(sig_list))
                    rows_out.append(meta_list + [src_name, sheet or "", ridx] + sig_list)
        if not rows_out:
            return pd.DataFrame()
        out_cols = list(fx.columns) + ["source_file", "sheet_name", "trace_index"] + [f"sig_{i+1}" for i in range(max_sig_len)]
        out_df = pd.DataFrame(rows_out, columns=out_cols)
        return out_df.sort_values(by=[pos_col_fusion, "source_file", "sheet_name", "trace_index"]).reset_index(drop=True)

    upl_fus = st.file_uploader("Excel Fusion", type=["xlsx", "xls"], key="upl_fusion_for_signals")
    df_fus, cols_fus = pd.DataFrame(), []
    if upl_fus is not None:
        df_fus = _read_excel_with_sheet(upl_fus, "Fusion")
        if not df_fus.empty:
            st.success(f"Fusion: {len(df_fus)} lignes")
            st.dataframe(df_fus.head(200), use_container_width=True)
            cols_fus = list(df_fus.columns)

    pos_fus = None
    if cols_fus:
        pos_fus = _safe_selectbox("Colonne **position** (Fusion)", cols_fus, ["position_cm"], "pos_fusion_signals")

    ant_files = st.file_uploader("Fichiers ANT", type=["xlsx", "xls"], accept_multiple_files=True, key="upl_ant_files")
    read_all_sheets = st.checkbox("Lire toutes les feuilles", value=True)

    if st.button("▶️ Lancer enrichissement", disabled=(df_fus.empty or not pos_fus or not ant_files)):
        enriched = _collect_traces_for_positions(df_fus, pos_fus, ant_files, read_all_sheets)
        if enriched.empty:
            st.warning("Aucun signal trouvé")
        else:
            st.success(f"{len(enriched)} lignes de signaux assemblées")
            st.dataframe(enriched.head(200), use_container_width=True)
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as w:
                enriched.to_excel(w, sheet_name="fusion_signaux", index=False)
            bio.seek(0)
            st.download_button("💾 Télécharger Excel enrichi", data=bio, file_name="fusion_signaux_enrichi.xlsx")
