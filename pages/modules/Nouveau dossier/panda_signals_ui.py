import io
import re
import numpy as np
import pandas as pd
import streamlit as st

from .panda_upsert_ui import _read_excel_with_sheet


def signals_block():
    st.header("Ajout des signaux à partir des fichiers ANT_..._start_end.xlsx")

    def _parse_interval_from_name(fname: str):
        """Extrait (start,end) depuis le nom du fichier : ..._<start>_<end>.xlsx"""
        base = str(fname).split("/")[-1]
        m = re.search(r"_([0-9]{4,})_([0-9]{4,})\.xlsx$", base)
        if not m:
            return None, None
        try:
            a, b = int(m.group(1)), int(m.group(2))
            return (a, b) if a <= b else (b, a)
        except Exception:
            return None, None

    def _find_position_column_in_ant(df: pd.DataFrame, start: int, end: int):
        if df.empty:
            return None
        first_col = df.columns[0]
        s = pd.to_numeric(df[first_col], errors="coerce")
        if s.notna().any():
            return first_col
        # Cherche colonnes candidates
        candidates = list(df.columns[:5])
        for c in candidates:
            cn = str(c).strip().lower()
            if cn in ["position", "position_cm", "x_cm", "pos", "km", "pk", "positioncm"]:
                return c
        # Heuristique par range
        best, best_ratio = None, -1.0
        for c in candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            valid = s.dropna()
            if valid.empty: continue
            in_range = ((valid >= start) & (valid <= end)).mean()
            if in_range > best_ratio:
                best_ratio, best = in_range, c
        if best: return best
        for c in candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any(): return c
        return None

    def _collect_traces_for_positions(fusion_df: pd.DataFrame,
                                      pos_col_fusion: str,
                                      ant_files,
                                      read_all_sheets: bool) -> pd.DataFrame:
        """
        Produit un tableau enrichi :
        [colonnes fusion (meta)] + [source_file, sheet_name, trace_index] + [sig_1..sig_M]
        """
        fx = fusion_df.copy()
        fx[pos_col_fusion] = pd.to_numeric(fx[pos_col_fusion], errors="coerce")
        fx = fx.dropna(subset=[pos_col_fusion])

        meta_cols = list(fx.columns)
        fusion_meta_by_pos = fx.drop_duplicates(pos_col_fusion, keep="first").set_index(pos_col_fusion)
        target_positions = set(fusion_meta_by_pos.index.astype(int).tolist())
        if not target_positions:
            return pd.DataFrame()

        rows_out, max_sig_len = [], 0
        gmin, gmax = min(target_positions), max(target_positions)

        for upl in (ant_files or []):
            if upl is None: continue
            src_name = getattr(upl, "name", None) or str(upl)
            start, end = _parse_interval_from_name(src_name)
            if start is None or end is None:
                st.warning(f"Nom ignoré (intervalle introuvable) : {src_name}")
                continue
            if gmin > end or gmax < start:
                continue

            try:
                xls = pd.ExcelFile(upl)
                sheets = xls.sheet_names or [None]
            except Exception as e:
                st.error(f"Impossible d'ouvrir {src_name} : {e}")
                continue

            sheets_to_read = sheets if read_all_sheets else sheets[:1]
            for sheet in sheets_to_read:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet)
                except Exception as e:
                    st.error(f"Lecture échouée {src_name} / {sheet}: {e}")
                    continue
                if df.empty: continue

                pos_col_ant = _find_position_column_in_ant(df, start, end)
                if not pos_col_ant:
                    st.warning(f"Colonne position introuvable (ANT) : {src_name} / {sheet}")
                    continue

                df[pos_col_ant] = pd.to_numeric(df[pos_col_ant], errors="coerce")
                df = df.dropna(subset=[pos_col_ant])
                df = df[(df[pos_col_ant] >= start) & (df[pos_col_ant] <= end)]
                df["__P__"] = df[pos_col_ant].astype(int)
                df = df[df["__P__"].isin(target_positions)]
                if df.empty: continue

                if df.shape[1] <= 5: continue  # pas de signaux

                sig_df = df.iloc[:, 5:].copy()
                sig_df = sig_df.apply(pd.to_numeric, errors="coerce")

                for ridx, (p_int, sig_vals) in enumerate(zip(df["__P__"], sig_df.values)):
                    if p_int not in fusion_meta_by_pos.index: continue
                    meta = fusion_meta_by_pos.loc[p_int]
                    meta_list = [meta.get(c, np.nan) if hasattr(meta, "get") else meta[c] for c in meta_cols]
                    sig_list = list(sig_vals.tolist())
                    max_sig_len = max(max_sig_len, len(sig_list))
                    rows_out.append(meta_list + [src_name, sheet or "", ridx] + sig_list)

        if not rows_out: return pd.DataFrame()

        out_cols = meta_cols + ["source_file", "sheet_name", "trace_index"] + [f"sig_{i+1}" for i in range(max_sig_len)]
        rows_out = [r + [np.nan] * (len(out_cols) - len(r)) for r in rows_out]
        out_df = pd.DataFrame(rows_out, columns=out_cols)

        sort_cols = [pos_col_fusion] if pos_col_fusion in out_df.columns else []
        sort_cols += ["source_file", "sheet_name", "trace_index"]
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)
        return out_df

    # === A) Charger Fusion
    st.subheader("A. Charger le fichier **Fusion Panda+GPR**")
    upl_fus = st.file_uploader("Excel Fusion (.xlsx/.xls)", type=["xlsx", "xls"], key="upl_fusion_for_signals")
    df_fus, cols_fus = pd.DataFrame(), []
    if upl_fus is not None:
        df_fus = _read_excel_with_sheet(upl_fus, "Fusion")
        if not df_fus.empty:
            st.success(f"Fusion: {len(df_fus)} lignes • {len(df_fus.columns)} colonnes")
            with st.expander("Aperçu Fusion"):
                st.dataframe(df_fus.head(200), use_container_width=True)
            cols_fus = list(df_fus.columns)

    pos_fus = None
    if cols_fus:
        pos_fus = st.selectbox("Colonne **position** (Fusion)", cols_fus, index=0, key="pos_fusion_signals")

    # === B) Charger ANT
    st.subheader("B. Charger les fichiers **ANT** (signaux)")
    ant_files = st.file_uploader("Fichiers ANT_..._start_end.xlsx (plusieurs)", type=["xlsx", "xls"], accept_multiple_files=True, key="upl_ant_files")
    if ant_files:
        st.info(f"{len(ant_files)} fichier(s) ANT importé(s).")

    read_all_sheets = st.checkbox("Lire toutes les feuilles (sinon seulement la première)", value=True)

    # === C) Exécuter
    st.subheader("C. Extraire & assembler les signaux")
    if st.button("▶️ Lancer l’enrichissement", disabled=(df_fus.empty or not pos_fus or not ant_files)):
        try:
            enriched = _collect_traces_for_positions(df_fus, pos_fus, ant_files, read_all_sheets)
        except Exception as e:
            st.error(f"Erreur durant l’assemblage : {e}")
            enriched = pd.DataFrame()

        if enriched.empty:
            st.warning("Aucun signal trouvé pour les positions du fichier Fusion.")
        else:
            st.success(f"OK : {len(enriched)} lignes de signaux assemblées.")
            with st.expander("Aperçu du résultat"):
                st.dataframe(enriched.head(200), use_container_width=True)

            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as w:
                enriched.to_excel(w, sheet_name="fusion_signaux", index=False)
            bio.seek(0)
            st.download_button("💾 Télécharger l'Excel enrichi (metadata + signaux)",
                               data=bio,
                               file_name="fusion_signaux_enrichi.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
