import io
import re
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

from core.utils import find_col, fuse_top_two_rows_as_header
from core.interp import clean_and_interpolate_local
from core.indicateur import sample_eps1_with_indicator, sample_eps2_with_indicator

# ---------------------------
# Config & chemins
# ---------------------------
st.set_page_config(page_title="Outil: Import • Nettoyage • Interpolation • ε₁/ε₂ + Indicateurs • Export", layout="wide")
st.title("📦 Mini outil (local) — Import • Nettoyage • Interpolation • ε₁/ε₂ + Indicateurs • Export")

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
INTERP_XLSX_PATH = CACHE_DIR / "last_interpolated.xlsx"  # Sauvegarde auto à l'étape Interpolation

# ---------------------------
# Helpers spécifiques
# ---------------------------
def _normalize(s: str) -> str:
    """
    Normalise pour comparaisons tolérantes : supprime accents, compacte espaces,
    met en minuscules, remplace N°/Nº -> N (on retire les symboles de degré/ordinal).
    """
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFD", s)
    # retire accents
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # retire symboles degré/ordinal pour capter "N°"/"Nº"
    s = s.replace("°", "").replace("º", "")
    # compacte espaces et minuscule
    return re.sub(r"\s+", " ", s).strip().lower()

def is_endoscopie_fin_col(col_name: str) -> bool:
    """
    Détecte des variantes du libellé 'Cote de fin de sondage endoscopique (m)'.
    Règle robuste : présence conjointe de {cote}, {fin}, {sondage}, {endoscop}.
    """
    n = _normalize(col_name)
    return all(k in n for k in ["cote", "fin", "sondage", "endoscop"])

# Libellés à masquer par défaut (comparaison sur libellé normalisé)
# ⚠️ NE contient plus "fond de ballast colmaté (m)" -> on souhaite l'interpoler.
DEFAULT_DROP_LABELS_CANON = {
    # —— base initiale (moins la ligne "fond de ballast colmaté (m)") ——
    "resistance de pointe (mpa)",
    "sous-couche",
    "couche intermediaire",
    "sol support",
    "coordonnees gps est",
    "altitude",
    "conditions climatiques",
    "fichier de sondage",
    "ecart type (mpa)",
    "cote de fin sondage endoscopique (m)",

    # —— variantes et champs demandés précédemment ——
    "n°", "n",  # N° géré explicitement et via normalisation
    "date",
    "distance de l'axe de la voie (m)",
    "nature",
    "consistance",
    "classe de qualite",
    "sol support cote (m)",
    "resistance de pointe (mpa)__1",
    "ecart type (mpa)__1",
    "resistance de pointe (mpa)__2",
    "ecart type (mpa)__2",
    "resistance de pointe (mpa)__3",
    "ecart type (mpa)__3",
    "resistance de pointe (mpa)__4",
    "ecart type (mpa)__4",
    "nature__1",
    "nature__2",
    "consistance__1",
    "consistance__2",

    # —— nouveaux à supprimer ——
    "sous-couche cote (m)",
    "couche intermediaire cote (m)",
    "nord",
    "cote de fin sondage panda (m)",
}

# Cibles par nom à pré-sélectionner dans "Colonnes numériques à interpoler"
TARGET_NUMERIC_DEFAULTS = {
    "km",
    "ballast cote touche (m)",
    "fond de ballast sain (m)",
    "fond de ballast colmate (m)",   # orthographe normalisée sans accent
}

def _should_auto_drop(col_name: str) -> bool:
    """
    True si la colonne doit être pré-sélectionnée pour suppression.
    - correspondance exacte avec DEFAULT_DROP_LABELS_CANON (après normalisation)
    - ou colonne détectée comme 'fin de sondage endoscopique'
    """
    n = _normalize(col_name)
    if n in DEFAULT_DROP_LABELS_CANON:
        return True
    if is_endoscopie_fin_col(col_name):
        return True
    return False

# ---------------------------
# État
# ---------------------------
if "raw_df" not in st.session_state: st.session_state.raw_df = None
if "processed_df" not in st.session_state: st.session_state.processed_df = None

# ---------------------------
# Sidebar paramètres
# ---------------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    step_cm = st.number_input("Pas d'interpolation (cm)", min_value=1, max_value=1000, value=10, step=1)
    half_window_cm = st.number_input("Fenêtre ± (cm)", min_value=10, max_value=10_000, value=200, step=10)
    st.caption("Flux: Import → Nettoyage → Interpoler (sauvegarde auto) → ε₁/ε₂ + Indicateurs → Export")

tab1, tab2, tab3, tab4 = st.tabs([
    "1) Import & Nettoyage",
    "2) Interpolation",
    "3) ε₁/ε₂ + Indicateurs",
    "4) Export"
])

# ---- Tab 1: Import & Nettoyage ----
with tab1:
    st.subheader("Importer un fichier")
    up = st.file_uploader("Excel ou CSV", type=["xlsx", "xls", "csv"])
    colA, colB = st.columns(2)
    with colA:
        merge_headers = st.checkbox("Fusionner les 2 premières lignes en en-têtes", value=True)
    with colB:
        skip_blank_top = st.checkbox("Ignorer lignes vides au-dessus", value=True)

    if up is not None:
        try:
            # Lecture
            if up.name.lower().endswith(".csv"):
                raw = pd.read_csv(up, header=None) if merge_headers else pd.read_csv(up)
            else:
                raw = pd.read_excel(up, header=None) if merge_headers else pd.read_excel(up)

            # Nettoyage des lignes vides du haut (si demandé)
            if merge_headers and skip_blank_top:
                while len(raw) and raw.iloc[0].isna().all():
                    raw = raw.iloc[1:].reset_index(drop=True)

            # Fusion 2 premières lignes -> en-têtes (si demandé)
            df = fuse_top_two_rows_as_header(raw) if merge_headers else raw.copy()

            st.session_state.raw_df = df.copy()
            st.session_state.processed_df = None  # reset après un nouvel import

            st.success(f"Fichier chargé: {up.name} • {df.shape[0]} lignes, {df.shape[1]} colonnes")
            st.dataframe(df.head(50), use_container_width=True)

            # Détection colonnes
            km_guess = find_col(df, ["Km", "KM", "kilometre", "kilometer", "pk", "position_km"])
            pos_col_guess = find_col(df, ["Position", "position"])
            numeric_guess = [c for c in df.columns if c != km_guess and pd.api.types.is_numeric_dtype(df[c])]

            # Auto-exclusion: 'cote de fin de sondage endoscopique (m)'
            auto_excluded = [c for c in df.columns if is_endoscopie_fin_col(c)]
            if auto_excluded:
                st.info("🛑 Colonne(s) fin de sondage endoscopique détectée(s) et masquée(s) automatiquement : "
                        + ", ".join(auto_excluded))

            st.markdown("### Sélection des colonnes")
            c1, c2 = st.columns(2)
            with c1:
                km_col = st.selectbox("Colonne Km", options=list(df.columns),
                                      index=(list(df.columns).index(km_guess) if km_guess in list(df.columns) else 0))
            with c2:
                position_col = st.selectbox("Colonne Position (optionnel)", options=[None] + list(df.columns),
                                            index=(0 if pos_col_guess is None else (1 + list(df.columns).index(pos_col_guess))))

            # Colonnes à supprimer
            st.markdown("### Nettoyage — colonnes à supprimer avant interpolation")
            # 1) Règle générique: colonnes "Unnamed", vides, etc.
            default_drop = [
                c for c in df.columns
                if re.match(r"(?i)^unnamed", str(c)) or str(c).strip() in {"", "None", "NaN"}
            ]
            # 2) Règle spécifique: libellés fournis (tolérants, gérés par _normalize)
            default_drop_specific = [c for c in df.columns if _should_auto_drop(c)]
            # 3) Fusion avec l’auto-exclusion endoscopie
            default_drop = sorted(list(set(default_drop + default_drop_specific + auto_excluded)))

            # Colonnes protégées (Km/Position) à ne jamais proposer par défaut
            protected = {km_col}
            if position_col:
                protected.add(position_col)

            drop_candidates = [c for c in df.columns if c not in protected]
            drop_cols = st.multiselect("Sélectionne les colonnes à supprimer",
                                       options=drop_candidates,
                                       default=[c for c in default_drop if c in drop_candidates])

            # Colonnes numériques à interpoler (⚠️ exclure celles supprimées + endoscopie fin)
            selectable_cols = [c for c in df.columns if c not in drop_cols and c not in auto_excluded]

            # Pré-sélection par nom (tolérant via _normalize) + ajoute les colonnes numériques détectées
            preferred_numeric = [c for c in selectable_cols if _normalize(c) in TARGET_NUMERIC_DEFAULTS]
            # Inclure Km visuellement si présent dans les colonnes sélectionnables
            if km_col in selectable_cols and km_col not in preferred_numeric:
                preferred_numeric = [km_col] + preferred_numeric

            # Concaténer avec la détection auto des numériques (et dédupliquer en conservant l'ordre)
            numeric_default_ordered = list(dict.fromkeys(preferred_numeric + [c for c in numeric_guess if c in selectable_cols]))

            numeric_cols = st.multiselect("Colonnes numériques à interpoler (m → seront converties en cm)",
                                          options=selectable_cols,
                                          default=numeric_default_ordered)

            # Filtre Position optionnel
            if position_col:
                pos_values = ["— Aucun filtre —"] + sorted(df[position_col].dropna().astype(str).unique().tolist())
                position_filter_value = st.selectbox("Filtrer Position (optionnel)", options=pos_values)
            else:
                position_filter_value = None

            # ⚠️ On retire km_col des colonnes numériques à transmettre à la fonction d'interpolation
            numeric_cols_effective = [c for c in numeric_cols if c != km_col]

            st.session_state._params = {
                "km_col": km_col,
                "position_col": position_col,
                "position_filter_value": position_filter_value,
                "numeric_cols": numeric_cols_effective,
                "drop_cols": sorted(list(set(drop_cols + auto_excluded))),
            }

        except Exception as e:
            st.error(f"Erreur de lecture: {e}")

# ---- Tab 2: Interpolation ----
with tab2:
    st.subheader("Interpolation locale (±2 m par défaut)")
    if st.session_state.raw_df is None:
        st.info("Charge d'abord un fichier dans l'onglet Import & Nettoyage.")
    else:
        p = st.session_state.get("_params", {})
        if not p.get("km_col"):
            st.warning("Sélectionne la colonne Km dans l'onglet précédent.")
        else:
            if st.button("🚀 Nettoyer + Interpoler (sauvegarde auto)"):
                df_proc, dropped = clean_and_interpolate_local(
                    df=st.session_state.raw_df.copy(),
                    km_col=p["km_col"],
                    numeric_cols=p["numeric_cols"],
                    position_filter_value=p["position_filter_value"],
                    position_col=p["position_col"],
                    drop_cols=p.get("drop_cols", []),
                    step_cm=step_cm,
                    half_window_cm=half_window_cm,
                )

                if df_proc is None or df_proc.empty:
                    st.error("Aucun résultat généré (vérifie Km, colonnes numériques, etc.).")
                else:
                    for _c in ("anchor_idx", "anchor_Km"):
                        if _c in df_proc.columns:
                            df_proc = df_proc.drop(columns=[_c])

                    endo_cols = [c for c in df_proc.columns if is_endoscopie_fin_col(c)]
                    if endo_cols:
                        df_proc = df_proc.drop(columns=endo_cols)
                        dropped = sorted(list(set((dropped or []) + endo_cols)))

                    km_col = p["km_col"]
                    if km_col in df_proc.columns:
                        df_proc[km_col] = pd.to_numeric(df_proc[km_col], errors="coerce") * 100_000.0

                    for col in df_proc.columns:
                        if col != km_col and pd.api.types.is_numeric_dtype(df_proc[col]):
                            df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce") * 100.0

                    try:
                        with pd.ExcelWriter(INTERP_XLSX_PATH, engine="openpyxl") as writer:
                            df_proc.to_excel(writer, index=False, sheet_name="interpolated")
                        st.session_state.processed_df = df_proc
                        st.session_state.interpolated_path = str(INTERP_XLSX_PATH)
                    except Exception as e:
                        st.error(f"Échec de la sauvegarde automatique du fichier interpolé: {e}")
                        st.stop()

                    msg = f"Nouveau tableau: {len(df_proc)} lignes."
                    if dropped:
                        msg += f" Colonnes supprimées: {', '.join(dropped)}."
                    msg += f" (Positions & valeurs numériques converties en **cm** • sauvegardé: {INTERP_XLSX_PATH})"
                    st.success(msg)
                    st.dataframe(df_proc.head(100), use_container_width=True)

# ---- Tab 3: ε₁/ε₂ + Indicateurs ----
with tab3:
    st.subheader("Assigner ε₁ / ε₂ + Ind_1 / Ind_2")

    if not INTERP_XLSX_PATH.exists():
        st.error("Aucun fichier interpolé trouvé. Va d'abord dans l'onglet **Interpolation**.")
        st.stop()

    try:
        df = pd.read_excel(INTERP_XLSX_PATH, engine="openpyxl")
    except Exception as e:
        st.error(f"Impossible de charger le fichier interpolé: {e}")
        st.stop()

    st.session_state.processed_df = df.copy()

    c_obs = st.selectbox("Colonne Observations", options=[None] + list(df.columns))
    c_hbs = st.selectbox("Colonne Humidité BS", options=[None] + list(df.columns))
    c_hbc = st.selectbox("Colonne Humidité BC", options=[None] + list(df.columns))
    c_colm = st.selectbox("Colonne Colmatage", options=[None] + list(df.columns))

    if st.button("🧪 Calculer ε₁/ε₂ + Indicateurs"):
        missing = [name for name, col in [
            ("Observations", c_obs), ("Humidité BS", c_hbs),
            ("Humidité BC", c_hbc), ("Colmatage", c_colm)
        ] if col is None]

        if missing:
            st.error(f"Colonnes manquantes: {', '.join(missing)}")
            st.stop()

        df[["Eps_1", "Ind_1"]] = df.apply(
            lambda r: pd.Series(sample_eps1_with_indicator(r[c_hbs], r[c_obs])), axis=1
        )
        df[["Eps_2", "Ind_2"]] = df.apply(
            lambda r: pd.Series(sample_eps2_with_indicator(r[c_hbc], r[c_colm], r[c_obs])), axis=1
        )

        try:
            with pd.ExcelWriter(INTERP_XLSX_PATH, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="interpolated_eps")
        except Exception as e:
            st.error(f"Échec de la sauvegarde avec ε₁/ε₂ + Indicateurs: {e}")
            st.stop()

        st.session_state.processed_df = df
        st.success(f"ε₁/ε₂ + Indicateurs ajoutés et sauvegardés dans {INTERP_XLSX_PATH}.")
        st.dataframe(df.head(100), use_container_width=True)

# ---- Tab 4: Export ----
with tab4:
    st.subheader("Export")
    if st.session_state.processed_df is None and st.session_state.raw_df is None:
        st.info("Rien à exporter.")
    else:
        choice = st.radio("Que veux-tu exporter ?", ["Tableau traité (si présent)", "Tableau brut (tel que chargé)"])
        if st.button("📥 Générer le fichier Excel"):
            if choice == "Tableau traité (si présent)" and INTERP_XLSX_PATH.exists():
                try:
                    df_to_save = pd.read_excel(INTERP_XLSX_PATH, engine="openpyxl")
                except Exception:
                    df_to_save = st.session_state.processed_df
            else:
                df_to_save = st.session_state.processed_df if (
                    choice == "Tableau traité (si présent)" and st.session_state.processed_df is not None
                ) else st.session_state.raw_df

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df_to_save.to_excel(writer, index=False, sheet_name="data")
            buf.seek(0)
            st.download_button(
                "Télécharger le fichier Excel",
                data=buf,
                file_name="resultat.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
