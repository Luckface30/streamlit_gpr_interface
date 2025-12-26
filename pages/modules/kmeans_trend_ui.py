import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from .maskrcnn_helpers import (
    _xcm_minmax, _filter_troncon, _plot_stratigraphy
)

# =====================================================
# ===  UTILITAIRES DE BASE ============================
# =====================================================

@st.cache_data(show_spinner=False)
def _load_strat_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name="stratigraphie")
    if "x_m" not in df.columns and "x_cm" in df.columns:
        df["x_m"] = df["x_cm"] / 100.0
    return df


def _safe_x_bounds(df_like: pd.DataFrame):
    if "x_m" in df_like.columns and df_like["x_m"].notna().any():
        return float(np.nanmin(df_like["x_m"].values)), float(np.nanmax(df_like["x_m"].values))
    try:
        return _xcm_minmax(df_like)
    except Exception:
        return 0.0, 1.0


# =====================================================
# ===  FUSION / CORRECTION COHÉRENTE =================
# =====================================================

def _correct_interfaces(df, eps_y=0.3, smooth_window=5):
    """
    Corrige les inversions locales et assure la continuité horizontale.
    - eps_y : tolérance verticale (ns) pour fusionner les interfaces proches
    - smooth_window : taille du filtre médian pour lisser les profils
    """
    df = df.copy()
    iface_cols = [c for c in df.columns if str(c).startswith("interface_")]

    # 1) Tri vertical correct (du haut vers le bas)
    df[iface_cols] = np.sort(df[iface_cols].values, axis=1)

    # 2) Lissage horizontal de chaque interface (médian sur X)
    for c in iface_cols:
        vals = df[c].interpolate(limit_direction="both").values
        df[c] = median_filter(vals, size=smooth_window)

    # 3) Correction locale : éviter les croisements et fusions si trop proches
    for i in range(1, len(iface_cols)):
        upper = iface_cols[i - 1]
        lower = iface_cols[i]

        # Si inversion locale (interface du dessous passe au-dessus)
        mask_invert = df[lower] < df[upper]
        if mask_invert.any():
            avg_line = (df[lower] + df[upper]) / 2
            df.loc[mask_invert, lower] = avg_line[mask_invert] + eps_y / 2
            df.loc[mask_invert, upper] = avg_line[mask_invert] - eps_y / 2

        # Si les deux sont trop proches (< eps_y)
        too_close = (df[lower] - df[upper]) < eps_y
        if too_close.any():
            df.loc[too_close, lower] = df[upper][too_close] + eps_y

    return df


# =====================================================
# ===  INTERFACE STREAMLIT ============================
# =====================================================

def dbscan_block():
    st.markdown("### 6) Correction automatique de la cohérence horizontale (sans DBSCAN/KMeans)")

    upl = st.file_uploader("Charger stratigraphie FUSION (.xlsx)", type=["xlsx", "xls"], key="upl_strat_autocorr")
    if upl is None:
        st.info("Charge un fichier avec une feuille 'stratigraphie' contenant des colonnes `interface_*`.")
        return

    try:
        df = _load_strat_excel(upl)
    except Exception as e:
        st.error(f"Erreur lecture Excel : {e}")
        return

    iface_cols = [c for c in df.columns if str(c).startswith("interface_")]
    if not iface_cols:
        st.warning("Aucune colonne `interface_*` détectée.")
        return

    st.success(f"✅ Stratigraphie chargée ({len(df)} lignes, {len(iface_cols)} interfaces)")

    # === Sélection de tronçon
    xmin_all, xmax_all = _safe_x_bounds(df)
    c1, c2 = st.columns(2)
    with c1:
        xmin_sel = st.number_input("Borne min (m)", min_value=float(xmin_all), max_value=float(xmax_all),
                                   value=float(xmin_all), step=1.0)
    with c2:
        xmax_sel = st.number_input("Borne max (m)", min_value=float(xmin_all), max_value=float(xmax_all),
                                   value=float(xmax_all), step=1.0)

    df_view = _filter_troncon(df, xmin_sel, xmax_sel)

    # === Affichage original
    st.markdown("#### Stratigraphie originale")
    _plot_stratigraphy(
        df_view, "Stratigraphie originale", "ns", 24.0,
        "Points", "x_m", "Position (m)", iface_cols, 5
    )

    # === Paramètres de correction
    st.markdown("---")
    st.subheader("Paramètres de correction de cohérence")
    c1, c2 = st.columns(2)
    with c1:
        eps_y = st.number_input("Tolérance verticale (ns)", 0.05, 2.0, 0.3, 0.05)
    with c2:
        smooth_window = st.slider("Fenêtre de lissage (points)", 1, 21, 5, 2)

    # === Lancer correction
    if st.button("▶️ Appliquer la correction horizontale", type="primary"):
        with st.spinner("Correction et réalignement des interfaces..."):
            df_corr = _correct_interfaces(df, eps_y=eps_y, smooth_window=smooth_window)
            df_corr_view = _filter_troncon(df_corr, xmin_sel, xmax_sel)

        st.success("✅ Correction terminée (nombre d’interfaces inchangé)")

        # === Affichage après correction
        st.markdown("#### Stratigraphie corrigée")
        selected_clusters = st.multiselect("Interfaces à afficher (résultat)",
                                           iface_cols, default=iface_cols)
        if not selected_clusters:
            selected_clusters = iface_cols

        _plot_stratigraphy(
            df_corr_view, "Stratigraphie corrigée (interfaces réalignées)", "ns", 24.0,
            "Points", "x_m", "Position (m)", selected_clusters, 5
        )

        # === Export Excel
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="original", index=False)
            df_corr.to_excel(w, sheet_name="stratigraphie_corrigee", index=False)
        bio.seek(0)
        st.download_button(
            "💾 Télécharger stratigraphie_corrigee.xlsx",
            data=bio,
            file_name="stratigraphie_corrigee.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
