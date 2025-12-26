import os
import numpy as np
import pandas as pd
import streamlit as st


def init_page():
    """Initialise la page principale Mask R-CNN"""
    st.set_page_config(page_title="Détection interfaces (Mask R-CNN)", layout="wide")
    st.title("🧠 Détection des interfaces — Mask R-CNN")
    _init_state()


def _init_state():
    """Initialise les variables de session avec leurs valeurs par défaut"""

    # état du scan
    st.session_state.setdefault("_scan", None)

    # RAW en px, jamais modifiés
    st.session_state.setdefault("df_asc_raw", None)
    st.session_state.setdefault("df_desc_raw", None)

    # unités/params par catégorie (affichage/export)
    st.session_state.setdefault("unit_asc", "px")
    st.session_state.setdefault("unit_desc", "px")
    st.session_state.setdefault("t_ns_asc", 24.0)   # Longueur image (ns) - ASC
    st.session_state.setdefault("t_ns_desc", 24.0)  # Longueur image (ns) - DESC
    st.session_state.setdefault("hreal_asc", 249)   # Hauteur réelle (px) - ASC
    st.session_state.setdefault("hreal_desc", 249)  # Hauteur réelle (px) - DESC

    # bornes par catégorie
    st.session_state.setdefault("xmin_asc", None)
    st.session_state.setdefault("xmax_asc", None)
    st.session_state.setdefault("xmin_desc", None)
    st.session_state.setdefault("xmax_desc", None)

    # style de tracé
    st.session_state.setdefault("style_asc", "Points")
    st.session_state.setdefault("style_desc", "Points")
    st.session_state.setdefault("style_fuse", "Points")

    # paramètres fusion (corrigés : 24 ns et 249 px par défaut)
    st.session_state.setdefault("unit_fuse", "px")
    st.session_state.setdefault("t_ns_fuse", 24.0)   # Longueur image (ns) - Fusion
    st.session_state.setdefault("hreal_fuse", 249)   # Hauteur réelle (px) - Fusion
    st.session_state.setdefault("xmin_fuse", None)
    st.session_state.setdefault("xmax_fuse", None)
