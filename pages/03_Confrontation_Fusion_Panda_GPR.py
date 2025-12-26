import streamlit as st
from pages.modules.panda_upsert_ui import upsert_block
from pages.modules.panda_fusion_ui import fusion_block
from pages.modules.panda_signals_ui import signals_block

# === Configuration de la page
st.set_page_config(page_title="Confrontation & Fusion — Panda / Vérités / GPR", layout="wide")
st.title("🧪 Confrontation + Ajout — Fusion — Signaux")

# === Onglets
tab1, tab2, tab3 = st.tabs([
    "1) UPsert Panda ⇄ Vérités (mapping)",
    "2) Fusion Panda + GPR",
    "3) Ajout des signaux",
])

with tab1:
    upsert_block()

with tab2:
    fusion_block()

with tab3:
    signals_block()
