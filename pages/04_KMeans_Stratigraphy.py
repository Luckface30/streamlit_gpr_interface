# 04_DBSCAN_Stratigraphy.py
import streamlit as st
from pages.modules.kmeans_trend_ui import dbscan_block

st.set_page_config(page_title="Clustering DBSCAN sur stratigraphie", layout="wide")
st.title("🔍 DBSCAN sur stratigraphie fusionnée")

dbscan_block()
