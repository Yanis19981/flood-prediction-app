import streamlit as st
import pandas as pd
import folium
import numpy as np
import branca

st.set_page_config(layout="wide")

# --- Données
DATA_PATH = "data/Niveaux_eau.csv"   # <- adapte le nom
lat_col, lon_col = "lat_wgs84", "long_wgs84"
val_col = "alt_m_cgvd"               # <- la valeur à colorer (ex: altitude/profondeur)

df = pd.read_csv(DATA_PATH)

# --- Carte centrée grossièrement Québec/ON
m = folium.Map(location=[46.8, -71.2], zoom_start=6, control_scale=True)

# 1) Imagerie ArcGIS (fond)
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Imagerie",
    overlay=False
).add_to(m)

# 2) Labels (overlay) = “Hybrid”
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Esri Reference Layer",
    name="Labels",
    overlay=True,
    control=True
).add_to(m)

# (optionnel) un fond clair alternatif
folium.TileLayer("CartoDB positron", name="Fond clair").add_to(m)
