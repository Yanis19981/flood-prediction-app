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
# --- Colormap (du bleu -> jaune -> rouge, par ex.)
cmap = branca.colormap.LinearColormap(
    colors=["#2c7fb8", "#ffff8c", "#d7191c"],  # bas -> moyen -> haut
    vmin=float(df[val_col].min()),
    vmax=float(df[val_col].max())
)
cmap.caption = f"{val_col} (min–max)"

# --- Ajout des points
for _, r in df.dropna(subset=[lat_col, lon_col]).iterrows():
    v = float(r[val_col]) if val_col in r and pd.notna(r[val_col]) else np.nan
    color = cmap(v) if not np.isnan(v) else "#999999"
    # rayon entre 4 et 12 en fonction de la valeur
    radius = 4 if np.isnan(v) else float(np.interp(v, [df[val_col].min(), df[val_col].max()], [4, 12]))
    popup = folium.Popup(
        folium.IFrame(
            html=r.to_frame().to_html(header=False), width=300, height=180
        ), max_width=320
    )
    folium.CircleMarker(
        location=[r[lat_col], r[lon_col]],
        radius=radius,
        color=color, fill=True, fill_opacity=0.85, weight=1,
        popup=popup
    ).add_to(m)

# --- Légende + contrôle de couches
cmap.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

# --- Affichage Streamlit
from streamlit_folium import st_folium
st.subheader("Carte (Imagery Hybrid + indicateurs de couleur)")
st_folium(m, height=700, use_container_width=True)
