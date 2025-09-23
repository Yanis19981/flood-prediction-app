import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

st.title("🌊 Application de prédiction d’inondation")

# Charger les données depuis le dossier data
df = pd.read_csv("data/Niveaux_eau.csv", encoding="utf-8")
stations = pd.read_csv("data/stations_hydrometriques.csv", encoding="utf-8")
metadonnees = pd.read_csv("data/metadonnees_stations_historiques.csv", encoding="utf-8")
modeles = pd.read_csv("data/Modeles_numeriques.csv", encoding="utf-8")

st.subheader("Aperçu des données")
st.dataframe(df.head())

# Exemple : afficher une carte centrée sur Québec
if "Latitude" in df.columns and "Longitude" in df.columns:
    m = folium.Map(location=[46.8, -71.2], zoom_start=6)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            popup=str(row.to_dict()),
            color="blue",
            fill=True,
            fill_color="blue"
        ).add_to(m)

    st_folium(m, width=700, height=500)
