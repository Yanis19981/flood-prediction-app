import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.title("📊 Application de prédiction d’inondation")

# Charger les données depuis le dossier data
df = pd.read_csv("data/Niveaux d’eau lors d’une inondation (depuis 2000).csv")

st.subheader("Aperçu des données")
st.dataframe(df.head())

# Exemple : vérifier qu'il y a Latitude et Longitude
if "Latitude" in df.columns and "Longitude" in df.columns:
    m = folium.Map(location=[46.8, -71.2], zoom_start=6)  # centré sur Québec
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            popup=str(row.to_dict()),
            color="blue",
            fill=True,
        ).add_to(m)

    st.subheader("Carte interactive")
    st_folium(m, width=700, height=500)
else:
    st.warning("⚠️ Votre CSV n’a pas de colonnes 'Latitude' et 'Longitude'")
