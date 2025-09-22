import streamlit as st
import folium
from streamlit_folium import st_folium

st.title("🌊 Application de prédiction d’inondation")

st.write("Ceci est une première version de test. Une carte interactive s'affiche ci-dessous :")

# Créer une carte centrée sur Montréal
m = folium.Map(location=[45.5, -73.6], zoom_start=8)

# Ajouter un marqueur test
folium.Marker(
    location=[45.5, -73.6],
    popup="Point test",
    icon=folium.Icon(color="blue", icon="info-sign")
).add_to(m)

# Afficher la carte dans Streamlit
st_folium(m, width=700, height=500)
