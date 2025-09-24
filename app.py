
import glob
import os
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

# ---------- CONFIG ----------
st.set_page_config(page_title="Carte satellite - Données CSV", layout="wide")

DATA_DIR = "data"  # tous vos CSV dans ce dossier
SATELLITE_TILE = "Esri.WorldImagery"  # fond imagerie satellite Esri

# ---------- UTILS ----------
@st.cache_data(show_spinner=False)
def list_csv_files(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    return files

@st.cache_data(show_spinner=True)
def load_csv(path: str, sep=","):
    # essaie de lire avec virgule, puis point-virgule si besoin
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_csv(path, sep=";", low_memory=False)
    # normalise les noms de colonnes
    df.columns = [c.strip() for c in df.columns]
    return df

def infer_lat_lon_columns(cols):
    """Tente de trouver des colonnes latitude/longitude usuelles."""
    candidates_lat = ["lat", "latitude", "Lat", "Latitude", "lat_wgs84", "lat_wgs"]
    candidates_lon = ["lon", "lng", "longitude", "Lon", "Longitude", "long_wgs84", "long_wgs"]
    lat = next((c for c in candidates_lat if c in cols), None)
    lon = next((c for c in candidates_lon if c in cols), None)
    return lat, lon

def to_float(s):
    """Convertit en float en gérant virgules décimales éventuelles."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

# ---------- SIDEBAR : chargement ----------
st.sidebar.title("Données")
csv_files = list_csv_files(DATA_DIR)
if not csv_files:
    st.sidebar.error(f"Aucun CSV trouvé dans `{DATA_DIR}/`")
    st.stop()

file_display = [os.path.basename(p) for p in csv_files]
file_choice = st.sidebar.selectbox("Choisir un fichier CSV", file_display, index=0)
csv_path = csv_files[file_display.index(file_choice)]

df = load_csv(csv_path)
st.sidebar.success(f"{file_choice} chargé ({len(df):,} lignes, {len(df.columns)} colonnes)")

# ---------- Sélection des colonnes ----------
st.sidebar.subheader("Colonnes géographiques")
auto_lat, auto_lon = infer_lat_lon_columns(df.columns)
lat_col = st.sidebar.selectbox("Colonne Latitude", [auto_lat] + list(df.columns) if auto_lat else list(df.columns), index=0 if auto_lat else 0)
lon_col = st.sidebar.selectbox("Colonne Longitude", [auto_lon] + list(df.columns) if auto_lon else list(df.columns), index=0 if auto_lon else 0)

value_col = st.sidebar.selectbox("Colonne valeur (taille/couleur - optionnel)", ["(aucune)"] + list(df.columns), index=0)
popup_cols = st.sidebar.multiselect("Colonnes à afficher dans le popup", df.columns.tolist(), default=df.columns[:5].tolist())

st.sidebar.subheader("Affichage")
view_mode = st.sidebar.radio("Mode", ["Clusters de points", "Cercles proportionnels", "Heatmap"], index=0)
max_points = st.sidebar.slider("Limiter le nombre de points (performance)", 200, 50000, 8000, step=200)
default_center = [46.8, -71.2]  # Québec

# ---------- Préparation des données ----------
df_work = df.copy()
# convertit lat/lon en float
df_work["__lat__"] = df_work[lat_col].map(to_float)
df_work["__lon__"] = df_work[lon_col].map(to_float)
df_work = df_work.dropna(subset=["__lat__", "__lon__"])

if value_col != "(aucune)":
    df_work["__val__"] = df_work[value_col].map(to_float)
else:
    df_work["__val__"] = np.nan

n_total = len(df_work)
if n_total > max_points:
    df_work = df_work.sample(max_points, random_state=42)
    st.info(f"Affichage échantillonné : {len(df_work):,}/{n_total:,} points (réglable dans la barre latérale).")

# centre de carte
if len(df_work):
    center = [df_work["__lat__"].mean(), df_work["__lon__"].mean()]
else:
    center = default_center

# ---------- UI : aperçu ----------
st.title("🛰️ Application de prédiction d’inondation")
with st.expander("Aperçu des données (100 premières lignes)", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)

# ---------- Carte Folium ----------
m = folium.Map(location=center, zoom_start=6, control_scale=True, tiles=None)
# Fond satellite
folium.TileLayer(SATELLITE_TILE, name="Imagerie satellite (Esri)").add_to(m)
# Un fond clair en option
folium.TileLayer("CartoDB positron", name="Fond clair").add_to(m)

if view_mode == "Clusters de points":
    cluster = MarkerCluster(name="Stations / points").add_to(m)
    for _, r in df_work.iterrows():
        popup_html = "<br>".join([f"<b>{c}</b>: {r.get(c)}" for c in popup_cols])
        folium.Marker(
            location=[r["__lat__"], r["__lon__"]],
            popup=folium.Popup(popup_html, max_width=350),
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(cluster)

elif view_mode == "Cercles proportionnels":
    # taille relative sur la valeur (si fournie), sinon taille fixe
    v = df_work["__val__"].to_numpy()
    if np.isfinite(v).any():
        v_norm = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-9)
        radii = 4 + 16 * v_norm  # 4 à 20 px
    else:
        radii = np.full(len(df_work), 6.0)

    for (_, r), radius in zip(df_work.iterrows(), radii):
        popup_html = "<br>".join([f"<b>{c}</b>: {r.get(c)}" for c in popup_cols])
        folium.CircleMarker(
            location=[r["__lat__"], r["__lon__"]],
            radius=float(radius),
            color="#1f77b4",
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=350),
        ).add_to(m)

elif view_mode == "Heatmap":
    heat_data = df_work[["__lat__", "__lon__", "__val__"]].to_numpy()
    # si valeur absente, ne garder que lat/lon
    if np.isnan(heat_data[:, 2]).all():
        heat_data = df_work[["__lat__", "__lon__"]].to_numpy()
    HeatMap(heat_data, name="Heatmap", radius=20, blur=15, max_zoom=12).add_to(m)

folium.LayerControl(collapsed=True).add_to(m)

# rendu dans Streamlit
st_map = st_folium(m, width=None, height=650, returned_objects=[])

# ---------- Tips ----------
st.caption(
    "💡 Astuces :\n"
    "- Placez tous vos CSV dans le dossier `data/` du dépôt.\n"
    "- Choisissez les colonnes latitude/longitude dans la barre latérale.\n"
    "- Sélectionnez un champ de valeur pour des cercles proportionnels ou une heatmap.\n"
    "- Si vos nombres ont des virgules décimales, le code les gère automatiquement."
)
