import streamlit as st
import pandas as pd
from pathlib import Path
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from branca.colormap import linear

st.set_page_config(page_title="Carte satellite + indicateurs", layout="wide")

DATA_DIR = Path("data")
st.title("Application de prédiction d’inondation – Carte")

# --------- Sélection du CSV ----------
csv_files = sorted([p.name for p in DATA_DIR.glob("*.csv")])
if not csv_files:
    st.warning("Aucun fichier CSV trouvé dans le dossier **data/**.")
    st.stop()

with st.sidebar:
    st.header("Données")
    csv_name = st.selectbox("Choisissez un CSV (dans data/)", csv_files, index=0)
    nrows = st.slider("Nombre max. de lignes à charger", 500, 50000, 5000, step=500)
    st.caption("Augmentez seulement si l’app reste fluide.")

# --------- Lecture CSV (simple & rapide) ----------
def read_csv_fast(path: Path, nrows: int) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(
                path, nrows=nrows, low_memory=False,
                on_bad_lines="skip", encoding=enc, engine="python"
            )
        except Exception:
            pass
    # dernier essai, sans encodage explicite
    return pd.read_csv(path, nrows=nrows, low_memory=False, on_bad_lines="skip", engine="python")

df = read_csv_fast(DATA_DIR / csv_name, nrows=nrows)
st.success(f"CSV chargé : {len(df):,} lignes, {df.shape[1]} colonnes")
st.dataframe(df.head(10), use_container_width=True)

# --------- Choix des colonnes géographiques ----------
def guess(colnames, candidates):
    for c in candidates:
        for name in colnames:
            if name.lower().strip() == c.lower().strip():
                return name
    return None

colnames = list(df.columns)

lat_guess = guess(colnames, ["lat", "latitude", "lat_wgs84", "lat_wgs", "Latitude", "LAT"])
lon_guess = guess(colnames, ["lon", "lng", "longitude", "long_wgs84", "long_wgs", "Longitude", "LON"])

left, right = st.columns(2)
with left:
    lat_col = st.selectbox("Colonne Latitude", colnames, index=colnames.index(lat_guess) if lat_guess in colnames else 0)
with right:
    lon_col = st.selectbox("Colonne Longitude", colnames, index=colnames.index(lon_guess) if lon_guess in colnames else 0)

# --------- Colonne valeur (pour la couleur) ----------
num_cols = [c for c in colnames if pd.api.types.is_numeric_dtype(df[c])]

with st.sidebar:
    st.header("Indicateur (couleur / taille)")
    val_col = st.selectbox("Colonne valeur (optionnel, pour la couleur)", ["(aucune)"] + num_cols, index=0)
    radius = st.slider("Rayon des points", 3, 14, 6)

# Nettoyage de base
df = df.copy()
df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
df_valid = df.dropna(subset=[lat_col, lon_col])

if val_col != "(aucune)":
    df_valid[val_col] = pd.to_numeric(df_valid[val_col], errors="coerce")

if df_valid.empty:
    st.error("Aucune ligne valide avec latitude/longitude.")
    st.stop()

# --------- Carte Folium (Imagery + Hybrid) ----------
center = [df_valid[lat_col].mean(), df_valid[lon_col].mean()]
m = folium.Map(location=center, zoom_start=6, tiles=None)

# Fond OSM (au cas où)
folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr="© OpenStreetMap", name="OpenStreetMap"
).add_to(m)

# Esri World Imagery
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    name="Esri World Imagery"
).add_to(m)

# Esri Hybrid (labels) – overlay
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Labels © Esri",
    name="Hybrid labels", overlay=True, control=True
).add_to(m)

mc = MarkerCluster(name="Points").add_to(m)

# Palette couleur si valeur choisie
if val_col != "(aucune)":
    vmin, vmax = float(df_valid[val_col].min()), float(df_valid[val_col].max())
    if vmin == vmax:  # éviter une palette dégénérée
        vmin, vmax = 0.0, vmin or 1.0
    cmap = linear.OrRd_09.scale(vmin, vmax)
else:
    cmap = None

# popup : on met quelques colonnes informatives si dispo
popup_cols = [c for c in ["rapport", "nom_mandataire", "version", "date_heure_obs"] if c in df_valid.columns]

for _, r in df_valid.iterrows():
    lat, lon = float(r[lat_col]), float(r[lon_col])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        continue

    color = cmap(r[val_col]) if (cmap is not None and pd.notna(r.get(val_col))) else "#3388ff"

    if popup_cols:
        html = "<br>".join(f"<b>{c}:</b> {r[c]}" for c in popup_cols)
    else:
        html = f"{lat_col}: {lat:.5f}<br>{lon_col}: {lon:.5f}"

    folium.CircleMarker(
        location=[lat, lon], radius=radius,
        color=color, fill=True, fill_color=color, fill_opacity=0.85,
        popup=folium.Popup(html, max_width=400),
    ).add_to(mc)

if cmap is not None:
    cmap.caption = val_col
    cmap.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width=None, height=720)
