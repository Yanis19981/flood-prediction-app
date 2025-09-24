# app.py
import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium import CircleMarker, TileLayer, LayerControl, FeatureGroup
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

# ---------- Réglages page & répertoire data ----------
st.set_page_config(page_title="Application de prédiction d’inondation – Carte",
                   layout="wide", page_icon="🌊")
DATA_DIR = "data"

# ---------- Utils ----------
def sniff_sep(sample: bytes) -> str:
    text = sample.decode("utf-8", errors="ignore")
    # ordre le plus courant
    for s in [",", ";", "\t", "|"]:
        if s in text:
            return s
    return ","  # fallback

def read_table_auto(file_like, nrows=None, sep_choice="auto", enc_choice="auto"):
    # lire petit échantillon pour deviner encodage/séparateur
    raw = file_like.read()
    if isinstance(raw, str):           # déjà string (cas GitHub raw)
        raw_bytes = raw.encode("utf-8", errors="ignore")
    else:
        raw_bytes = raw
    # remettre le curseur
    bio = io.BytesIO(raw_bytes)

    sep = sniff_sep(raw_bytes) if sep_choice == "auto" else sep_choice
    encoding = None if enc_choice == "auto" else enc_choice

    # 1er essai
    try:
        df = pd.read_csv(bio, sep=sep, nrows=nrows, encoding=encoding,
                         engine="python", on_bad_lines="skip", low_memory=False)
        return df
    except Exception:
        # 2e essai avec latin-1
        bio.seek(0)
        df = pd.read_csv(bio, sep=sep, nrows=nrows, encoding="latin-1",
                         engine="python", on_bad_lines="skip", low_memory=False)
        return df

def guess_lat_lon_columns(df):
    lat_candidates = [c for c in df.columns if c.lower() in
                      ("lat","latitude","lat_wgs84","lat_wgs","y","y_coord","ycoord")]
    lon_candidates = [c for c in df.columns if c.lower() in
                      ("lon","long","longitude","long_wgs84","long_wgs","x","x_coord","xcoord")]
    lat = lat_candidates[0] if lat_candidates else None
    lon = lon_candidates[0] if lon_candidates else None
    return lat, lon

def normalize_series(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.dropna()
    if s.empty:
        return None, None, None
    vmin, vmax = float(s.min()), float(s.max())
    return s, vmin, vmax

# ---------- Barre latérale : source des données ----------
st.sidebar.header("Données")

source = st.sidebar.radio(
    "Choisissez la source",
    ("CSV du dossier data/", "Téléverser un CSV"),
    horizontal=False,
)

# Limite de lignes pour garder l’app fluide
max_rows = st.sidebar.slider("Nombre max. de lignes à charger",
                             min_value=200, max_value=100000, step=200, value=5000)

sep_choice = st.sidebar.selectbox("Séparateur", ["auto", ",", ";", "\\t", "|"], index=0)
enc_choice = st.sidebar.selectbox("Encodage", ["auto", "utf-8", "latin-1"], index=0)

df = None
csv_name = None

if source == "CSV du dossier data/":
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not files:
        st.sidebar.warning("Aucun CSV trouvé dans le dossier 'data/'.")
    else:
        csv_name = st.sidebar.selectbox("Fichier CSV (data/)", files)
        with open(os.path.join(DATA_DIR, csv_name), "rb") as f:
            df = read_table_auto(f, nrows=max_rows, sep_choice=sep_choice, enc_choice=enc_choice)

else:
    up = st.sidebar.file_uploader("Déposer un CSV", type=["csv"])
    if up is not None:
        csv_name = up.name
        df = read_table_auto(up, nrows=max_rows, sep_choice=sep_choice, enc_choice=enc_choice)
        st.sidebar.success(f"Fichier chargé : {csv_name} ({len(df):,} lignes)")

# Stop si pas de données
if df is None or df.empty:
    st.title("Application de prédiction d’inondation – Carte")
    st.info("Sélectionnez ou téléversez un fichier CSV dans la barre latérale pour afficher la carte.")
    st.stop()

# ---------- Choix colonnes géographiques ----------
st.title("Application de prédiction d’inondation – Carte")

st.subheader("Colonnes géographiques")
auto_lat, auto_lon = guess_lat_lon_columns(df)

lat_col = st.selectbox("Colonne Latitude", [auto_lat] + list(df.columns) if auto_lat else list(df.columns),
                       index=0 if auto_lat else 0, key="latcol")
lon_col = st.selectbox("Colonne Longitude", [auto_lon] + list(df.columns) if auto_lon else list(df.columns),
                       index=0 if auto_lon else 0, key="loncol")

# Nettoyage lat/lon
df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
df_valid = df.dropna(subset=["_lat", "_lon"]).copy()
if df_valid.empty:
    st.error("Aucune ligne valide après conversion Latitude/Longitude.")
    st.stop()

# ---------- Indicateur couleur/taille ----------
st.subheader("Indicateur (couleur / taille)")
value_col = st.selectbox("Colonne valeur (pour la couleur/taille)", ["(aucune)"] + list(df_valid.columns))
color_mode = st.radio("Type de coloration", ["Continu", "Classes (quantiles)"], horizontal=True)
n_classes = st.slider("Nombre de classes", min_value=3, max_value=9, value=5, disabled=(color_mode!="Classes (quantiles)"))
base_radius = st.slider("Taille de base des points (rayon)", min_value=2, max_value=20, value=8)

# ---------- Préparation style ----------
use_value = value_col != "(aucune)"
if use_value:
    series_raw, vmin, vmax = normalize_series(df_valid[value_col])
    if series_raw is None:
        st.warning("La colonne choisie ne contient pas de valeurs numériques utilisables. Les points seront uniformes.")
        use_value = False
    else:
        df_valid["_value"] = pd.to_numeric(df_valid[value_col], errors="coerce")

# ---------- Carte Folium (grand format) ----------
st.subheader("Carte")
center = (float(df_valid["_lat"].astype(float).mean()),
          float(df_valid["_lon"].astype(float).mean()))

m = folium.Map(location=center, zoom_start=6, control_scale=True, tiles=None, prefer_canvas=True)

# Fond imagerie hybride (imagerie + labels overlay)
TileLayer(tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
          attr="Esri World Imagery", name="Imagerie (Esri)", control=True).add_to(m)
TileLayer(tiles="https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png",
          attr="© CARTO", name="Labels", control=True, overlay=True, opacity=0.9).add_to(m)

# Quelques fonds en plus
TileLayer("OpenStreetMap", name="OSM").add_to(m)
TileLayer("CartoDB positron", name="Topographie").add_to(m)

pts_layer = FeatureGroup(name="Stations / points", show=True).add_to(m)

# Color scale
if use_value:
    if color_mode == "Continu":
        cmap = LinearColormap(colors=["#2c7bb6", "#ffffbf", "#d7191c"], vmin=vmin, vmax=vmax)
        def color_fn(v):
            try:
                return cmap(float(v))
            except Exception:
                return "#3388ff"
        size_fn = lambda v: base_radius + 6 * (0 if vmax==vmin else (float(v)-vmin)/(vmax-vmin))
    else:
        cats = pd.qcut(df_valid["_value"], q=n_classes, duplicates="drop")
        bins = sorted(set(cats.cat.categories.left.tolist() + [cats.cat.categories.right.tolist()[-1]]))
        cmap = LinearColormap(colors=["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"][:len(bins)-1],
                              vmin=bins[0], vmax=bins[-1])
        def color_fn(v):
            return cmap(float(v))
        size_fn = lambda v: base_radius + 6 * (0 if (bins[-1]-bins[0]==0) else (float(v)-bins[0])/(bins[-1]-bins[0]))
else:
    color_fn = lambda v: "#3388ff"
    size_fn = lambda v: base_radius

# Ajout des points
for _, r in df_valid.iterrows():
    lat, lon = float(r["_lat"]), float(r["_lon"])
    val = r[value_col] if use_value else None
    color = color_fn(val)
    radius = float(size_fn(val)) if use_value else base_radius

    popup_cols = [c for c in df_valid.columns if not c.startswith("_")]
    popup_html = "<br>".join([f"<b>{c}</b>: {r[c]}" for c in popup_cols[:20]])

    CircleMarker(
        location=(lat, lon),
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        weight=1,
        popup=folium.Popup(popup_html, max_width=400)
    ).add_to(pts_layer)

LayerControl(collapsed=False).add_to(m)

# 👉 grande carte : height ~ 750px, pleine largeur
st_folium(m, use_container_width=True, height=750)
