# -*- coding: utf-8 -*-
import io
import os
import re
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm

# -----------------------------
# Configuration générale
# -----------------------------
st.set_page_config(page_title="Application de prédiction d’inondation – Carte",
                   layout="wide",
                   page_icon="🌊")

st.title("Application de prédiction d’inondation – Carte")

DATA_DIR = "data"   # dossier où tu ranges tes CSV dans le repo

# -----------------------------
# Utilitaires de lecture robuste
# -----------------------------
def sniff_sep(raw_bytes: bytes) -> str:
    """Devine le séparateur dominant (parmi ; , \t |)"""
    sample = raw_bytes[:20000].decode("utf-8", errors="ignore")
    # tester dans l’ordre courant (souvent , ou ;)
    cands = [",", ";", "\t", "|"]
    counts = {c: sample.count(c) for c in cands}
    # si aucune virgule/point-virgule/tab/pipe n’apparaît, pandas saura gérer
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","

def _read_csv_bytes(raw_bytes: bytes, nrows=None, sep="auto", enc="auto"):
    """
    Lecture CSV tolérante aux erreurs (compatible pandas Cloud).
    Essaye plusieurs combinaisons sans figer engine="python".
    """
    # normaliser sep
    if sep == "\\t":
        sep = "\t"
    if sep == "auto":
        sep = sniff_sep(raw_bytes)

    def try_read(enc_opt, with_bad_lines=True):
        bio = io.BytesIO(raw_bytes)
        kwargs = dict(sep=sep, nrows=nrows, low_memory=False, encoding=enc_opt)
        # pandas récent accepte on_bad_lines="skip"
        if with_bad_lines:
            kwargs["on_bad_lines"] = "skip"
            kwargs["encoding_errors"] = "ignore"
        return pd.read_csv(bio, **kwargs)

    # 1) enc fourni
    if enc != "auto":
        try:
            return try_read(enc, with_bad_lines=True)
        except Exception:
            try:
                return try_read(enc, with_bad_lines=False)
            except Exception:
                pass

    # 2) auto : essayer utf-8 puis latin-1
    for enc_try in ("utf-8", "latin-1"):
        try:
            return try_read(enc_try, with_bad_lines=True)
        except Exception:
            try:
                return try_read(enc_try, with_bad_lines=False)
            except Exception:
                continue

    # dernier recours : laisser pandas décider sans encodage
    bio = io.BytesIO(raw_bytes)
    return pd.read_csv(bio, sep=sep, nrows=nrows, low_memory=False)

@st.cache_data(show_spinner=False)
def read_table_auto(file_like, nrows=None, sep_choice="auto", enc_choice="auto"):
    """Wrapper cache pour lire fichiers uploadés (ou issus de /data)."""
    raw = file_like.read()
    raw_bytes = raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8", errors="ignore")
    return _read_csv_bytes(raw_bytes, nrows=nrows, sep=sep_choice, enc=enc_choice)

# -----------------------------
# Barre latérale : choix des données
# -----------------------------
with st.sidebar:
    st.header("Données")

    # lister les CSV présents dans /data
    data_files = []
    if os.path.isdir(DATA_DIR):
        data_files = [f for f in os.listdir(DATA_DIR)
                      if f.lower().endswith(".csv")]
        data_files.sort()

    source_mode = st.radio("Source", ("Depuis data/", "Uploader"), horizontal=True)
    max_rows = st.slider("Nombre max. de lignes à charger", 100, 5000, 5000, step=100)

    col_sep = st.selectbox("Séparateur", ["auto", ",", ";", "\\t", "|"], index=0)
    col_enc = st.selectbox("Encodage", ["auto", "utf-8", "latin-1"], index=0)

    uploaded_file = None
    csv_name = None
    if source_mode == "Depuis data/":
        if len(data_files) == 0:
            st.warning("Aucun CSV trouvé dans le dossier `data/` du dépôt.")
        csv_name = st.selectbox("Choisir un CSV dans data/", data_files) if len(data_files) else None
    else:
        uploaded_file = st.file_uploader("Glisser-déposer un CSV", type=["csv"])

# lire le DataFrame choisi
df = None
if source_mode == "Depuis data/" and csv_name:
    path = os.path.join(DATA_DIR, csv_name)
    with open(path, "rb") as f:
        df = read_table_auto(f, nrows=max_rows, sep_choice=col_sep, enc_choice=col_enc)
elif source_mode == "Uploader" and uploaded_file is not None:
    df = read_table_auto(uploaded_file, nrows=max_rows, sep_choice=col_sep, enc_choice=col_enc)

if df is None:
    st.info("Sélectionne un fichier CSV (dans `data/` ou par Uploader) pour afficher la carte.")
    st.stop()

# -----------------------------
# Détection des colonnes Geo
# -----------------------------
def find_col(candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

lat_guess = find_col(["lat", "latitude", "lat_wgs84", "Latitude"])
lon_guess = find_col(["lon", "lng", "longitude", "long_wgs84", "Longitude"])

st.subheader("Colonnes géographiques")
c1, c2, c3 = st.columns([1,1,2])

with c1:
    lat_col = st.selectbox("Colonne Latitude", [ "— choisir —" ] + list(df.columns), 
                           index=(list(df.columns).index(lat_guess) + 1 if lat_guess in df.columns else 0))
with c2:
    lon_col = st.selectbox("Colonne Longitude", [ "— choisir —" ] + list(df.columns), 
                           index=(list(df.columns).index(lon_guess) + 1 if lon_guess in df.columns else 0))
with c3:
    popup_cols = st.multiselect("Colonnes à afficher dans le popup", list(df.columns))

if lat_col == "— choisir —" or lon_col == "— choisir —":
    st.warning("Choisis les colonnes Latitude et Longitude.")
    st.stop()

# nettoyage/filtrage géo
df_valid = df.copy()
df_valid = df_valid[(df_valid[lat_col].astype(str).str.strip() != "") &
                    (df_valid[lon_col].astype(str).str.strip() != "")]
# conversion num
df_valid[lat_col] = pd.to_numeric(df_valid[lat_col], errors="coerce")
df_valid[lon_col] = pd.to_numeric(df_valid[lon_col], errors="coerce")
df_valid = df_valid.dropna(subset=[lat_col, lon_col])
if df_valid.empty:
    st.error("Aucune ligne valide (Latitude/Longitude).")
    st.stop()

# -----------------------------
# Indicateurs couleur / taille
# -----------------------------
st.subheader("Indicateur (couleur / taille)")
cval_col = st.selectbox("Colonne valeur (pour couleur/taille) – optionnel", ["(aucune)"] + list(df.columns))

mode_color = st.radio("Type de coloration", ["Continu", "Classes (quantiles)"], horizontal=True)
nb_classes = st.slider("Nombre de classes (si quantiles)", 3, 9, 5) if mode_color == "Classes (quantiles)" else None
base_radius = st.slider("Taille de base des points (rayon)", 2, 14, 6)

# préparation palette
values = None
cmap = None
bins = None
if cval_col != "(aucune)":
    try:
        values = pd.to_numeric(df_valid[cval_col], errors="coerce")
        mask = values.notna()
        vmin, vmax = values[mask].min(), values[mask].max()
        if mode_color == "Continu":
            cmap = cm.LinearColormap(colors=["#2c7bb6", "#ffffbf", "#d7191c"], vmin=vmin, vmax=vmax)
        else:
            # classes quantiles
            qs = np.linspace(0, 1, nb_classes+1)
            bins = np.unique(np.quantile(values[mask], qs))
            # si valeurs peu variées
            if len(bins) < 3:
                bins = np.linspace(vmin, vmax, 4)
            cmap = cm.LinearColormap(colors=["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"],
                                     vmin=vmin, vmax=vmax)
    except Exception:
        values = None
        cmap = None
        bins = None

# -----------------------------
# Carte Folium (Imagery Hybrid)
# -----------------------------
st.subheader("Carte")

# centre sur moyenne
center = [float(df_valid[lat_col].astype(float).mean()),
          float(df_valid[lon_col].astype(float).mean())]

# carte large
m = folium.Map(location=center, zoom_start=6, control_scale=True)

# Esri World Imagery (fond imagerie)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr='Esri — World Imagery',
    name="Imagerie (Esri)",
    overlay=False,
    control=True,
).add_to(m)

# Overlay labels pour le mode hybride
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Esri — Reference (labels)",
    name="Labels",
    overlay=True,
    control=True,
    opacity=0.9,
).add_to(m)

# Topographie (Esri)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri — World Topo",
    name="Topographie",
    overlay=False,
    control=True,
).add_to(m)

# OSM
folium.TileLayer("OpenStreetMap", name="OSM", overlay=False, control=True).add_to(m)

# clusters
cluster = MarkerCluster(name="Stations / points", control=True, show=True)
cluster.add_to(m)

def row_popup(row):
    if not popup_cols:
        return None
    items = []
    for c in popup_cols:
        try:
            val = row.get(c, "")
        except Exception:
            val = ""
        items.append(f"<b>{c}</b>: {val}")
    return folium.Popup(html="<br>".join(items), max_width=450)

# ajout des points
for _, r in df_valid.iterrows():
    lat, lon = float(r[lat_col]), float(r[lon_col])

    # couleur et taille
    color = "#2E86AB"
    radius = base_radius
    if values is not None and np.isfinite(r.get(cval_col, np.nan)):
        val = float(r[cval_col])
        if mode_color == "Continu":
            color = cmap(val)
        else:
            # trouver classe
            idx = np.digitize([val], bins=bins, right=False)[0] - 1
            idx = max(0, min(idx, len(bins)-2))
            # normaliser vers [0..1] pour cmap
            frac = 0 if bins[-1] == bins[0] else (val - bins[0]) / (bins[-1] - bins[0])
            color = cmap(frac)
        # grossir légèrement avec la valeur
        radius = base_radius + 2

    folium.CircleMarker(
        location=(lat, lon),
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=row_popup(r)
    ).add_to(cluster)

folium.LayerControl(collapsed=False).add_to(m)

# affichage
st_folium(m, width=None, height=720)  # carte plus grande

# --- Téléchargement CSV filtré ---
st.subheader("Téléchargements")
csv_bytes = df_valid.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ Télécharger le CSV filtré (lat/lon valides)",
    data=csv_bytes,
    file_name="donnees_filtrees.csv",
    mime="text/csv"
)


# --- Export HTML de la carte Folium ---
html_str = m.get_root().render()
html_bytes = html_str.encode("utf-8")

st.download_button(
    "⬇️ Télécharger la carte (HTML interactif)",
    data=html_bytes,
    file_name="carte_inondation.html",
    mime="text/html"
)


# --- Export GeoJSON des points ---
import json

features = []
for _, r in df_valid.iterrows():
    try:
        lat, lon = float(r[lat_col]), float(r[lon_col])
    except Exception:
        continue
    props = {k: (None if pd.isna(r[k]) else r[k]) for k in df_valid.columns if k not in [lat_col, lon_col]}
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props
    })

geojson = {"type": "FeatureCollection", "features": features}
geojson_bytes = json.dumps(geojson, ensure_ascii=False).encode("utf-8")

st.download_button(
    "⬇️ Télécharger les points (GeoJSON)",
    data=geojson_bytes,
    file_name="points_inondation.geojson",
    mime="application/geo+json"
)

