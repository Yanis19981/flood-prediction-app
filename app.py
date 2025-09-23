import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap, StepColormap
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ---------- Utilitaires ----------
CANDIDATE_LAT = ["lat", "latitude", "lat_wgs84", "y", "lat_deg", "geom_lat"]
CANDIDATE_LON = ["lon", "lng", "longitude", "long_wgs84", "x", "lon_deg", "geom_lon"]
NUM_CANDIDATES = [
    "alt_m_cgvd", "altitude", "valeur", "value", "hauteur_eau", "water_level",
    "No_tuile1", "No_tuile2", "magnitude", "score"
]

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", s.lower())

def guess_column(df: pd.DataFrame, candidates):
    cols = {norm(c): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    for k, v in cols.items():
        if any(c in k for c in candidates):
            return v
    return None

def list_csvs(data_dir="data"):
    if not os.path.isdir(data_dir):
        return []
    return [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]

# ---------- UI ----------
st.set_page_config(layout="wide")
st.title("🛰️ Application de prédiction d’inondation")

left, right = st.columns([0.35, 0.65], gap="large")

with left:
    st.header("Données")

    # Permet l'upload direct (utile sur Streamlit Cloud)
    uploaded = st.file_uploader("Importer un CSV", type=["csv"])

    # Fallback: lister les CSV du dossier local data/
    csv_files = list_csvs("data")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        csv_choice = f"(upload) {uploaded.name}"
    else:
        if not csv_files:
            st.error("Aucun CSV trouvé et aucun fichier importé. "
                     "Uploade un CSV ou ajoute-en un dans le dossier `data/`.")
            st.stop()
        csv_choice = st.selectbox("Choisir un CSV dans `data/`", options=csv_files, index=0)
        path = os.path.join("data", csv_choice)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            st.error(f"Lecture impossible de `{csv_choice}` : {e}")
            st.stop()

    # Colonnes candidates auto
    lat_auto = guess_column(df, CANDIDATE_LAT)
    lon_auto = guess_column(df, CANDIDATE_LON)

    st.subheader("Colonnes géographiques")
    lat_col = st.selectbox(
        "Colonne Latitude",
        options=["<auto>"] + list(df.columns),
        index=0 if lat_auto is None else list(df.columns).index(lat_auto) + 1
    )
    lon_col = st.selectbox(
        "Colonne Longitude",
        options=["<auto>"] + list(df.columns),
        index=0 if lon_auto is None else list(df.columns).index(lon_auto) + 1
    )
    if lat_col == "<auto>":
        lat_col = lat_auto
    if lon_col == "<auto>":
        lon_col = lon_auto

    if lat_col is None or lon_col is None:
        st.warning("Impossible de détecter les colonnes latitude/longitude. "
                   "Sélectionne-les manuellement.")
        st.stop()

    # Colonne numérique pour couleur/taille
    st.subheader("Indicateur (couleur / taille)")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    guess_num = None
    for cand in NUM_CANDIDATES:
        for c in df.columns:
            if cand in norm(c) and c in numeric_cols:
                guess_num = c
                break
        if guess_num:
            break

    val_col = st.selectbox(
        "Colonne valeur (pour la couleur)",
        options=["(aucune)"] + numeric_cols,
        index=0 if guess_num is None else numeric_cols.index(guess_num) + 1
    )

    color_mode = st.radio("Mode de coloration", ["Continu", "Classes (seuils)"], horizontal=True)
    point_radius = st.slider("Taille des points (rayon)", 3, 18, 8)

    # Champs à afficher dans le popup
    st.subheader("Colonnes à afficher dans le popup")
    default_popup = [c for c in df.columns[:4]]
    popup_cols = st.multiselect("Champs popup", options=list(df.columns), default=default_popup)

with right:
    st.header("Carte")

    # Nettoyage / casting
    keep_cols = [lat_col, lon_col] + ([val_col] if val_col != "(aucune)" else []) + popup_cols
    keep_cols = list(dict.fromkeys(keep_cols))  # unique, conserve l'ordre
    df_valid = df[keep_cols].dropna(subset=[lat_col, lon_col]).copy()

    for c in [lat_col, lon_col]:
        df_valid[c] = pd.to_numeric(df_valid[c], errors="coerce")
    df_valid = df_valid.dropna(subset=[lat_col, lon_col])

    if df_valid.empty:
        st.warning("Aucune ligne valide avec lat/lon.")
        st.stop()

    # Centre de la carte
    center = [
        df_valid[lat_col].astype(float).mean(),
        df_valid[lon_col].astype(float).mean()
    ]

    # Carte Folium
    m = folium.Map(location=center, zoom_start=6, tiles=None)

    # ---- Fond "Imagery Hybrid / Imagerie Hybride" ----
    # Imagerie
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri | World Imagery', name='Imagery'
    ).add_to(m)
    # Labels (overlay)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        attr='Esri | Boundaries & Places', name='Labels', overlay=True, control=True
    ).add_to(m)
    # Optionnel : fond clair
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)

    # Colormap
    cmap = None
    if val_col != "(aucune)":
        v = pd.to_numeric(df_valid[val_col], errors="coerce")
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
            if color_mode.startswith("Continu"):
                cmap = LinearColormap(
                    colors=["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"],
                    vmin=vmin, vmax=vmax
                )
                cmap.caption = val_col
            else:
                bins = np.linspace(vmin, vmax, 6)
                cmap = StepColormap(
                    colors=["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"],
                    vmin=vmin, vmax=vmax, index=bins
                )
                cmap.caption = f"{val_col} (classes)"
            cmap.add_to(m)

    # Cluster
    cluster = MarkerCluster(name="Stations / points", disableClusteringAtZoom=10).add_to(m)

    # Ajout des points
    for _, r in df_valid.iterrows():
        lat, lon = float(r[lat_col]), float(r[lon_col])

        # couleur
        if cmap is not None and (val_col in r) and pd.notna(r[val_col]):
            color = cmap(float(r[val_col]))
        else:
            color = "#3186cc"

        # popup HTML
        if popup_cols:
            rows = "".join(
                f"<tr><th style='text-align:left;padding-right:6px'>{c}</th><td>{r.get(c)}</td></tr>"
                for c in popup_cols
            )
            html = f"<table>{rows}</table>"
        else:
            html = f"({lat:.5f}, {lon:.5f})"

        folium.CircleMarker(
            location=(lat, lon),
            radius=point_radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(html, max_width=350),
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=700, width=None)
