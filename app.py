# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import folium
import branca
from streamlit_folium import st_folium

# ---------- Réglages généraux Streamlit ----------
st.set_page_config(page_title="Carte satellite + indicateurs", layout="wide")

st.title("🛰️ Application de prédiction / visualisation (Imagery Hybrid + indicateurs)")

DATA_DIR = "data"  # dossier où vous mettez vos CSV


# ---------- Utilitaires ----------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    # encodage robuste pour CSV divers
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # dernier essai par défaut
    return pd.read_csv(path)


def list_csvs(data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "*.csv")))


def make_imagery_hybrid(map_obj: folium.Map):
    """Ajoute l'imagerie Esri + la couche de labels (hybride)."""
    # Imagerie (fond)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Imagerie",
        overlay=False,
        control=True,
    ).add_to(map_obj)

    # Labels / limites / toponymes (overlay)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Reference Layer",
        name="Labels (hybride)",
        overlay=True,
        control=True,
    ).add_to(map_obj)


# ---------- Barre latérale ----------
st.sidebar.header("Données")
csv_files = list_csvs(DATA_DIR)
if not csv_files:
    st.sidebar.error("Aucun CSV trouvé dans le dossier `data/`.")
    st.stop()

csv_label_map = {os.path.basename(p): p for p in csv_files}
csv_name = st.sidebar.selectbox("Choisir un CSV dans `data/`", list(csv_label_map.keys()))
csv_path = csv_label_map[csv_name]

df = load_csv(csv_path)

st.sidebar.caption(f"**{csv_name}** chargé ({len(df):,} lignes, {len(df.columns)} colonnes)")

# Détection heuristique des colonnes lat/lon
def guess(colnames, keys):
    name_lower = {c.lower(): c for c in colnames}
    for k in keys:
        if k in name_lower:
            return name_lower[k]
    return None

lat_guess = guess(df.columns, ["lat_wgs84", "latitude", "lat", "y"])
lon_guess = guess(df.columns, ["long_wgs84", "longitude", "lon", "x"])

lat_col = st.sidebar.selectbox("Colonne Latitude", df.columns, index=df.columns.get_loc(lat_guess) if lat_guess in df.columns else 0)
lon_col = st.sidebar.selectbox("Colonne Longitude", df.columns, index=df.columns.get_loc(lon_guess) if lon_guess in df.columns else 0)

# Colonne valeur pour la couleur
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
val_col = st.sidebar.selectbox("Colonne valeur (pour la couleur)", numeric_cols if numeric_cols else df.columns)

# Paramètres d’indicateur
st.sidebar.header("Indicateur (couleur / taille)")
mode_couleur = st.sidebar.radio("Type de coloration", ["Continu", "Classes (seuils)"], horizontal=True)
min_r, max_r = st.sidebar.slider("Taille des points (rayon)", 4, 20, (5, 12))
opacity = st.sidebar.slider("Opacité de remplissage", 0.3, 1.0, 0.85)

# Classes (si choisi)
nb_classes = 5
seuils = None
if mode_couleur == "Classes (seuils)":
    nb_classes = st.sidebar.number_input("Nombre de classes", 3, 9, 5)
    # coupures automatiques quantiles
    try:
        seuils = list(np.unique(np.quantile(df[val_col].dropna().values, np.linspace(0, 1, nb_classes + 1))))
    except Exception:
        seuils = None

# Champs affichés au clic
st.sidebar.header("Info-bulle (popup)")
popup_cols = st.sidebar.multiselect("Colonnes à afficher", df.columns.tolist(), default=[c for c in df.columns[:4]])

# ---------- Aperçu ----------
st.subheader("Aperçu des données")
st.dataframe(df.head(), use_container_width=True)


# ---------- Carte ----------
# Boucle de sécurité : on filtre les lignes valides
df_valid = df.copy()
df_valid = df_valid.replace([np.inf, -np.inf], np.nan)
df_valid = df_valid.dropna(subset=[lat_col, lon_col])

if df_valid.empty:
    st.warning("Aucune ligne valide avec latitude/longitude.")
    st.stop()

# Centre carte sur le centroïde approximatif
center = [df_valid[lat_col].astype(float).mean(), df_valid[lon_col].astype(float).mean()]
m = folium.Map(location=center, zoom_start=6, control_scale=True)

# Basemap: Imagery Hybrid
make_imagery_hybrid(m)
# (optionnel) fond clair alternatif
folium.TileLayer("CartoDB positron", name="Fond clair").add_to(m)

# Palette / colormap
vmin, vmax = float(df_valid[val_col].min()), float(df_valid[val_col].max())

if mode_couleur == "Continu":
    cmap = branca.colormap.LinearColormap(
        colors=["#2c7fb8", "#ffff8c", "#d7191c"],  # bleu -> jaune -> rouge
        vmin=vmin, vmax=vmax
    )
else:
    # classes discrètes
    if not seuils or len(seuils) < 2:
        seuils = [vmin, *list(np.linspace(vmin, vmax, nb_classes)[1:-1]), vmax]
    colors = ["#2c7fb8", "#41ab5d", "#ffff8c", "#fdae61", "#d7191c", "#a50026"][: max(2, len(seuils) - 1)]
    def class_color(v: float):
        for i in range(len(seuils) - 1):
            if seuils[i] <= v <= seuils[i + 1]:
                return colors[i]
        return colors[-1]

# Ajout des points
for _, r in df_valid.iterrows():
    try:
        lat = float(r[lat_col])
        lon = float(r[lon_col])
    except Exception:
        continue

    # valeur pour style
    try:
        value = float(r[val_col])
    except Exception:
        value = np.nan

    if mode_couleur == "Continu":
        color = "#999999" if np.isnan(value) else branca.colormap.linear._to_hex(cmap(value))
    else:
        color = "#999999" if np.isnan(value) else class_color(value)

    # taille proportionnelle
    if np.isnan(value):
        radius = min_r
    else:
        radius = float(np.interp(value, [vmin, vmax], [min_r, max_r]))

    # popup HTML
    if popup_cols:
        html = r[popup_cols].to_frame().to_html(header=False)
    else:
        html = r.to_frame().head(12).to_html(header=False)

    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_opacity=opacity,
        weight=1,
        popup=folium.Popup(folium.IFrame(html=html, width=320, height=200), max_width=340),
    ).add_to(m)

# Légende
if mode_couleur == "Continu":
    cmap.caption = f"{val_col} (min–max)"
    cmap.add_to(m)
else:
    # légende manuelle discrète
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; padding: 10px 12px; border:1px solid #999; border-radius: 6px;">
      <b>""" + f"{val_col}" + """</b><br>
    """
    for i in range(len(seuils) - 1):
        legend_html += f'<i style="background:{colors[i]}; width:12px; height:12px; display:inline-block; margin-right:6px;"></i>'
        legend_html += f"{seuils[i]:.2f} – {seuils[i+1]:.2f}<br>"
    legend_html += "</div>"
    folium.map.CustomPane("legend").add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl(collapsed=False).add_to(m)

st.subheader("Carte (Imagery Hybrid + indicateurs)")
st_folium(m, height=720, use_container_width=True)
