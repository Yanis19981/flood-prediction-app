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
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def list_csvs(data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "*.csv")))

def make_imagery_hybrid(map_obj: folium.Map):
    """Ajoute l'imagerie Esri + la couche de labels (hybride)."""
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Imagerie",
        overlay=False,
        control=True,
    ).add_to(map_obj)
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

# Champs affichés au clic
st.sidebar.header("Info-bulle (popup)")
popup_cols = st.sidebar.multiselect("Colonnes à afficher", df.columns.tolist(), default=[c for c in df.columns[:4]])

# ---------- Aperçu ----------
st.subheader("Aperçu des données")
st.dataframe(df.head(), use_container_width=True)

# ---------- Carte ----------
# Lignes valides + conversion
df_valid = df.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=[lat_col, lon_col])
if df_valid.empty:
    st.warning("Aucune ligne valide avec latitude/longitude.")
    st.stop()

# Conversion numérique robuste pour la colonne valeur
vals = pd.to_numeric(df_valid[val_col], errors="coerce")
if vals.notna().sum() == 0:
    st.error("La colonne sélectionnée pour la couleur n'est pas numérique (ou toutes les valeurs sont NaN).")
    st.stop()

vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
same_range = not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax

# Centre carte
center = [pd.to_numeric(df_valid[lat_col], errors="coerce").mean(),
          pd.to_numeric(df_valid[lon_col], errors="coerce").mean()]
m = folium.Map(location=center, zoom_start=6, control_scale=True)

# Basemap
make_imagery_hybrid(m)
folium.TileLayer("CartoDB positron", name="Fond clair").add_to(m)

# Palette / colormap
if mode_couleur == "Continu":
    if same_range:
        cmap = None
    else:
        cmap = branca.colormap.LinearColormap(
            colors=["#2c7fb8", "#ffff8c", "#d7191c"], vmin=vmin, vmax=vmax
        )
else:
    # seuils basés sur les quantiles de la colonne numérique
    q = vals.dropna().values
    if q.size == 0:
        seuils = None
    else:
        qs = np.linspace(0, 1, nb_classes + 1)
        seuils = list(np.unique(np.quantile(q, qs)))
        if len(seuils) < 2:  # fallback si tout est identique
            seuils = [vmin, vmax]
    colors = ["#2c7fb8", "#41ab5d", "#ffff8c", "#fdae61", "#d7191c", "#a50026"][: max(2, (len(seuils) - 1) if seuils else 2)]
    def class_color(v: float):
        if not seuils or len(seuils) < 2:
            return colors[-1]
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

    value = pd.to_numeric(r.get(val_col), errors="coerce")

    # couleur
    if mode_couleur == "Continu":
        if cmap is None or pd.isna(value):
            color = "#999999"
        else:
            color = cmap(float(value))   # ✅ plus de _to_hex
    else:
        color = "#999999" if pd.isna(value) else class_color(float(value))

    # taille
    if pd.notna(value) and not same_range:
        radius = float(np.interp(float(value), [vmin, vmax], [min_r, max_r]))
    else:
        radius = min_r

    # popup
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
if mode_couleur == "Continu" and (locals().get("cmap") is not None):
    cmap.caption = f"{val_col} (min–max)"
    cmap.add_to(m)
elif mode_couleur == "Classes (seuils)" and seuils and len(seuils) >= 2:
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; padding: 10px 12px; border:1px solid #999; border-radius: 6px;">
      <b>""" + f"{val_col}" + """</b><br>
    """
    for i in range(len(seuils) - 1):
        legend_html += f'<i style="background:{colors[i]}; width:12px; height:12px; display:inline-block; margin-right:6px;"></i>'
        legend_html += f"{seuils[i]:.2f} – {seuils[i+1]:.2f}<br>"
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl(collapsed=False).add_to(m)
st.subheader("Carte (Imagery Hybrid + indicateurs)")
st_folium(m, height=720, use_container_width=True)
