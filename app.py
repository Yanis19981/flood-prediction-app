# app.py
import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Application de prédiction d’inondation",
                   layout="wide")

st.title("📊 Application de prédiction d’inondation")

# -------------------------------
# 1) Trouver les CSV disponibles
# -------------------------------
def list_csvs():
    in_data = sorted(glob.glob("data/*.csv"))
    in_root = sorted(glob.glob("*.csv"))
    # Ne pas dupliquer si même nom
    seen = set()
    files = []
    for p in in_data + in_root:
        n = os.path.basename(p)
        if n not in seen:
            files.append(p)
            seen.add(n)
    return files

csv_files = list_csvs()
if not csv_files:
    st.warning("Aucun CSV trouvé. Place tes fichiers dans `data/` ou à la racine du repo.")
    st.stop()

# -------------------------------
# 2) Lecture CSV (cache)
# -------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8")
    return df

left, right = st.columns([0.32, 0.68])
with left:
    st.subheader("Données")
    # On filtre de préférence dans data/
    only_in_data = [p for p in csv_files if p.startswith("data/")]
    folder_label = "data/" if only_in_data else "(racine du repo)"
    st.caption(f"Choisir un CSV dans **{folder_label}**")
    path = st.selectbox("Fichiers CSV", options=only_in_data or csv_files, index=0,
                        format_func=lambda p: os.path.basename(p))
    df = load_csv(path)
    st.caption(f"{os.path.basename(path)} chargé ({len(df):,} lignes, {len(df.columns)} colonnes)")

# ---------------------------------------------------------
# 3) Détection/choix des colonnes Latitude / Longitude
# ---------------------------------------------------------
def auto_pick(candidates, columns):
    for c in candidates:
        if c in columns:
            return c
    return None

# Propositions courantes
lat_candidates = ["lat_wgs84", "Latitude", "latitude", "lat", "y", "Y", "Lat"]
lon_candidates = ["long_wgs84", "Longitude", "longitude", "lon", "x", "X", "Lon", "long"]

with left:
    st.subheader("Colonnes géographiques")

    lat_col = st.selectbox(
        "Colonne Latitude",
        options=[c for c in df.columns] + ["<auto>"],
        index=(df.columns.tolist() + ["<auto>"]).index("<auto>"),
    )
    lon_col = st.selectbox(
        "Colonne Longitude",
        options=[c for c in df.columns] + ["<auto>"],
        index=(df.columns.tolist() + ["<auto>"]).index("<auto>"),
    )

# Auto-détection si demandé
if lat_col == "<auto>":
    lat_col = auto_pick(lat_candidates, df.columns)
if lon_col == "<auto>":
    lon_col = auto_pick(lon_candidates, df.columns)

# Vérifs
if lat_col is None or lon_col is None:
    with right:
        st.error(
            "Aucune colonne de **latitude/longitude** détectée dans ce CSV.\n\n"
            "- Utilise un fichier avec des coordonnées (ex. `lat_wgs84` / `long_wgs84`)\n"
            "- OU fournis un CSV de *centroïdes* si ton fichier a seulement des codes de tuiles."
        )
    st.stop()

# Conversion tolérante (virgules -> points, strings -> float)
lat_num = pd.to_numeric(df[lat_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
lon_num = pd.to_numeric(df[lon_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")

df_valid = df.copy()
df_valid["_lat"] = lat_num
df_valid["_lon"] = lon_num
df_valid = df_valid.dropna(subset=["_lat", "_lon"])

if df_valid.empty:
    with right:
        st.error(
            f"Les colonnes repérées `{lat_col}` / `{lon_col}` n'ont pas de valeurs numériques valides.\n"
            "Vérifie ton CSV ou fournis un CSV de centroïdes."
        )
    st.stop()

# ---------------------------------------------------------
# 4) Colonne de valeur pour couleur / taille (optionnel)
# ---------------------------------------------------------
with left:
    st.subheader("Indicateur (couleur / taille)")

    value_col = st.selectbox(
        "Colonne valeur (pour la couleur)",
        options=["(aucune)"] + df.columns.tolist(),
        index=0
    )

    color_mode = st.radio("Type de coloration", ["Continu", "Classes (seuils)"], horizontal=False)
    point_radius = st.slider("Taille des points (rayon)", 3, 12, 7)

    # Pour le popup
    st.subheader("Colonnes à afficher dans le popup")
    popup_cols = st.multiselect(
        "Choisis 1 à 4 colonnes",
        options=df.columns.tolist(),
        default=df.columns[:2].tolist()
    )

# Préparer valeur numérique si demandé
val_series = None
if value_col != "(aucune)":
    val_series = pd.to_numeric(df_valid[value_col].astype(str).str.replace(",", ".", regex=False),
                               errors="coerce")

# ---------------------------------------------------------
# 5) Carte Folium - base “Imagery Hybrid / Imagerie Hybride”
# ---------------------------------------------------------
def add_imagery_hybrid(m):
    """
    Imagery + calque des noms (hybride Esri).
    """
    # Imagerie Esri
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, GeoEye, Earthstar, CNES/Airbus DS, USDA, USGS, AeroGRID",
        name="Imagery (Esri)",
        overlay=False,
        control=True,
        max_zoom=19
    ).add_to(m)

    # Calque des noms/limites (overlay)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Labels (Esri)",
        overlay=True,
        control=True,
        max_zoom=19
    ).add_to(m)

# Centre carte
center = [df_valid["_lat"].mean(), df_valid["_lon"].mean()]
m = folium.Map(location=center, zoom_start=6, control_scale=True, tiles=None)
add_imagery_hybrid(m)
folium.LayerControl(position="topleft").add_to(m)

# Cluster
cluster = MarkerCluster().add_to(m)

# ---------------------------------------------------------
# 6) Couleurs (continu ou classes)
# ---------------------------------------------------------
def make_color_func(vals):
    """
    Renvoie une fonction color(value)->hex.
    - Continu: dégradé du vert (#2ECC71) -> jaune (#F1C40F) -> rouge (#E74C3C)
    - Classes: 5 classes par défaut (modifiable)
    """
    if vals is None:
        # Pas de valeur -> point bleu
        return lambda _: "#2E86DE", None

    arr = vals.to_numpy(dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return lambda _: "#2E86DE", None

    vmin, vmax = np.nanmin(arr[valid]), np.nanmax(arr[valid])
    if vmin == vmax:
        # Valeur constante
        return lambda _: "#2E86DE", (vmin, vmax)

    if color_mode == "Continu":
        def _interp_color(v):
            if v is None or not np.isfinite(v):
                return "#95A5A6"  # gris
            t = (v - vmin) / (vmax - vmin)
            # 0..0.5 : vert->jaune ; 0.5..1 : jaune->rouge
            if t <= 0.5:
                # vert(46,204,113) -> jaune(241,196,15)
                u = t / 0.5
                r = int(46 + (241 - 46) * u)
                g = int(204 + (196 - 204) * u)
                b = int(113 + (15 - 113) * u)
            else:
                # jaune(241,196,15) -> rouge(231,76,60)
                u = (t - 0.5) / 0.5
                r = int(241 + (231 - 241) * u)
                g = int(196 + (76 - 196) * u)
                b = int(15 + (60 - 15) * u)
            return f"#{r:02X}{g:02X}{b:02X}"

        return _interp_color, (vmin, vmax)

    else:
        # Classes (5 par défaut)
        n = 5
        bins = np.linspace(vmin, vmax, n + 1)
        palette = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C", "#8E44AD"]
        def _class_color(v):
            if v is None or not np.isfinite(v):
                return "#95A5A6"
            k = np.searchsorted(bins, v, side="right") - 1
            k = int(np.clip(k, 0, n - 1))
            return palette[k]
        return _class_color, bins

color_func, scale = make_color_func(val_series)

# ---------------------------------------------------------
# 7) Ajout des points (popup + couleur + taille)
# ---------------------------------------------------------
def make_popup(row):
    if not popup_cols:
        return None
    parts = []
    for c in popup_cols[:4]:
        if c in row:
            parts.append(f"<b>{c}</b>: {row[c]}")
    return "<br>".join(parts) if parts else None

for _, row in df_valid.iterrows():
    lat, lon = float(row["_lat"]), float(row["_lon"])
    c = color_func(float(row[value_col])) if (value_col != "(aucune)") else "#2E86DE"
    popup_html = make_popup(row)
    folium.CircleMarker(
        location=(lat, lon),
        radius=point_radius,
        color=c,
        fill=True,
        fill_color=c,
        fill_opacity=0.85,
        weight=1,
        tooltip=None,
        popup=folium.Popup(popup_html, max_width=350) if popup_html else None
    ).add_to(cluster)

with right:
    st.subheader("Carte")
    st_folium(m, height=720, use_container_width=True)

    # Légende simple
    if value_col != "(aucune)":
        if color_mode == "Continu" and scale is not None:
            st.caption(
                f"Échelle (continu) — **{value_col}** : min = `{scale[0]:.3g}`, max = `{scale[1]:.3g}`"
            )
        elif color_mode == "Classes (seuils)" and scale is not None:
            edges = ", ".join(f"{v:.3g}" for v in scale)
            st.caption(f"Classes (seuils) — **{value_col}** : [{edges}]")
