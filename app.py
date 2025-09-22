import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, MiniMap, Fullscreen
from streamlit_folium import st_folium

st.set_page_config(page_title="Application de prédiction d'inondation", layout="wide")

st.title("🌊 Application de prédiction d'inondation")
st.write("Charge un CSV (stations, niveaux d’eau…) et visualise les points sur une carte.")

# --------- Barre latérale : chargement des données ---------
st.sidebar.header("Données")

src = st.sidebar.radio(
    "Source des données",
    ["Téléverser un CSV", "Depuis une URL (GitHub raw, etc.)"],
    index=0
)

df = None
if src == "Téléverser un CSV":
    up = st.sidebar.file_uploader("Choisir un fichier .csv", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            up.seek(0)
            df = pd.read_csv(up, encoding="latin1")
else:
    url = st.sidebar.text_input(
        "URL CSV (ex: lien raw GitHub)",
        value=""
    )
    if url:
        try:
            df = pd.read_csv(url)
        except Exception:
            df = pd.read_csv(url, encoding="latin1")

if df is None:
    st.info("➡️ Charge un CSV pour commencer. Exemples de colonnes possibles : `Latitude/Longitude` ou `lat_wgs84/long_wgs84`.")
    st.stop()

st.success(f"Données chargées : {df.shape[0]} lignes × {df.shape[1]} colonnes")
with st.expander("Aperçu des données"):
    st.dataframe(df.head(20), use_container_width=True)

# --------- Sélection des colonnes latitude/longitude ---------
st.sidebar.header("Colonnes géographiques")
# on propose des colonnes candidates courantes
candidates_lat = [c for c in df.columns if c.lower() in ("lat","latitude","lat_wgs84","y","y_wgs84")]
candidates_lon = [c for c in df.columns if c.lower() in ("lon","lng","longitude","long_wgs84","x","x_wgs84")]

lat_col = st.sidebar.selectbox("Colonne Latitude", options=df.columns, index=(df.columns.get_loc(candidates_lat[0]) if candidates_lat else 0))
lon_col = st.sidebar.selectbox("Colonne Longitude", options=df.columns, index=(df.columns.get_loc(candidates_lon[0]) if candidates_lon else 1))

# Colonne optionnelle pour la taille/couleur (ex: alt_m_cgvd, niveau d’eau, etc.)
value_col = st.sidebar.selectbox("Colonne valeur (optionnel, pour taille/couleur)", options=["(aucune)"] + list(df.columns), index=0)

# Nettoyage des coordonnées
def _to_num(s):
    try:
        return pd.to_numeric(s)
    except Exception:
        return np.nan

df["_lat"] = _to_num(df[lat_col])
df["_lon"] = _to_num(df[lon_col])
df = df.dropna(subset=["_lat","_lon"])

if df.empty:
    st.error("Aucune ligne avec des coordonnées valides. Vérifie les colonnes sélectionnées.")
    st.stop()

# --------- Filtres simples (optionnels) ----------
with st.expander("🎛️ Filtres (optionnels)"):
    # Filtre sur plage de valeurs si une colonne est choisie
    if value_col != "(aucune)":
        vals = pd.to_numeric(df[value_col], errors="coerce")
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        r = st.slider(f"Filtrer {value_col}", min_value=float(vmin), max_value=float(vmax), value=(float(vmin), float(vmax)))
        df = df[(vals >= r[0]) & (vals <= r[1])]

# --------- Préparation carte ----------
center = [df["_lat"].mean(), df["_lon"].mean()]
m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")
MiniMap(toggle_display=True).add_to(m)
Fullscreen().add_to(m)

# Cluster pour lisibilité
cluster = MarkerCluster(name="Stations / points").add_to(m)

# Fonction couleur/size
def value_to_style(val_series):
    # renvoie (rayon, couleur) arrays
    if value_col == "(aucune)":
        return np.full(len(val_series), 6), np.full(len(val_series), "#3186cc")
    v = pd.to_numeric(val_series, errors="coerce")
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    # normalisation 0-1
    vn = (v - vmin) / (vmax - vmin + 1e-9)
    # rayon 6-16
    radius = 6 + 10 * vn
    # couleur du bleu (#3186cc) -> rouge (#d95f02) approx
    def lerp(a,b,t): 
        return int(a + (b-a)*t)
    cols = []
    for t in vn.fillna(0):
        r = lerp(0x31, 0xd9, t)
        g = lerp(0x86, 0x5f, t)
        b = lerp(0xcc, 0x02, t)
        cols.append(f"#{r:02x}{g:02x}{b:02x}")
    return radius, np.array(cols)

radii, colors = value_to_style(df[value_col] if value_col != "(aucune)" else pd.Series([np.nan]*len(df)))

# Colonnes à montrer dans le popup
popup_cols = [c for c in df.columns if not c.startswith("_")]
if len(popup_cols) > 10:
    popup_cols = popup_cols[:10]  # limiter la taille du popup

for i, row in df.iterrows():
    lat = float(row["_lat"])
    lon = float(row["_lon"])
    radius = float(radii[df.index.get_loc(i)])
    color = colors[df.index.get_loc(i)]

    # petit HTML pour le popup
    infos = "<br>".join([f"<b>{c}:</b> {row[c]}" for c in popup_cols])
    html = folium.Html(infos, script=True)
    popup = folium.Popup(html, max_width=400, min_width=200)

    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=popup
    ).add_to(cluster)

folium.LayerControl(collapsed=False).add_to(m)

st.subheader("🗺️ Carte")
st_data = st_folium(m, height=700, width="100%")

st.caption("Astuce : si tes colonnes s’appellent `lat_wgs84/long_wgs84` ou `Latitude/Longitude`, sélectionne-les dans la barre latérale.")
