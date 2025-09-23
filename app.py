# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

import folium
from folium.plugins import MarkerCluster
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

# ---------------------- PARAMS GÉNÉRAUX ----------------------
st.set_page_config(page_title="Application de prédiction d’inondation",
                   layout="wide")

DATA_DIR = Path("data")

# Tuiles Imagerie (Esri) + étiquettes (CARTO) = "Imagerie Hybride"
ESRI_IMAGERY = {
    "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "attr": "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
}
CARTO_LABELS = {
    "tiles": "https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png",
    "attr": "© OpenStreetMap contributors © CARTO",
    "subdomains": "abcd",
}

# ---------------------- UTILITAIRES ----------------------
@st.cache_data(show_spinner=False)
def list_csvs(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted([p.name for p in data_dir.glob("*.csv")])

@st.cache_data(show_spinner=False)
def read_csv_safely(path: Path) -> pd.DataFrame:
    # On essaie plusieurs encodages courants
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except Exception:
            continue
    # Dernière tentative sans précisions
    return pd.read_csv(path, low_memory=False)

def smart_find_column(cols: list[str], candidates: list[str]) -> str | None:
    lc = [c.casefold() for c in cols]
    for cand in candidates:
        if cand.casefold() in lc:
            return cols[lc.index(cand.casefold())]
    # recherche partielle
    for i, c in enumerate(lc):
        if any(tok in c for tok in candidates):
            return cols[i]
    return None

def numeric_columns(df: pd.DataFrame) -> list[str]:
    nums = []
    for c in df.columns:
        try:
            pd.to_numeric(df[c], errors="coerce")
            # garder si au moins une valeur numérique vraie
            if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
                nums.append(c)
        except Exception:
            pass
    return nums

def make_colormap(values: np.ndarray, palette: list[str]) -> LinearColormap:
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmin == vmax:
        vmax = vmin + 1.0
    return LinearColormap(colors=palette, vmin=vmin, vmax=vmax)

def build_popup(row: pd.Series, max_items: int = 10) -> str:
    # un petit tableau HTML des premières colonnes
    items = row.items()
    html = "<b>Données</b><br><table style='font-size:12px'>"
    k = 0
    for k, (k_, v_) in enumerate(items):
        if k >= max_items:
            html += "<tr><td colspan='2'>…</td></tr>"
            break
        html += f"<tr><td style='padding-right:6px'><b>{k_}</b></td><td>{v_}</td></tr>"
    html += "</table>"
    return html

# ---------------------- INTERFACE ----------------------
st.title("📊 Application de prédiction d’inondation")
left, right = st.columns([0.35, 0.65], vertical_alignment="top")

with left:
    st.subheader("Données")

    csv_files = list_csvs(DATA_DIR)
    if not csv_files:
        st.info("Dépose tes fichiers CSV dans le dossier **data/** du dépôt GitHub, puis relance.")
        st.stop()

    csv_name = st.selectbox("Choisir un CSV dans `data/`", csv_files, index=0)
    df = read_csv_safely(DATA_DIR / csv_name)

    st.caption(f"**{csv_name}** chargé ({len(df):,} lignes, {df.shape[1]} colonnes)")

    # Détection des colonnes lat/lon
    # Ajoute tes variantes si besoin
    lat_guess = smart_find_column(
        list(df.columns),
        ["lat_wgs84", "latitude", "lat", "y", "lat_y", "lat_wgs", "lat_decimal"]
    )
    lon_guess = smart_find_column(
        list(df.columns),
        ["long_wgs84", "longitude", "lon", "x", "long", "lng", "lon_decimal"]
    )

    st.subheader("Colonnes géographiques")
    lat_col = st.selectbox("Colonne Latitude", [*df.columns], index=(list(df.columns).index(lat_guess) if lat_guess else 0))
    lon_col = st.selectbox("Colonne Longitude", [*df.columns], index=(list(df.columns).index(lon_guess) if lon_guess else 0))

    # Colonnes pour style (optionnel)
    st.subheader("Indicateur (couleur / taille)")
    num_cols = [c for c in numeric_columns(df) if c not in (lat_col, lon_col)]
    color_col = st.selectbox("Colonne valeur (pour la couleur) (optionnel)", ["(aucune)"] + num_cols, index=0)
    size_col = st.selectbox("Colonne valeur (pour la taille) (optionnel)", ["(aucune)"] + num_cols, index=0)

    st.markdown("**Mode de coloration**")
    mode = st.radio("Type de coloration", ["Continu", "Classes (seuils)"], horizontal=True, index=0)
    thresholds = None
    if mode == "Classes (seuils)" and color_col != "(aucune)":
        default_thr = ""
        thresholds_str = st.text_input(
            "Seuils (ex: 10, 50, 100) – délimitent les classes",
            value=default_thr,
            placeholder="10, 50, 100"
        )
        if thresholds_str.strip():
            try:
                thresholds = [float(x) for x in thresholds_str.replace(";", ",").split(",")]
                thresholds = sorted(thresholds)
            except Exception:
                st.warning("Impossible de lire les seuils; on repasse en mode Continu.")
                thresholds = None
                mode = "Continu"

    radius_px = st.slider("Taille des points (rayon en pixels, si taille non fournie)", 3, 18, 8)

    st.subheader("Fond de carte")
    basemap = st.selectbox(
        "Fond",
        ["Imagerie Hybride (Esri)", "Topographique (Esri)", "OSM standard"]
    )

with right:
    st.subheader("Carte")

    # Filtrage des lat/lon valides
    df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df_valid = df[df[["_lat", "_lon"]].notna().all(axis=1)].copy()

    if df_valid.empty:
        st.warning("Aucune ligne avec latitude/longitude valides. Vérifie tes colonnes.")
        st.stop()

    # Calcul du centre
    try:
        center = [df_valid["_lat"].astype(float).mean(), df_valid["_lon"].astype(float).mean()]
    except Exception:
        center = [46.8, -71.2]  # Québec approx.

    # Préparer carte Folium
    m = folium.Map(location=center, zoom_start=6, control_scale=True, tiles=None)

    if basemap == "Imagerie Hybride (Esri)":
        folium.TileLayer(tiles=ESRI_IMAGERY["tiles"], attr=ESRI_IMAGERY["attr"], name="Esri.WorldImagery").add_to(m)
        folium.TileLayer(
            tiles=CARTO_LABELS["tiles"],
            attr=CARTO_LABELS["attr"],
            name="Labels",
            subdomains=CARTO_LABELS["subdomains"],
            overlay=True,
            control=True
        ).add_to(m)
    elif basemap == "Topographique (Esri)":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles © Esri",
            name="Esri.WorldTopoMap",
        ).add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)

    # Préparer styles
    color_vals = None
    size_vals = None

    if color_col != "(aucune)" and color_col in df_valid.columns:
        color_vals = pd.to_numeric(df_valid[color_col], errors="coerce").values

    if size_col != "(aucune)" and size_col in df_valid.columns:
        sv = pd.to_numeric(df_valid[size_col], errors="coerce").values
        # Échelle simple (quantiles) -> taille ~ 4..18 px
        q = np.nanpercentile(sv, [5, 95]) if np.isfinite(sv).any() else [0, 1]
        lo, hi = q[0], q[1] if q[1] > q[0] else (q[0] + 1)
        size_vals = np.clip(4 + 14 * (sv - lo) / (hi - lo), 3, 18)

    # Colormap / classes
    palette = ["#2dc937", "#e7b416", "#cc3232"]  # vert -> jaune -> rouge
    class_colors = ["#2dc937", "#99c140", "#e7b416", "#db7b2b", "#cc3232"]  # 5 classes

    if color_vals is not None and np.isfinite(color_vals).any():
        if mode == "Continu" or thresholds is None:
            cmap = make_colormap(color_vals[np.isfinite(color_vals)], palette)
            color_func = lambda v: cmap(v) if np.isfinite(v) else "#3388ff"
            # Ajouter légende
            cmap.caption = color_col
            cmap.add_to(m)
        else:
            bins = [-np.inf, *thresholds, np.inf]
            def class_color(v):
                if not np.isfinite(v):
                    return "#3388ff"
                idx = int(np.digitize([v], bins=bins)[0] - 1)
                idx = max(0, min(idx, len(class_colors)-1))
                return class_colors[idx]
            color_func = class_color
            # Légende "maison"
            html_legend = "<div style='position: fixed; bottom: 30px; left: 10px; z-index: 9999; background: white; padding: 8px; border: 1px solid #ccc; font-size: 12px;'>"
            html_legend += f"<b>{color_col}</b><br>"
            edges = [-np.inf, *thresholds, np.inf]
            for i in range(len(edges)-1):
                a, b = edges[i], edges[i+1]
                label = f"{'-∞' if np.isneginf(a) else a:g} – {'+∞' if np.isposinf(b) else b:g}"
                html_legend += f"<div><span style='display:inline-block;width:12px;height:12px;background:{class_colors[i]};margin-right:6px;'></span>{label}</div>"
            html_legend += "</div>"
            folium.map.CustomPane("legend").add_to(m)
            folium.Marker(
                location=[-90, -180],  # hors écran, mais nécessaire pour injecter la légende
                icon=folium.DivIcon(html=html_legend)
            ).add_to(m)
    else:
        color_func = lambda v: "#3388ff"

    # Ajout des points (cluster)
    cluster = MarkerCluster(name="Stations / points", show=True).add_to(m)

    for _, row in df_valid.iterrows():
        lat, lon = float(row["_lat"]), float(row["_lon"])

        # couleur & rayon
        if color_vals is not None:
            try:
                c = color_func(float(row[color_col]))
            except Exception:
                c = "#3388ff"
        else:
            c = "#3388ff"

        if size_vals is not None:
            try:
                r = float(row[size_col])
                if np.isfinite(r):
                    # r est déjà normalisé 4..18
                    pass
                else:
                    r = radius_px
            except Exception:
                r = radius_px
        else:
            r = radius_px

        folium.CircleMarker(
            location=[lat, lon],
            radius=r if isinstance(r, (int, float)) else radius_px,
            color=c,
            fill=True,
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup(build_popup(row), max_width=350),
            tooltip=str(row.get("nom_mandataire", row.get("id_msp", f"{lat:.4f},{lon:.4f}")))
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)

    st_map = st_folium(m, height=700, use_container_width=True)
