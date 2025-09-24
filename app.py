import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path
import csv

st.set_page_config(page_title="Application de prédiction d’inondation – Carte",
                   layout="wide", page_icon="🌊")

# ------------- PARAMÈTRES DOSSIER -------------
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ------------- UTILITAIRES -------------
@st.cache_data(show_spinner=False)
def list_csvs(data_dir: Path) -> list[str]:
    # XLS/XLSX aussi autorisés
    exts = (".csv", ".CSV", ".xls", ".xlsx", ".XLS", ".XLSX")
    files = sorted([p.name for p in data_dir.iterdir() if p.suffix in exts])
    return files

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    # gère virgules décimales et espaces fines
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
        errors="coerce"
    )

@st.cache_data(show_spinner=False)
def read_table_auto(path: Path, nrows: int, sep_choice: str, enc_choice: str) -> pd.DataFrame:
    # Excel ?
    if path.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(path, nrows=nrows, engine="openpyxl")

    # 1) séparateur depuis l'UI (ou auto)
    if sep_choice == "Virgule (,)":
        sep = ","
    elif sep_choice == "Point-virgule (;)":
        sep = ";"
    elif sep_choice == "Tabulation (\\t)":
        sep = "\t"
    elif sep_choice == "Pipe (|)":
        sep = "|"
    else:
        sep = None  # laisser pandas/sniffer essayer

    # 2) encodages à tester
    encs = [enc_choice] if enc_choice != "Auto" else ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    # 3) première passe : ce que l'utilisateur a demandé
    for enc in encs:
        try:
            return pd.read_csv(
                path, nrows=nrows, sep=sep, engine="python",
                on_bad_lines="skip", encoding=enc, low_memory=False
            )
        except Exception:
            continue

    # 4) seconde passe : tester séparateurs courants
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        for sep_fb in [";", "\t", ",", "|", None]:
            try:
                return pd.read_csv(
                    path, nrows=nrows, sep=sep_fb, engine="python",
                    on_bad_lines="skip", encoding=enc, low_memory=False
                )
            except Exception:
                continue

    # 5) mode agressif : remplace les caractères illisibles, tente auto-inférence
    try:
        return pd.read_csv(
            path, nrows=nrows, sep=None, engine="python",  # permet d’inférer
            on_bad_lines="skip", encoding="utf-8", low_memory=False, quoting=3,  # 3 = QUOTE_NONE
        )
    except Exception:
        pass

    # 6) dernier essai : lecture en texte + split heuristique
    try:
        txt = path.read_text(encoding="utf-8", errors="replace").splitlines()
        # devine le séparateur dominant
        counts = {";": 0, ",": 0, "\t": 0, "|": 0}
        for line in txt[:200]:
            for k in counts: counts[k] += line.count(k)
        guess = max(counts, key=counts.get) if max(counts.values()) > 0 else ","
        from io import StringIO
        return pd.read_csv(StringIO("\n".join(txt)), nrows=nrows, sep=guess,
                           engine="python", on_bad_lines="skip")
    except Exception as e:
        raise ValueError("Impossible de lire le fichier. Vérifiez séparateur/encodage ou re-exportez en CSV.") from e


def classify_values(s: pd.Series, mode: str, nb: int = 5):
    """Retourne couleurs/valeurs normalisées pour affichage."""
    s = pd.to_numeric(s, errors="coerce")
    v = s.to_numpy(dtype=float)
    mask = np.isfinite(v)
    colors = np.array(["#7cb342"] * len(v))  # défaut
    sizes = np.where(mask, 6 + 14 * (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-9), 6)

    if mode == "Continu":
        # dégradé jaune -> rouge
        # couleur calculée à partir d'une palette simple
        # 0 => #f1c40f, 1 => #e74c3c
        def lerp_color(t):
            c1 = np.array([241, 196, 15])   # f1c40f
            c2 = np.array([231, 76, 60])    # e74c3c
            c = (1 - t) * c1 + t * c2
            return '#%02x%02x%02x' % tuple(c.astype(int))

        t = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-9)
        t[~mask] = 0.0
        colors = np.array([lerp_color(x) for x in t])
        return colors, sizes, None

    # Classes (quantiles)
    try:
        q = np.nanquantile(v, np.linspace(0, 1, nb + 1))
    except Exception:
        q = np.linspace(np.nanmin(v), np.nanmax(v), nb + 1)

    # palette 5 classes
    pal = ["#66bb6a", "#ffee58", "#ffb74d", "#ff7043", "#e53935"]
    bins = np.digitize(v, q[1:-1], right=False)
    colors = np.array([pal[min(b, nb-1)] if m else "#bdbdbd" for b, m in zip(bins, mask)])
    return colors, sizes, q


# ------------- SIDEBAR -------------
with st.sidebar:
    st.header("Données")
    csv_files = list_csvs(DATA_DIR)
    if not csv_files:
        st.info("Place des fichiers dans le dossier **data/** du dépôt (CSV/XLS/XLSX).")
        st.stop()

    csv_name = st.selectbox("Choisissez un CSV (dans data/)", csv_files, index=0)
    nrows = st.slider("Nombre max. de lignes à charger", 500, 50000, 5000, step=500)

    st.subheader("Options d'import")
    sep_choice = st.selectbox("Séparateur", ["Auto", "Virgule (,)", "Point-virgule (;)", "Tabulation (\\t)", "Pipe (|)"])
    enc_choice = st.selectbox("Encodage", ["Auto", "utf-8", "utf-8-sig", "latin1", "cp1252"])

# ------------- LECTURE -------------
df = read_table_auto(DATA_DIR / csv_name, nrows, sep_choice, enc_choice)
st.caption(f"**{csv_name}** chargé ({len(df)} lignes, {df.shape[1]} colonnes)")
if len(df) == 0:
    st.warning("Aucune ligne lisible. Essayez un autre séparateur/encodage.")
    st.stop()

# ------------- MAPPING UI -------------
st.title("Application de prédiction d’inondation – Carte")
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Colonnes géographiques")
    col_names = list(df.columns)
    lat_col = st.selectbox("Colonne Latitude", ["<auto>"] + col_names)
    lon_col = st.selectbox("Colonne Longitude", ["<auto>"] + col_names)

    # Auto-détection simple si <auto>
    if lat_col == "<auto>":
        for c in col_names:
            if c.strip().lower().startswith(("lat_", "latitude", "lat")):
                lat_col = c; break
    if lon_col == "<auto>":
        for c in col_names:
            if c.strip().lower().startswith(("lon", "lng", "long_", "longitude")):
                lon_col = c; break

    if lat_col not in df.columns or lon_col not in df.columns:
        st.warning("Sélectionne les colonnes lat/lon.")
        st.stop()

    # Conversion robuste
    df = df.copy()
    df[lat_col] = _coerce_numeric_series(df[lat_col])
    df[lon_col] = _coerce_numeric_series(df[lon_col])
    df_valid = df[np.isfinite(df[lat_col]) & np.isfinite(df[lon_col])]
    dropped = len(df) - len(df_valid)
    if dropped > 0:
        st.caption(f"{dropped} lignes ignorées (lat/lon invalides).")

    st.subheader("Indicateur (couleur / taille)")
    value_cols = ["(aucun)"] + [c for c in col_names if c not in (lat_col, lon_col)]
    val_col = st.selectbox("Colonne valeur (pour la couleur/taille)", value_cols)
    mode = st.radio("Type de coloration", ["Continu", "Classes (quantiles)"], horizontal=True)
    nb_classes = st.slider("Nombre de classes", 3, 7, 5, disabled=(mode != "Classes (quantiles)"))
    radius_base = st.slider("Taille de base des points (rayon)", 4, 16, 8)

    st.subheader("Popup")
    popup_cols = st.multiselect("Colonnes à afficher au clic", col_names[:20], default=col_names[:3])

with right:
    st.subheader("Carte")

    # centre carte
    if len(df_valid):
        center = [df_valid[lat_col].astype(float).mean(), df_valid[lon_col].astype(float).mean()]
    else:
        center = [46.8, -71.2]  # Québec (fallback)

    # Fond Imagerie Hybride (Esri World Imagery + labels)
    m = folium.Map(location=center, zoom_start=6, tiles=None, control_scale=True)
    folium.raster_layers.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Imagerie (Esri)"
    ).add_to(m)
    folium.raster_layers.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap (CC-BY-SA)",
        name="Topographie"
    ).add_to(m)
    folium.raster_layers.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap",
        name="OSM"
    ).add_to(m)
    # Labels (stamen-toner-Lite ou Esri labels si dispo)
    folium.raster_layers.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}.png",
        attr="Stamen Toner Lite",
        name="Labels"
    ).add_to(m)

    # Couleurs / tailles
    if val_col != "(aucun)" and val_col in df_valid.columns:
        colors, sizes, breaks = classify_values(df_valid[val_col], "Continu" if mode == "Continu" else "Classes", nb_classes)
    else:
        colors = np.array(["#1976d2"] * len(df_valid))
        sizes = np.array([radius_base] * len(df_valid))
        breaks = None

    cluster = MarkerCluster(name="Stations / points", disableClusteringAtZoom=12)
    cluster.add_to(m)

    for i, row in df_valid.iterrows():
        lat, lon = float(row[lat_col]), float(row[lon_col])
        color = colors[df_valid.index.get_loc(i)]
        r = float(sizes[df_valid.index.get_loc(i)])

        if popup_cols:
            items = [f"<b>{c}</b>: {row.get(c, '')}" for c in popup_cols]
            popup_html = "<br>".join(items)
        else:
            popup_html = ""

        folium.CircleMarker(
            location=[lat, lon],
            radius=max(3.0, r if val_col != "(aucun)" else radius_base),
            color=color, fill=True, fill_color=color, fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=400)
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, height=650, width=None, returned_objects=[])

st.caption("Fond de carte : Esri World Imagery (Imagerie), OpenStreetMap, Stamen.  |  App Streamlit.")
