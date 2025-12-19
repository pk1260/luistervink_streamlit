# main.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from pathlib import Path
import math
import re

st.set_page_config(page_title="Luistervink Dashboard", layout="wide")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    # Parse timestamp (UTC), drop mislukte parses
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # Zet om naar NL tijd
    df["timestamp_nl"] = df["timestamp"].dt.tz_convert("Europe/Amsterdam")

    # Maak date/hour robuust
    df["date"] = df["timestamp_nl"].dt.date
    df["hour"] = df["timestamp_nl"].dt.hour

    return df


def pick_species_col(df: pd.DataFrame) -> str:
    if "species_common_name" in df.columns:
        return "species_common_name"
    if "species.common_name" in df.columns:
        return "species.common_name"
    if "species.commonName" in df.columns:
        return "species.commonName"
    for c in df.columns:
        if "species" in c and "common" in c:
            return c
    return "species_common_name"


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    # afstand in meters
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parse_point_wkt(wkt: str):
    """
    Verwacht 'POINT(lon lat)' of 'POINT(lat lon)' afhankelijk van kolomnaam.
    Geeft tuple (a, b) terug als floats, anders (nan, nan).
    """
    if not isinstance(wkt, str):
        return (np.nan, np.nan)
    m = re.search(r"POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)", wkt.strip(), re.IGNORECASE)
    if not m:
        return (np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)))


@st.cache_data
def load_broedvogels(path: Path) -> pd.DataFrame:
    # Probeer ; en , (zonder user input)
    try:
        bv = pd.read_csv(path, sep=";")
    except Exception:
        bv = pd.read_csv(path, sep=",")

    # Lowercase lookup
    cols = {c.lower(): c for c in bv.columns}

    # 1) direct LAT/LNG of latitude/longitude
    if "lat" in cols and "lng" in cols:
        bv = bv.rename(columns={cols["lat"]: "LAT", cols["lng"]: "LNG"})
    elif "latitude" in cols and "longitude" in cols:
        bv = bv.rename(columns={cols["latitude"]: "LAT", cols["longitude"]: "LNG"})

    # 2) WKT varianten
    if "lat" not in (c.lower() for c in bv.columns) and "LAT" not in bv.columns:
        if "wkt_lng_lat" in cols:
            # POINT(lon lat)
            a = bv[cols["wkt_lng_lat"]].apply(parse_point_wkt)
            bv["LNG"] = a.apply(lambda t: t[0])
            bv["LAT"] = a.apply(lambda t: t[1])
        elif "wkt_lat_lng" in cols:
            # POINT(lat lon)
            a = bv[cols["wkt_lat_lng"]].apply(parse_point_wkt)
            bv["LAT"] = a.apply(lambda t: t[0])
            bv["LNG"] = a.apply(lambda t: t[1])

    # Force numeric + drop missing
    if "LAT" in bv.columns and "LNG" in bv.columns:
        bv["LAT"] = pd.to_numeric(bv["LAT"], errors="coerce")
        bv["LNG"] = pd.to_numeric(bv["LNG"], errors="coerce")
        bv = bv.dropna(subset=["LAT", "LNG"]).copy()
    else:
        # Geen coordinaten gevonden
        bv = bv.iloc[0:0].copy()

    return bv


st.title("🐦 Luistervink – Vogel detecties")

# --- CSV PAD ---
csv_path = Path(__file__).parent / "luistervink_GEOLOCATED.csv"
if not csv_path.exists():
    st.error(
        f"Bestand niet gevonden:\n{csv_path}\n\n"
        "Zet luistervink_GEOLOCATED.csv in dezelfde map als main.py "
        "of pas het pad aan."
    )
    st.stop()

df_all = load_data(csv_path)
species_col = pick_species_col(df_all)

# --- Sidebar filters (werken voor beide kaarten & grafieken) ---
st.sidebar.header("Filters")

eligible_only = st.sidebar.checkbox("Alleen eligible=True", value=True)
min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.80, 0.01)

device_options = sorted(df_all["device_name"].dropna().unique()) if "device_name" in df_all.columns else []
selected_devices = st.sidebar.multiselect(
    "Device(s)", device_options, default=device_options[:1] if device_options else []
)

species_options = sorted(df_all[species_col].dropna().unique()) if species_col in df_all.columns else []
selected_species = st.sidebar.multiselect(
    "Soort(en)", species_options, default=species_options[:5] if len(species_options) >= 5 else species_options
)

# Date range op basis van df_all
min_date_all = df_all["date"].min()
max_date_all = df_all["date"].max()
date_range = st.sidebar.date_input("Periode", (min_date_all, max_date_all))

# Extra: simpele “buurt/locatie” zoekbalk
loc_query = st.sidebar.text_input("Zoek locatie (wijk/park/buurt)", value="").strip()

# --- Pas filters toe ---
df = df_all.copy()

if eligible_only and "eligible" in df.columns:
    df = df[df["eligible"] == True]

if "confidence" in df.columns:
    df = df[df["confidence"] >= min_conf]

if selected_devices and "device_name" in df.columns:
    df = df[df["device_name"].isin(selected_devices)]

if selected_species and species_col in df.columns:
    df = df[df[species_col].isin(selected_species)]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    df = df[(df["date"] >= start) & (df["date"] <= end)]

if loc_query and "locatie_naam" in df.columns:
    df = df[df["locatie_naam"].astype(str).str.contains(loc_query, case=False, na=False)]

st.sidebar.caption(f"Rijen na filters: {len(df):,}")

if df.empty:
    st.warning("Geen data over na filters. Verlaag filters (bv. confidence) of kies een andere periode.")
    st.stop()

# --- Tabs: buurtkaart / NL-overzicht / grafieken / overlap ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Buurtkaart (detecties)",
    "🇳🇱 Nederland-overzicht (kasten)",
    "📊 Grafieken & samenvatting",
    "🧪 Overlap broedvogels vs detecties"
])


with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🗺️ Kaart – detecties (gefilterd)")

        center_lat = float(df["latitude"].dropna().mean())
        center_lon = float(df["longitude"].dropna().mean())

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True, tiles="OpenStreetMap")
        mc = MarkerCluster(name="Detecties").add_to(m)

        max_points = st.slider("Max punten op kaart (performance)", 500, 10000, 3000, 500)
        df_points = df.dropna(subset=["latitude", "longitude"]).head(max_points)

        for _, r in df_points.iterrows():
            popup = f"""
            <b>Soort:</b> {safe_str(r.get(species_col,''))}<br>
            <b>Confidence:</b> {safe_str(r.get('confidence',''))}<br>
            <b>Tijd (NL):</b> {safe_str(r.get('timestamp_nl',''))}<br>
            <b>Locatie:</b> {safe_str(r.get('locatie_naam',''))}<br>
            <b>Kast:</b> {safe_str(r.get('device_name',''))}
            """
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=5,
                popup=folium.Popup(popup, max_width=320),
            ).add_to(mc)

        if st.checkbox("Toon heatmap (gefilterde detecties)", value=False):
            heat_data = df_points[["latitude", "longitude"]].dropna().values.tolist()
            HeatMap(heat_data, radius=20, name="Heatmap").add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, use_container_width=True, height=540)

    with col2:
        st.subheader("📌 Samenvatting (gefilterd)")
        st.metric("Detecties", f"{len(df):,}")
        st.metric("Unieke soorten", f"{df[species_col].nunique():,}" if species_col in df.columns else "—")
        st.metric("Devices", f"{df['device_name'].nunique():,}" if "device_name" in df.columns else "—")
        st.metric("Locaties", f"{df['locatie_naam'].nunique():,}" if "locatie_naam" in df.columns else "—")

        st.write("**Top 10 soorten (gefilterd)**")
        if species_col in df.columns:
            st.dataframe(df[species_col].value_counts().head(10))
        else:
            st.info("Geen species kolom gevonden.")


with tab2:
    st.subheader("🇳🇱 Overzichtskaart Nederland – alle kasten (devices)")

    base = df_all.dropna(subset=["latitude", "longitude"]).copy()

    if "device_id" in base.columns:
        devices_df = base.sort_values("timestamp_nl").groupby("device_id", as_index=False).tail(1)
        device_key = "device_id"
    elif "device_name" in base.columns:
        devices_df = base.sort_values("timestamp_nl").groupby("device_name", as_index=False).tail(1)
        device_key = "device_name"
    else:
        devices_df = base.copy()
        device_key = None

    nl_center = [52.1326, 5.2913]
    nl_map = folium.Map(location=nl_center, zoom_start=7, control_scale=True, tiles="OpenStreetMap")

    kast_cluster = MarkerCluster(name="Kasten (devices)").add_to(nl_map)

    if device_key and species_col in base.columns:
        top_by_device = (
            base.groupby([device_key, species_col]).size()
            .reset_index(name="n")
            .sort_values(["n"], ascending=False)
        )
        top_species_map = {}
        for _, row in top_by_device.iterrows():
            k = row[device_key]
            if k not in top_species_map:
                top_species_map[k] = (row[species_col], int(row["n"]))
    else:
        top_species_map = {}

    for _, r in devices_df.iterrows():
        key_val = r.get(device_key, None) if device_key else None
        top_txt = ""
        if key_val in top_species_map:
            sp, n = top_species_map[key_val]
            top_txt = f"<b>Meest gehoord:</b> {safe_str(sp)} ({n} detecties)<br>"

        popup = f"""
        <b>Kast:</b> {safe_str(r.get('device_name','Luistervink'))}<br>
        <b>Locatie:</b> {safe_str(r.get('locatie_naam',''))}<br>
        <b>Laatste detectie (NL):</b> {safe_str(r.get('timestamp_nl',''))}<br>
        {top_txt}
        """

        folium.Marker(
            location=[r["latitude"], r["longitude"]],
            popup=folium.Popup(popup, max_width=360),
            tooltip=f"{safe_str(r.get('device_name','Luistervink'))} – {safe_str(r.get('locatie_naam',''))}",
        ).add_to(kast_cluster)

    show_heat_nl = st.checkbox("Toon heatmap (alle detecties)", value=False)
    if show_heat_nl:
        sample_size = min(20000, len(base))
        base_sample = base.sample(sample_size, random_state=42) if len(base) > sample_size else base
        heat_data = base_sample[["latitude", "longitude"]].values.tolist()
        HeatMap(heat_data, radius=18, name="Detectie heatmap").add_to(nl_map)

    folium.LayerControl().add_to(nl_map)

    bounds = devices_df[["latitude", "longitude"]].dropna().values.tolist()
    if len(bounds) >= 2:
        nl_map.fit_bounds(bounds)

    st.caption(
        "Deze kaart toont **alle kasten** als overzicht. Handig voor bewoners: waar hangt er een Luistervink in de buurt?"
    )
    st_folium(nl_map, use_container_width=True, height=560)


with tab3:
    st.subheader("📊 Grafieken (gefilterd)")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.write("Detecties per uur (NL)")
        per_hour = df.groupby("hour").size().reindex(range(24), fill_value=0)
        fig, ax = plt.subplots()
        ax.plot(per_hour.index, per_hour.values)
        ax.set_xlabel("Uur")
        ax.set_ylabel("Aantal detecties")
        st.pyplot(fig)

    with g2:
        st.write("Detecties per dag")
        per_day = df.groupby("date").size()
        fig, ax = plt.subplots()
        ax.plot(per_day.index, per_day.values)
        ax.set_xlabel("Dag")
        ax.set_ylabel("Aantal detecties")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with g3:
        st.write("Confidence verdeling")
        fig, ax = plt.subplots()
        ax.hist(df["confidence"].dropna(), bins=20)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Aantal")
        st.pyplot(fig)

    st.subheader("🏆 Top soorten & locaties (gefilterd)")
    c1, c2 = st.columns(2)

    with c1:
        st.write("Top 15 soorten")
        if species_col in df.columns:
            st.dataframe(df[species_col].value_counts().head(15))
        else:
            st.info("Geen species kolom gevonden.")

    with c2:
        st.write("Top 15 locaties")
        if "locatie_naam" in df.columns:
            st.dataframe(df["locatie_naam"].value_counts().head(15))
        else:
            st.info("Geen locatie_naam kolom gevonden.")


with tab4:
    st.subheader("🧪 Onderzoek: overlap broedvogel-locaties vs Luistervink-detecties")

    broed_path = Path(__file__).parent / "VOGELS.csv"
    if not broed_path.exists():
        st.error(
            f"Broedvogelbestand niet gevonden:\n{broed_path}\n\n"
            "Zet broedvogels.csv in dezelfde map als main.py."
        )
        st.stop()

    bv = load_broedvogels(broed_path)
    if bv.empty:
        st.error(
            "Ik kon geen coordinaten vinden in broedvogels.csv.\n\n"
            "Ondersteund: LAT/LNG, latitude/longitude, of WKT_LNG_LAT / WKT_LAT_LNG."
        )
        st.stop()

    radius_m = st.slider("Match-radius (meter)", 25, 500, 150, 25)
    max_det = st.slider("Max detecties in analyse (performance)", 1000, 50000, 10000, 1000)

    det = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(det) > max_det:
        det = det.sample(max_det, random_state=42)

    # Performance guard
    if len(bv) > 20000:
        st.warning(
            "Broedvogelpunten zijn heel veel (>20k). Dan wordt afstand-berekening traag met deze methode.\n"
            "Als je zegt hoeveel punten jullie hebben, maak ik een snellere grid/k-d-tree variant."
        )
        st.stop()

    points_lat = bv["LAT"].to_numpy()
    points_lon = bv["LNG"].to_numpy()

    # Nearest distance per detectie (brute force, maar prima tot enkele duizenden * punten)
    def nearest_distance(lat, lon):
        dmin = None
        for la, lo in zip(points_lat, points_lon):
            d = haversine_m(lat, lon, float(la), float(lo))
            if (dmin is None) or (d < dmin):
                dmin = d
        return dmin if dmin is not None else np.nan

    det["dist_to_broed_m"] = det.apply(lambda r: nearest_distance(r["latitude"], r["longitude"]), axis=1)
    det["overlap"] = det["dist_to_broed_m"] <= radius_m

    overlap_n = int(det["overlap"].sum())
    total_n = int(len(det))

    c1, c2, c3 = st.columns(3)
    c1.metric("Detecties geanalyseerd", f"{total_n:,}")
    c2.metric(f"Overlaps (≤ {radius_m} m)", f"{overlap_n:,}")
    c3.metric("Overlap-percentage", f"{(overlap_n / total_n * 100):.1f}%")

    st.markdown("### 🐦 Soorten: binnen vs buiten broedlocaties")
    left, right = st.columns(2)
    with left:
        st.write("**Top 15 soorten – binnen radius**")
        st.dataframe(det[det["overlap"]][species_col].value_counts().head(15))
    with right:
        st.write("**Top 15 soorten – buiten radius**")
        st.dataframe(det[~det["overlap"]][species_col].value_counts().head(15))

    st.markdown("### 🗺️ Kaart: broedpunten + overlappende detecties")
    center_lat = float(det["latitude"].mean())
    center_lon = float(det["longitude"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True, tiles="OpenStreetMap")

    broed_layer = folium.FeatureGroup(name="Broedvogelpunten").add_to(m)
    for _, r in bv.iterrows():
        label = safe_str(r.get("Vogel", r.get("species_common_name", "")))
        folium.CircleMarker(
            location=[r["LAT"], r["LNG"]],
            radius=4,
            popup=folium.Popup(f"Broedpunt<br>{label}", max_width=260),
        ).add_to(broed_layer)

    overlap_layer = folium.FeatureGroup(name=f"Detecties binnen {radius_m}m").add_to(m)
    det_overlap = det[det["overlap"]]

    # ook hier een max om kaart snel te houden
    max_map_pts = st.slider("Max overlap-punten op kaart (performance)", 200, 10000, 2000, 200)
    det_overlap_map = det_overlap.head(max_map_pts)

    for _, r in det_overlap_map.iterrows():
        popup = f"""
        <b>Soort:</b> {safe_str(r.get(species_col,''))}<br>
        <b>Afstand:</b> {safe_str(round(r.get('dist_to_broed_m', np.nan), 1))} m<br>
        <b>Tijd (NL):</b> {safe_str(r.get('timestamp_nl',''))}<br>
        <b>Locatie:</b> {safe_str(r.get('locatie_naam',''))}<br>
        <b>Kast:</b> {safe_str(r.get('device_name',''))}
        """
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=5,
            popup=folium.Popup(popup, max_width=340),
        ).add_to(overlap_layer)

    if st.checkbox("Toon heatmap van overlap-detecties", value=False):
        heat_data = det_overlap_map[["latitude", "longitude"]].values.tolist()
        HeatMap(heat_data, radius=20, name="Overlap heatmap").add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, height=560)

    st.markdown("### 📌 Interpretatie")
    st.write(
        "Dit tabblad laat **ruimtelijke samenhang** zien: detecties die dichtbij bekende broedlocaties vallen. "
        "Overlap is een **indicatie**, geen bewijs. Maak het sterker door te filteren op broedseizoen (maart–juli), "
        "of door overlap te vergelijken tussen locaties en periodes."
    )
