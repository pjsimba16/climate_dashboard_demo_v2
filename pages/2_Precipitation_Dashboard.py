# pages/2_Precipitation_Dashboard.py
import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import streamlit as st
from huggingface_hub import hf_hub_download

try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

try:
    fragment = st.fragment
    _FRAGMENT_SUPPORTED = True
except Exception:
    _FRAGMENT_SUPPORTED = False

st.set_page_config(page_title="Precipitation Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ====== SWITCH THIS WHEN READY ======
GEOJSON_SOURCE = "Hugging Face"   # or "Local"
# ====================================

INDICATOR_LABELS = [
    "Temperature", "Temperature Thresholds", "Heatwaves", "Coldwaves",
    "Precipitation", "Dry Conditions", "Wet Conditions", "Humidity", "Windspeeds"
]
ELEVATION_CANDIDATES = ["elevation", "Elevation", "elev", "elev_m", "elevation_m", "altitude", "Altitude"]
CBLIND = {
    "blue":"#0072B2","orange":"#E69F00","sky":"#56B4E9","green":"#009E73",
    "yellow":"#F0E442","navy":"#332288","verm":"#D55E00","pink":"#CC79A7","grey":"#999999","red":"#d62728"
}

# ---------- Color/luminance helpers ----------
from plotly.colors import sample_colorscale

_RGB_RE  = re.compile(r"rgba?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)(?:\s*,\s*(\d+(?:\.\d+)?))?\s*\)", re.I)
def _color_to_rgb01(color_str: str):
    if not isinstance(color_str, str): raise ValueError("Color must be a string")
    s = color_str.strip()
    m = _RGB_RE.fullmatch(s)
    if m: return float(m.group(1))/255.0, float(m.group(2))/255.0, float(m.group(3))/255.0
    if s.startswith("#"):
        s = s.lstrip("#"); s = "".join(ch*2 for ch in s) if len(s)==3 else s
        if len(s)!=6: raise ValueError("Invalid hex length")
        return int(s[0:2],16)/255.0, int(s[2:4],16)/255.0, int(s[4:6],16)/255.0
    raise ValueError(f"Unsupported color format: {color_str}")
def _perceived_luminance(rgb01): r,g,b = rgb01; return 0.299*r + 0.587*g + 0.114*b
def pick_label_colors(names, value_map, colorscale_name: str, vmin: float, vmax: float, default_dark_on_missing=True):
    cs = getattr(px.colors.sequential, colorscale_name, px.colors.sequential.YlOrRd) if isinstance(colorscale_name, str) else colorscale_name
    denom = (vmax - vmin) if vmax != vmin else 1.0
    out = []
    for nm in names:
        val = value_map.get(nm, None)
        if val is None or not np.isfinite(val): out.append("black" if default_dark_on_missing else "white"); continue
        t = float(np.clip((val - vmin) / denom, 0.0, 1.0))
        try:
            col = sample_colorscale(cs, [t])[0]
            lum = _perceived_luminance(_color_to_rgb01(col))
            out.append("black" if lum > 0.55 else "white")
        except Exception: out.append("black")
    return out

# -------------------- Secrets / config helpers --------------------
def _secret_or_env(key: str, default: str = "") -> str:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)
HF_REPO_ID   = _secret_or_env("HF_REPO_ID",   "pjsimba16/adb_climate_dashboard_v1")
HF_REPO_TYPE = _secret_or_env("HF_REPO_TYPE", "space")
def _get_hf_token():
    tok = _secret_or_env("HF_TOKEN", ""); return tok or None

# -------------------- Data loaders --------------------
@st.cache_data(ttl=24*3600, show_spinner=False)
def _dl(filename: str) -> str:
    return hf_hub_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=filename, token=_get_hf_token())
@st.cache_data(ttl=24*3600, show_spinner=False)
def read_parquet_from_hf(filename: str) -> pd.DataFrame:
    return pd.read_parquet(_dl(filename))
@st.cache_data(ttl=7*24*3600, show_spinner=False)
def load_country_adm1_geojson(iso3: str, source: str):
    if source == "Hugging Face":
        path = _dl(f"ADM1_geodata/{iso3}.geojson")
    else:
        path = os.path.join("ADM1_geodata", f"{iso3}.geojson")
        if not os.path.isfile(path): raise FileNotFoundError(f"Local GeoJSON not found at {path}")
    gdf = gpd.read_file(path)
    try: gdf["geometry"] = gdf["geometry"].buffer(0)
    except Exception: pass
    gdf = gdf.to_crs(4326)
    bounds = tuple(gdf.total_bounds); n_features = len(gdf)
    return gdf.__geo_interface__, bounds, n_features
@st.cache_data(ttl=7*24*3600, show_spinner=False)
def adm1_label_points(iso3: str, source: str):
    if source == "Hugging Face":
        path = _dl(f"ADM1_geodata/{iso3}.geojson")
    else:
        path = os.path.join("ADM1_geodata", f"{iso3}.geojson")
        if not os.path.isfile(path): raise FileNotFoundError(f"Local GeoJSON not found at {path}")
    gdf = gpd.read_file(path)
    try: gdf["geometry"] = gdf["geometry"].buffer(0)
    except Exception: pass
    gdf = gdf.to_crs(4326)
    name_col = "shapeName" if "shapeName" in gdf.columns else ("NAME_1" if "NAME_1" in gdf.columns else gdf.columns[0])
    pts = gdf.representative_point()
    return pts.x.to_numpy(), pts.y.to_numpy(), gdf[name_col].astype(str).to_numpy()

def _elevation_completeness(iso3: str, geojson_dict, city_map: pd.DataFrame, elev_candidates) -> tuple:
    """(elev_col, is_complete, available_count, total_adm1) vs ALL ADM1 shapes."""
    try:
        features = geojson_dict.get("features", [])
        adm1_names = [f["properties"].get("shapeName") for f in features if f and "properties" in f]
    except Exception:
        adm1_names = []
    cm_iso = city_map[city_map["Country"].astype(str).str.upper() == iso3].copy()
    elev_col = next((c for c in elev_candidates if c in cm_iso.columns), None)
    if not adm1_names or elev_col is None:
        return (elev_col, False, 0, len(adm1_names))
    chk = pd.DataFrame({"City": adm1_names})
    tmp = cm_iso[["City", elev_col]].copy()
    tmp[elev_col] = pd.to_numeric(tmp[elev_col], errors="coerce")
    chk = chk.merge(tmp, on="City", how="left")
    avail = int(chk[elev_col].notna().sum()); total = len(chk)
    return (elev_col, avail == total and total > 0, avail, total)

# -------------------- Datasets --------------------
with st.spinner("Loading base datasets…"):
    CITY_PRECIP    = read_parquet_from_hf("city_precipitation.snappy.parquet")
    COUNTRY_PRECIP = read_parquet_from_hf("country_precipitation.snappy.parquet")
    #CITY_MAP       = read_parquet_from_hf("city_mapper_with_coords_v2.snappy.parquet")
    CITY_MAP       = pd.read_csv('city_mapper_with_coords_v3.csv')

# -------------------- Display helpers (names) --------------------
try:
    import pycountry
except Exception:
    pycountry = None
_CUSTOM_COUNTRY_DISPLAY = {"CHN": "People's Republic of China","TWN": "Taipei,China","HKG": "Hong Kong, China"}
def iso3_to_name(iso: str) -> str:
    iso = (iso or "").upper().strip()
    if pycountry:
        try:
            c = pycountry.countries.get(alpha_3=iso)
            if c and getattr(c, "name", None): return c.name
        except Exception: pass
    return iso
def display_country_name(iso: str) -> str:
    iso = (iso or "").upper().strip()
    return _CUSTOM_COUNTRY_DISPLAY.get(iso, iso3_to_name(iso))

# ------------------ Read URL query params ------------------
qp = st.query_params
iso3_q  = (qp.get("iso3") or st.session_state.get("nav_iso3") or "").upper()
city_q  = qp.get("city", "")
start_q = qp.get("start", ""); end_q   = qp.get("end", "")

# ---------------------- Header ----------------------
top_l, top_r = st.columns([0.12, 0.88])
with top_l:
    if st.button("← Home"):
        keep_iso3 = st.query_params.get("iso3", ""); keep_city = st.query_params.get("city", "")
        st.query_params.clear()
        if keep_iso3: st.query_params.update({"iso3": keep_iso3})
        if keep_city: st.query_params.update({"city": keep_city})
        try: st.switch_page("Home_Page.py")
        except Exception: st.rerun()
st.markdown(f"### Precipitation - {display_country_name(iso3_q) if iso3_q else '…'}")

# --------------- Country select ----------
countries_iso = sorted(COUNTRY_PRECIP["Country"].dropna().astype(str).str.upper().unique().tolist())
country_options = ["—"] + countries_iso
if iso3_q in country_options: st.session_state["opt_iso3_p"] = iso3_q
elif "opt_iso3_p" not in st.session_state: st.session_state["opt_iso3_p"] = "—"

# -------------------------- Layout -------------------------------
lc, rc = st.columns([0.34, 0.66], gap="large")

with lc:
    iso3 = st.selectbox(
        "Select Country",
        options=country_options,
        index=country_options.index(st.session_state["opt_iso3_p"]) if st.session_state["opt_iso3_p"] in country_options else 0,
        key="opt_iso3_p",
        format_func=lambda v: ("Select…" if v=="—" else display_country_name(v)),
        help="Pick a country, or click one on the Home map to arrive here pre-selected."
    )
    desired_iso = "" if iso3 == "—" else iso3
    if desired_iso != st.query_params.get("iso3", ""):
        st.query_params.update({"iso3": desired_iso, "city": ""}); st.rerun()

    MAP_HEIGHT = 640

    if iso3 and iso3 != "—":
        df_iso = CITY_PRECIP[CITY_PRECIP["Country"].astype(str).str.upper() == iso3].copy()
        if not df_iso.empty:
            df_iso["Date"] = pd.to_datetime(df_iso["Date"], errors="coerce")
            df_iso = df_iso.dropna(subset=["Date"])
            df_iso["City"] = df_iso["City"].astype(str)

            try:
                grp = df_iso.sort_values("Date").groupby("City", observed=True)["Date"]
                idx = grp.idxmax()
                df_map = df_iso.loc[idx, ["City", "Precipitation (Sum)", "Date"]].copy()
            except Exception:
                df_map = (df_iso.sort_values(["City", "Date"])
                          .drop_duplicates(subset=["City"], keep="last")[["City", "Precipitation (Sum)", "Date"]].copy())

            if not df_map.empty:
                with st.spinner("Loading map & drawing choropleth…"):
                    try:
                        geojson_dict, bounds, n_features = load_country_adm1_geojson(iso3, GEOJSON_SOURCE)
                        minx, miny, maxx, maxy = bounds

                        # ---- Elevation completeness vs ALL ADM1s ----
                        elev_col, elev_complete, _avail, _total = _elevation_completeness(iso3, geojson_dict, CITY_MAP, ELEVATION_CANDIDATES)

                        if elev_col is None:
                            df_map["__elev"] = np.nan
                            elev_use_col = "__elev"
                        else:
                            cm_iso = CITY_MAP[CITY_MAP["Country"].astype(str).str.upper() == iso3][["City", elev_col]].copy()
                            cm_iso[elev_col] = pd.to_numeric(cm_iso[elev_col], errors="coerce")
                            df_map = df_map.merge(cm_iso, on="City", how="left")
                            elev_use_col = elev_col

                        # --- TOP-LEFT: Map data selector (default per country) ---
                        if st.session_state.get("last_iso3_for_choice_p") != iso3 or "map_data_choice_p" not in st.session_state:
                            st.session_state["map_data_choice_p"] = "Elevation" if elev_complete else "Precipitation"
                            st.session_state["last_iso3_for_choice_p"] = iso3

                        choice = st.radio(
                            "Map data",
                            ["Precipitation", "Elevation"],
                            horizontal=True,
                            key="map_data_choice_p",
                            help="If Elevation is selected but missing for some regions, those ADM1s appear white."
                        )

                        show_labels_current = st.session_state.get(f"show_labels_p_{iso3}", True)

                        if choice == "Elevation":
                            map_color_col = elev_use_col
                            colorbar_title = "Elevation (m)"
                            colorscale = "Viridis"
                            hover_tmpl = (
                                "<b>%{customdata[0]}</b><br>"
                                "Elevation: %{customdata[1]:.0f} m<br>"
                                "As of: %{customdata[2]}<extra></extra>"
                            )
                        else:
                            map_color_col = "Precipitation (Sum)"
                            colorbar_title = "Precipitation (mm)"
                            colorscale = "Blues"
                            hover_tmpl = (
                                "<b>%{customdata[0]}</b><br>"
                                "Latest: %{customdata[1]:.1f} mm<br>"
                                "As of: %{customdata[2]}<extra></extra>"
                            )

                        fig_city = px.choropleth(
                            df_map,
                            geojson=geojson_dict,
                            locations="City",
                            featureidkey="properties.shapeName",
                            color=map_color_col,
                            projection="mercator",
                            color_continuous_scale=colorscale
                        )
                        pad = 0.35
                        fig_city.update_geos(
                            projection_type="mercator",
                            fitbounds="locations",
                            lonaxis_range=[minx - pad, maxx + pad],
                            lataxis_range=[miny - pad, maxy + pad],
                            showland=False, showcountries=False, showcoastlines=False,
                            showocean=False, visible=False
                        )
                        fig_city.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=MAP_HEIGHT,
                            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                            hovermode="closest", showlegend=False,
                            coloraxis_colorbar=dict(title=colorbar_title)
                        )
                        df_map["_date_str"] = pd.to_datetime(df_map["Date"]).dt.strftime("%Y-%m")
                        fig_city.data[0].customdata = np.stack(
                            [df_map["City"].values, df_map[map_color_col].values, df_map["_date_str"].values], axis=-1
                        )
                        fig_city.data[0].hovertemplate = hover_tmpl
                        fig_city.update_traces(marker_line_width=0.3, marker_line_color="#999")

                        if choice == "Elevation" and elev_col is not None:
                            arr_chk = df_map[elev_use_col].to_numpy(dtype=float)
                            if np.isnan(arr_chk).any():
                                st.caption("White regions indicate **no elevation data** for those ADM1.")

                        if show_labels_current:
                            try:
                                lons, lats, names = adm1_label_points(iso3, GEOJSON_SOURCE)
                                value_map = {str(k): (float(v) if pd.notna(v) else np.nan)
                                             for k, v in zip(df_map["City"], df_map[map_color_col])}
                                colorscale_name = "Viridis" if choice == "Elevation" else "YlGnBu"
                                arr = df_map[map_color_col].to_numpy(dtype=float)
                                vmin = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
                                vmax = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
                                label_colors = pick_label_colors(names, value_map, colorscale_name, vmin, vmax)
                                mask_white = [c == "white" for c in label_colors]
                                mask_black = [c == "black" for c in label_colors]
                                if any(mask_white):
                                    fig_city.add_trace(go.Scattergeo(
                                        lon=np.array(lons)[mask_white], lat=np.array(lats)[mask_white],
                                        text=np.array(names)[mask_white], mode="text",
                                        textfont=dict(size=10, color="white"),
                                        hoverinfo="skip", showlegend=False
                                    ))
                                if any(mask_black):
                                    fig_city.add_trace(go.Scattergeo(
                                        lon=np.array(lons)[mask_black], lat=np.array(lats)[mask_black],
                                        text=np.array(names)[mask_black], mode="text",
                                        textfont=dict(size=10, color="black"),
                                        hoverinfo="skip", showlegend=False
                                    ))
                            except Exception as e:
                                st.caption(f"ADM1 labels unavailable for {iso3}: {e}")

                        st.plotly_chart(fig_city, use_container_width=True, config={"displayModeBar": False})

                        # --- BOTTOM-LEFT: Show labels toggle ---
                        st.toggle("Show ADM1 labels", value=show_labels_current, key=f"show_labels_p_{iso3}")

                        latest_month = df_map["Date"].max()
                        latest_month = pd.to_datetime(latest_month).strftime("%Y-%m") if pd.notna(latest_month) else "—"
                        regions_with_data = df_map["City"].nunique()
                        st.markdown(
                            f"""
                            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:.5rem; margin-bottom:.5rem;">
                              <span style="font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid #cbd5e1;color:#334155;">
                                Country: {display_country_name(iso3)} ({iso3})
                              </span>
                              <span style="font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid #cbd5e1;color:#334155;">
                                ADM1 with data: {regions_with_data}/{n_features}
                              </span>
                              <span style="font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid #cbd5e1;color:#334155;">
                                Latest month: {latest_month}
                              </span>
                            </div>
                            """, unsafe_allow_html=True
                        )
                        st.markdown("**Country Map**  \n"
                                    "Elevation data is from NASA SRTM (Feb 2000). Switch between elevation and precipitation using the control above.")

                    except FileNotFoundError as e:
                        st.info(str(e), icon="ℹ️")
            else:
                st.info("No latest-by-region precipitation values to map for this country.", icon="ℹ️")
        else:
            st.info("No city-level precipitation data available for this country.", icon="ℹ️")

# RIGHT: controls
with rc:
    if iso3 and iso3 != "—":
        _cm = CITY_MAP[CITY_MAP["Country"].astype(str).str.upper() == iso3]
        _cities = sorted(_cm["City"].dropna().astype(str).str.strip().unique().tolist())
    else:
        _cities = []
    city_options = ["Country (all)"] + _cities

    url_city = st.query_params.get("city", "")
    if url_city in city_options:
        st.session_state["opt_city_p"] = url_city
    elif st.session_state.get("opt_city_p") not in city_options:
        st.session_state["opt_city_p"] = "Country (all)"

    sel_city = st.selectbox(
        "Select Province/City/State",
        options=city_options,
        index=city_options.index(st.session_state["opt_city_p"]),
        key="opt_city_p",
        help="Use this dropdown to change the region focus for the charts."
    )
    desired_city = "" if sel_city in ("", "Country (all)") else sel_city
    if desired_city != st.query_params.get("city", ""):
        st.query_params.update({"city": desired_city})

    col_ind, col_form = st.columns([0.35, 0.65], gap="small")
    with col_ind:
        st.markdown("<div style='margin-top:0.2rem'></div>", unsafe_allow_html=True)
        indicator = st.radio(
            "Select climate indicator",
            INDICATOR_LABELS, index=4,
            key="opt_indicator_precip",
            help="Switch indicators. Temperature opens its own page."
        )
        if indicator == "Temperature":
            carry_iso = st.session_state.get("opt_iso3_p", "—")
            carry_city = st.session_state.get("opt_city_p", "")
            st.query_params.update({
                "iso3": "" if carry_iso == "—" else carry_iso,
                "city": "" if carry_city in ("", "Country (all)") else carry_city
            })
            try: st.switch_page("pages/1_Temperature_Dashboard.py")
            except Exception: st.switch_page("1_Temperature_Dashboard.py")

    with col_form:
        st.markdown("#### Chart Options")
        with st.form("options_form_precip", clear_on_submit=False):
            colA, colB, colC = st.columns(3)
            with colA:
                data_type = st.radio("Type", ["Historical Observations", "Projections (SSPs)"], index=0, key="opt_type_p")
            with colB:
                freq = st.radio("Frequency", ["Monthly", "Seasonal", "Annual"], index=0, key="opt_freq_p")
            with colC:
                source = st.radio("Data Source", ["CDS/CCKP", "CRU", "ERA5"], index=2, key="opt_source_p")
            st.form_submit_button("Apply changes", type="primary")

        st.markdown(
            f"""
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:0.25rem; margin-bottom:0.25rem;">
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Indicator: {indicator}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Type: {"Historical" if data_type.startswith("Historical") else "Projections"}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Frequency: {freq}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Source: {source}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Area: {('Country' if sel_city in ('', 'Country (all)') else sel_city)}</span>
            </div>
            """, unsafe_allow_html=True
        )

    # --- Data scope for charts ---
    iso3_now = st.query_params.get("iso3", (st.session_state.get("opt_iso3_p") or "")) or ""
    city_now = st.query_params.get("city", "") or st.session_state.get("opt_city_p", "")
    if iso3_now:
        using_city = (city_now not in ("", "Country (all)"))
        df_scope = (CITY_PRECIP if using_city else COUNTRY_PRECIP)
        df_scope = df_scope[df_scope["Country"].astype(str).str.upper() == iso3_now].copy()
    else:
        df_scope = COUNTRY_PRECIP.copy()

    df_scope = df_scope.rename(columns={
        "Precipitation (Sum)":"avg", "Precipitation (Variance)":"var",
        "Date":"date", "City":"City", "Country":"Country"
    })
    df_scope["date"] = pd.to_datetime(df_scope["date"], errors="coerce")
    df_scope = df_scope.dropna(subset=["date"]).sort_values("date")
    if city_now not in ("", "Country (all)"):
        df_scope = df_scope[df_scope["City"].astype(str).str.strip() == city_now.strip()]
    if df_scope.empty: st.warning("No precipitation data found for this selection."); st.stop()

    dmin = df_scope["date"].min().date(); dmax = df_scope["date"].max().date()
    dstart_default = pd.to_datetime(start_q).date() if start_q else dmin
    dend_default   = pd.to_datetime(end_q).date() if end_q else dmax
    dstart_date, dend_date = st.slider("Date range", min_value=dmin, max_value=dmax, value=(dstart_default, dend_default), format="YYYY-MM", key="date_slider_precip")
    st.query_params.update({"iso3": iso3_now, "city": "" if city_now in ("", "Country (all)") else city_now, "start": dstart_date.isoformat(), "end": dend_date.isoformat(), "page": "2_Precipitation_Dashboard"})
    mask = (df_scope["date"] >= pd.to_datetime(dstart_date)) & (df_scope["date"] <= pd.to_datetime(dend_date))
    series = df_scope.loc[mask, ["date","avg","var"]].reset_index(drop=True)

    st.markdown("""
        <style>
        div[data-testid="metric-container"] > div:last-child span {font-size: 18px !important;}
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {font-size: 28px !important;}
        </style>
    """, unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    latest = series["avg"].iloc[-1] if not series.empty else np.nan
    prev   = series["avg"].iloc[-2] if len(series) > 1 else np.nan
    with k1: st.metric("Latest Total Precip", f"{latest:.1f} mm" if np.isfinite(latest) else "—")
    with k2: st.metric("Δ vs previous point", f"{(latest - prev):+0.1f} mm" if np.isfinite(prev) and np.isfinite(latest) else "—")
    yoy_delta = "—"
    if not series.empty:
        last_date = pd.to_datetime(series["date"].iloc[-1]); target = (last_date - pd.DateOffset(years=1)).strftime("%Y-%m")
        prev_year_row = series[series["date"].dt.strftime("%Y-%m") == target]
        if not prev_year_row.empty and np.isfinite(latest):
            lyv = prev_year_row["avg"].iloc[-1]
            if np.isfinite(lyv): yoy_delta = f"{(latest - lyv):+0.1f} mm"
    with k3: st.metric("Δ vs same month LY", yoy_delta)
    with k4:
        mean_v = series["avg"].mean() if not series.empty else np.nan
        std_v  = series["avg"].std(ddof=0) if not series.empty else np.nan
        st.metric("Mean / σ in range", f"{mean_v:.1f} mm • {std_v:.1f}" if np.isfinite(mean_v) and np.isfinite(std_v) else "—")

    btn_dl, btn_custom = st.columns([0.55, 0.45], gap="small")
    with btn_dl:
        st.download_button(
            "Download data (CSV)", data=series.to_csv(index=False).encode("utf-8"),
            file_name=f"{iso3_now}_{('country' if city_now in ('', 'Country (all)') else city_now.replace(' ','_'))}_precipitation.csv",
            mime="text/csv", type="primary", use_container_width=True,
        )
    with btn_custom:
        if st.button("📈 Generate a custom chart", type="secondary", use_container_width=True):
            try: st.switch_page("pages/0_Custom_Chart.py")
            except Exception: st.switch_page("0_Custom_Chart.py")

    def _latest_year_chart(s: pd.DataFrame):
        if s.empty: return None
        end_dt = s["date"].max(); start_dt = end_dt - pd.DateOffset(years=1)
        s1 = s[(s["date"] >= start_dt) & (s["date"] <= end_dt)].copy()
        if s1.empty: s1 = s.tail(12).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s1["date"], y=s1["avg"], mode="lines+markers", name="Total Precip", line=dict(color=CBLIND["sky"])))
        fig.update_layout(height=260, margin=dict(l=30,r=30,t=40,b=40), title="Latest 12 months — Total Precipitation", xaxis_title="Date", yaxis_title="mm")
        return fig
    fig_last_year = _latest_year_chart(series)
    if fig_last_year is not None: st.plotly_chart(fig_last_year, use_container_width=True)

# ========================= Charts (own rows) =========================
st.markdown("---")

def chart_with_ribbon(title: str, s: pd.DataFrame, color_key="sky", ylab="mm"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["date"], y=s["avg"], mode="lines", name="Total", line=dict(color=CBLIND[color_key], width=2)))
    if "var" in s.columns and s["var"].notna().any():
        sigma = np.sqrt(s["var"].clip(lower=0)); upper = s["avg"] + sigma; lower = s["avg"] - sigma
        fig.add_trace(go.Scatter(x=s["date"], y=upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=s["date"], y=lower, mode="lines", fill='tonexty', line=dict(width=0), name="±1σ", hoverinfo='skip', fillcolor="rgba(86,180,233,0.18)"))
    fig.update_layout(height=320, margin=dict(l=30,r=30,t=40,b=40), title=title, xaxis_title="Date", yaxis_title=ylab, xaxis_title_standoff=34, yaxis_title_standoff=34)
    fig.update_xaxes(automargin=True, title_standoff=34); fig.update_yaxes(automargin=True, title_standoff=34)
    return fig

def story_block(title: str, text: str): st.markdown(f"**{title}**  \n{text}")

cL, cR = st.columns([0.2, 0.67], gap="large")
with cL: story_block("Story - Total Precipitation", "Describe totals, seasonality, and anomalies.")
with cR: st.plotly_chart(chart_with_ribbon("Total Precipitation", series, "sky"), use_container_width=True)

cL, cR = st.columns([0.2, 0.67], gap="large")
with cL: story_block("Story - Wet Conditions", "Discuss above-normal rainfall and flood risks.")
with cR: st.plotly_chart(chart_with_ribbon("Wet Conditions (placeholder)", series, "blue"), use_container_width=True)

cL, cR = st.columns([0.2, 0.67], gap="large")
with cL: story_block("Story - Dry Conditions", "Call out drought periods and rainfall deficits vs normal.")
with cR: st.plotly_chart(chart_with_ribbon("Dry Conditions (placeholder)", series, "orange"), use_container_width=True)

cL, cR = st.columns([0.2, 0.67], gap="large")
with cL: story_block("Story - Heavy Rainfall", "Highlight monthly maxima and heavy downpours.")
with cR: st.plotly_chart(chart_with_ribbon("Heavy Rainfall (placeholder)", series, "navy"), use_container_width=True)

# ===== Percentiles (simple radio version) =====
def percentile_series(dates: pd.Series, values: pd.Series, pct: int) -> pd.DataFrame:
    dfp = pd.DataFrame({"date": pd.to_datetime(dates), "val": pd.to_numeric(values, errors="coerce")}).dropna()
    if dfp.empty: return pd.DataFrame(columns=["date", "p"])
    dfp["month"] = dfp["date"].dt.month
    ref = dfp.groupby("month")["val"].quantile(pct / 100.0)
    return pd.DataFrame({"date": dfp["date"], "p": dfp["month"].map(ref)})

def percentile_chart(title_base: str, s: pd.DataFrame, pct: int, color_key="green", ylab="mm"):
    ps = percentile_series(s["date"], s["avg"], pct)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ps["date"], y=ps["p"], mode="lines", name=f"P{pct}", line=dict(color=CBLIND[color_key], width=2)))
    fig.update_layout(height=320, margin=dict(l=30,r=30,t=40,b=40), title=f"{title_base} — {pct}th Percentile", xaxis_title="Date", yaxis_title=ylab)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Percentile selection")
if "opt_pct_p" not in st.session_state: st.session_state["opt_pct_p"] = 50
st.radio("Percentile selection", [10,20,30,40,50,60,70,80,90,100], horizontal=True, key="opt_pct_p", label_visibility="collapsed")
pct = int(st.session_state["opt_pct_p"])

cL, cR = st.columns([0.2, 0.67], gap="large")
with cL: st.markdown("**Story - Percentile (Total Precip)**  \nDescribe what the chosen percentile shows for rainfall.")
with cR: percentile_chart("Total Precipitation", series, pct, "green")

cL, cR = st.columns([0.2, 0.67], gap="large")
with cL: st.markdown("**Story - Percentile (Wet Conditions)**  \nUpper-tail/wet risk context.")
with cR: percentile_chart("Wet Conditions (placeholder)", series, pct, "pink")

cL, cR = st.columns([0.2, 0.67], gap="large")
with cL: st.markdown("**Story - Percentile (Dry Conditions)**  \nLower-tail/dry risk context.")
with cR: percentile_chart("Dry Conditions (placeholder)", series, pct, "yellow")
