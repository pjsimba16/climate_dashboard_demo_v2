# pages/1_Temperature_Dashboard.py
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import streamlit as st
from huggingface_hub import hf_hub_download

# Optional: fast click capture for Plotly
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

# Optional: isolate reruns (Streamlit >= 1.31)
try:
    fragment = st.fragment
    _FRAGMENT_SUPPORTED = True
except Exception:
    _FRAGMENT_SUPPORTED = False

st.set_page_config(page_title="Temperature Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ====== SWITCH THIS WHEN READY ======
GEOJSON_SOURCE = "Hugging Face"   # or "Local"
# ====================================

# ---------- Visual constants ----------
INDICATOR_LABELS = [
    "Temperature", "Temperature Thresholds", "Heatwaves", "Coldwaves",
    "Precipitation", "Dry Conditions", "Wet Conditions", "Humidity", "Windspeeds"
]

ELEVATION_CANDIDATES = ["elevation", "Elevation", "elev", "elev_m", "elevation_m", "altitude", "Altitude"]

CBLIND = {
    "blue":"#0072B2","orange":"#E69F00","sky":"#56B4E9","green":"#009E73",
    "yellow":"#F0E442","navy":"#332288","verm":"#D55E00","pink":"#CC79A7","grey":"#999999","red":"#d62728"
}

# -------------------- Secrets / config helpers (UNCHANGED logic) --------------------
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
    tok = _secret_or_env("HF_TOKEN", "")
    return tok or None

# -------------------- Data loaders (UNCHANGED access pattern) --------------------
@st.cache_data(ttl=24*3600, show_spinner=False)
def _dl(filename: str) -> str:
    """Download a file from HF and return local path (HF caches; Streamlit caches this path)."""
    return hf_hub_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=filename, token=_get_hf_token())

@st.cache_data(ttl=24*3600, show_spinner=False)
def read_parquet_from_hf(filename: str) -> pd.DataFrame:
    return pd.read_parquet(_dl(filename))

@st.cache_data(ttl=7*24*3600, show_spinner=False)
def load_country_adm1_geojson(iso3: str, source: str):
    """Returns (geojson_dict, bounds, n_features)."""
    if source == "Hugging Face":
        path = _dl(f"ADM1_geodata/{iso3}.geojson")
    else:
        path = os.path.join("ADM1_geodata", f"{iso3}.geojson")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Local GeoJSON not found at {path}")
    gdf = gpd.read_file(path)
    try:
        gdf["geometry"] = gdf["geometry"].buffer(0)
    except Exception:
        pass
    gdf = gdf.to_crs(4326)
    bounds = tuple(gdf.total_bounds)
    n_features = len(gdf)
    return gdf.__geo_interface__, bounds, n_features

# -------------------- Datasets (UNCHANGED) --------------------
with st.spinner("Loading base datasets…"):
    CITY_TEMP     = read_parquet_from_hf("city_temperature.snappy.parquet")
    COUNTRY_TEMP  = read_parquet_from_hf("country_temperature.snappy.parquet")
    #CITY_MAP      = read_parquet_from_hf("city_mapper_with_coords_v2.snappy.parquet")
    CITY_MAP      = pd.read_csv('city_mapper_with_coords_v3.csv')

# -------------------- Display helpers (names) --------------------
try:
    import pycountry
except Exception:
    pycountry = None

_CUSTOM_COUNTRY_DISPLAY = {
    "CHN": "People's Republic of China",
    "TWN": "Taipei,China",
    "HKG": "Hong Kong, China",
}
def iso3_to_name(iso: str) -> str:
    iso = (iso or "").upper().strip()
    if pycountry:
        try:
            c = pycountry.countries.get(alpha_3=iso)
            if c and getattr(c, "name", None):
                return c.name
        except Exception:
            pass
    return iso

def display_country_name(iso: str) -> str:
    iso = (iso or "").upper().strip()
    if iso in _CUSTOM_COUNTRY_DISPLAY:
        return _CUSTOM_COUNTRY_DISPLAY[iso]
    return iso3_to_name(iso)

# ------------------ Read URL query params ------------------
qp = st.query_params
iso3_q  = (qp.get("iso3") or st.session_state.get("nav_iso3") or "").upper()
city_q  = qp.get("city", "")
start_q = qp.get("start", "")
end_q   = qp.get("end", "")

# ---------------------- Header ----------------------
top_l, top_r = st.columns([0.12, 0.88])
with top_l:
    if st.button("← Home", help="Back to Home map"):
        keep_iso3 = st.query_params.get("iso3", "")
        keep_city = st.query_params.get("city", "")
        st.query_params.clear()
        if keep_iso3:
            st.query_params.update({"iso3": keep_iso3})
        if keep_city:
            st.query_params.update({"city": keep_city})
        try:
            st.switch_page("Home_Page.py")
        except Exception:
            st.rerun()
st.markdown(f"### Temperature - {display_country_name(iso3_q) if iso3_q else '…'}")

# --------------- Country list & single selector (left) ----------
countries_iso = sorted(COUNTRY_TEMP["Country"].dropna().astype(str).str.upper().unique().tolist())
country_options = ["—"] + countries_iso

# URL always wins for country (initialize selectbox default from URL)
if iso3_q in country_options:
    st.session_state["opt_iso3"] = iso3_q
elif "opt_iso3" not in st.session_state:
    st.session_state["opt_iso3"] = "—"

# -------------------------- Top Layout (map left, controls right) -------------------------------
lc, rc = st.columns([0.34, 0.66], gap="large")

# LEFT: Country select (single), Map + badges + story
with lc:
    iso3 = st.selectbox(
        "Select Country",
        options=country_options,
        index=country_options.index(st.session_state["opt_iso3"]) if st.session_state["opt_iso3"] in country_options else 0,
        key="opt_iso3",
        format_func=lambda v: ("Select…" if v=="—" else display_country_name(v)),
        help="Pick a country, or click one on the Home map to arrive here pre-selected."
    )

    # write iso3 to URL; if country changed vs current URL, clear city to avoid stale selections
    desired_iso = "" if iso3 == "—" else iso3
    current_iso = st.query_params.get("iso3", "")
    if desired_iso != current_iso:
        updates = {"iso3": desired_iso, "city": ""}  # clear city only when country changes
        st.query_params.update(updates)
        st.rerun()

    MAP_HEIGHT = 640

    if iso3 and iso3 != "—":
        df_iso = CITY_TEMP[CITY_TEMP["Country"].astype(str).str.upper() == iso3].copy()
        if not df_iso.empty:
            df_iso["Date"] = pd.to_datetime(df_iso["Date"], errors="coerce")
            df_iso = df_iso.dropna(subset=["Date"])
            df_iso["City"] = df_iso["City"].astype(str)
            # --- Build latest-by-ADM1 table (unchanged) ---
            try:
                grp = df_iso.sort_values("Date").groupby("City", observed=True)["Date"]
                idx = grp.idxmax()
                df_map = df_iso.loc[idx, ["City", "Temperature (Mean)", "Date"]].copy()
            except Exception:
                df_map = (
                    df_iso.sort_values(["City", "Date"])
                        .drop_duplicates(subset=["City"], keep="last")[["City", "Temperature (Mean)", "Date"]]
                        .copy()
                )

            if not df_map.empty:
                with st.spinner("Loading map & drawing choropleth…"):
                    try:
                        geojson_dict, bounds, n_features = load_country_adm1_geojson(iso3, GEOJSON_SOURCE)
                        minx, miny, maxx, maxy = bounds

                        # ---- Elevation merge (if present) ----
                        cm_iso = CITY_MAP[CITY_MAP["Country"].astype(str).str.upper() == iso3].copy()
                        elev_col = next((c for c in ELEVATION_CANDIDATES if c in cm_iso.columns), None)

                        colorbar_title = "Temperature"
                        colorscale = "YlOrRd"
                        hover_tmpl = (
                            "<b>%{customdata[0]}</b><br>"
                            "Latest: %{customdata[1]:.2f} °C<br>"
                            "As of: %{customdata[2]}<extra></extra>"
                        )
                        map_color_col = "Temperature (Mean)"

                        if elev_col is not None:
                            df_elev = cm_iso[["City", elev_col]].copy()
                            df_elev[elev_col] = pd.to_numeric(df_elev[elev_col], errors="coerce")
                            df_map = df_map.merge(df_elev, on="City", how="left")

                            total = len(df_map)
                            avail = int(df_map[elev_col].notna().sum())
                            if avail == total and total > 0:
                                map_color_col = elev_col
                                colorbar_title = "Elevation (m)"
                                colorscale = "Viridis"
                                hover_tmpl = (
                                    "<b>%{customdata[0]}</b><br>"
                                    "Elevation: %{customdata[1]:.0f} m<br>"
                                    "As of: %{customdata[2]}<extra></extra>"
                                )
                            else:
                                st.warning(
                                    f"Only {avail}/{total} ADM1 have elevation - falling back to temperature coloring.",
                                    icon="⚠️"
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
                            paper_bgcolor="#ffffff",
                            plot_bgcolor="#ffffff",
                            hovermode="closest",
                            showlegend=False,
                            coloraxis_colorbar=dict(title=colorbar_title)
                        )
                        df_map["_date_str"] = pd.to_datetime(df_map["Date"]).dt.strftime("%Y-%m")
                        fig_city.data[0].customdata = np.stack(
                            [
                                df_map["City"].values,
                                df_map[map_color_col].values,
                                df_map["_date_str"].values
                            ],
                            axis=-1
                        )
                        fig_city.data[0].hovertemplate = hover_tmpl
                        fig_city.update_traces(marker_line_width=0.3, marker_line_color="#999")
                        st.plotly_chart(fig_city, use_container_width=True, config={"displayModeBar": False})

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
                            """,
                            unsafe_allow_html=True
                        )

                        # Story (non-editable)
                        st.markdown("**Country Map**  \n"
                                    "Elevation data is extracted from NASA's global topographic database collected through their Shuttle Radar Topography Mission (SRTM) in February of 2000. In case elevation isn't fully populated, the map falls back to temperature data.")

                    except FileNotFoundError as e:
                        st.info(str(e), icon="ℹ️")


            else:
                st.info("No latest-by-region temperature values to map for this country.", icon="ℹ️")
        else:
            st.info("No city-level temperature data available for this country.", icon="ℹ️")

# RIGHT: controls — first row aligns with country selector
with rc:
    # Build city list for the chosen country
    if iso3 and iso3 != "—":
        _cm = CITY_MAP[CITY_MAP["Country"].astype(str).str.upper() == iso3]
        _cities = sorted(_cm["City"].dropna().astype(str).str.strip().unique().tolist())
    else:
        _cities = []
    city_options = ["Country (all)"] + _cities

    # URL always wins for city (initialize from URL if valid for this country)
    url_city = st.query_params.get("city", "")
    if url_city in city_options:
        st.session_state["opt_city"] = url_city
    elif st.session_state.get("opt_city") not in city_options:
        st.session_state["opt_city"] = "Country (all)"

    sel_city = st.selectbox(
        "Select Province/City/State",
        options=city_options,
        index=city_options.index(st.session_state["opt_city"]),
        key="opt_city",
        help="Use this dropdown to change the region focus for the charts."
    )

    # persist city to URL (do NOT clear when switching indicators)
    qp_city = st.query_params.get("city", "")
    desired_city = "" if sel_city in ("", "Country (all)") else sel_city
    if desired_city != qp_city:
        st.query_params.update({"city": desired_city})

    # Two columns: left = climate indicator (narrow), right = Chart Options (wider)
    col_ind, col_form = st.columns([0.35, 0.65], gap="small")

    with col_ind:
        st.markdown("<div style='margin-top:0.2rem'></div>", unsafe_allow_html=True)
        indicator = st.radio(
            "Select climate indicator",
            INDICATOR_LABELS,
            index=0,
            key="opt_indicator_temp",
            help="Switch indicators. Precipitation opens its own page."
        )
        if indicator == "Precipitation":
            # carry BOTH iso3 and city across to the other page
            carry_iso = st.session_state.get("opt_iso3", "—")
            carry_city = st.session_state.get("opt_city", "")
            st.query_params.update({
                "iso3": "" if carry_iso == "—" else carry_iso,
                "city": "" if carry_city in ("", "Country (all)") else carry_city
            })
            try:
                st.switch_page("pages/2_Precipitation_Dashboard.py")
            except Exception:
                st.switch_page("2_Precipitation_Dashboard.py")

    with col_form:
        st.markdown("#### Chart Options")
        with st.form("options_form_temp", clear_on_submit=False):
            colA, colB, colC = st.columns(3)
            with colA:
                data_type = st.radio(
                    "Type",
                    ["Historical Observations", "Projections (SSPs)"],
                    index=0, key="opt_type_t",
                )
            with colB:
                freq = st.radio(
                    "Frequency",
                    ["Monthly", "Seasonal", "Annual"],
                    index=0, key="opt_freq_t",
                )
            with colC:
                source = st.radio(
                    "Data Source",
                    ["CDS/CCKP", "CRU", "ERA5"],
                    index=2, key="opt_source_t",
                )
            st.form_submit_button("Apply changes", type="primary")

        # Active badges under Chart Options
        st.markdown(
            f"""
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:0.25rem; margin-bottom:0.25rem;">
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Indicator: {indicator}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Type: {"Historical" if data_type.startswith("Historical") else "Projections"}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Frequency: {freq}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Source: {source}</span>
              <span style="font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid #e5e7eb;color:#374151;">Area: {('Country' if sel_city in ('', 'Country (all)') else sel_city)}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Data scope (UNCHANGED, but read iso3/city from URL/session) ---
    iso3_now = st.query_params.get("iso3", (st.session_state.get("opt_iso3") or "")) or ""
    city_now = st.query_params.get("city", "") or st.session_state.get("opt_city", "")
    if iso3_now:
        using_city = (city_now not in ("", "Country (all)"))
        df_scope = (CITY_TEMP if using_city else COUNTRY_TEMP)
        df_scope = df_scope[df_scope["Country"].astype(str).str.upper() == iso3_now].copy()
    else:
        df_scope = COUNTRY_TEMP.copy()

    df_scope = df_scope.rename(columns={
        "Temperature (Mean)":"avg",
        "Temperature (Variance)":"var",
        "Date":"date",
        "City":"City",
        "Country":"Country"
    })
    df_scope["date"] = pd.to_datetime(df_scope["date"], errors="coerce")
    df_scope = df_scope.dropna(subset=["date"]).sort_values("date")
    if city_now not in ("", "Country (all)"):
        df_scope = df_scope[df_scope["City"].astype(str).str.strip() == city_now.strip()]
    if df_scope.empty:
        st.warning("No temperature data found for this selection.")
        st.stop()

    # ---------------- Date Range (RIGHT column) ----------------
    dmin = df_scope["date"].min().date(); dmax = df_scope["date"].max().date()
    dstart_default = pd.to_datetime(start_q).date() if start_q else dmin
    dend_default   = pd.to_datetime(end_q).date() if end_q else dmax
    dstart_date, dend_date = st.slider(
        "Date range",
        min_value=dmin, max_value=dmax, value=(dstart_default, dend_default),
        format="YYYY-MM",
        key="date_slider_temp",
        help="Drag the handles to restrict the period shown in all charts and KPIs."
    )
    st.query_params.update({
        "iso3": iso3_now,
        "city": "" if city_now in ("", "Country (all)") else city_now,
        "start": dstart_date.isoformat(),
        "end": dend_date.isoformat(),
        "page": "1_Temperature_Dashboard",
    })
    mask = (df_scope["date"] >= pd.to_datetime(dstart_date)) & (df_scope["date"] <= pd.to_datetime(dend_date))
    series = df_scope.loc[mask, ["date","avg","var"]].reset_index(drop=True)

    # ---------------- KPIs (RIGHT column) ----------------
    st.markdown("""
        <style>
        div[data-testid="metric-container"] > div:last-child span {font-size: 18px !important;}
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {font-size: 28px !important;}
        </style>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    latest = series["avg"].iloc[-1] if not series.empty else np.nan
    prev   = series["avg"].iloc[-2] if len(series) > 1 else np.nan
    with k1:
        st.metric("Latest Avg Temp",
                  f"{latest:.2f} °C" if np.isfinite(latest) else "—",
                  help="The most recent value in the filtered series.")
    with k2:
        dprev = f"{(latest - prev):+0.2f} °C" if np.isfinite(prev) and np.isfinite(latest) else "—"
        st.metric("Δ vs previous point", dprev,
                  help="Latest value minus the immediately previous data point.")
    yoy_delta = "—"
    if not series.empty:
        last_date = pd.to_datetime(series["date"].iloc[-1])
        target = (last_date - pd.DateOffset(years=1)).strftime("%Y-%m")
        prev_year_row = series[series["date"].dt.strftime("%Y-%m") == target]
        if not prev_year_row.empty and np.isfinite(latest):
            last_year_val = prev_year_row["avg"].iloc[-1]
            if np.isfinite(last_year_val):
                yoy_delta = f"{(latest - last_year_val):+0.2f} °C"
    with k3:
        st.metric("Δ vs same month LY", yoy_delta,
                  help="Latest value minus the value from the same calendar month one year earlier.")
    with k4:
        mean_v = series["avg"].mean() if not series.empty else np.nan
        std_v  = series["avg"].std(ddof=0) if not series.empty else np.nan
        st.metric("Mean / σ in range",
                  f"{mean_v:.2f} °C • {std_v:.2f}" if np.isfinite(mean_v) and np.isfinite(std_v) else "—",
                  help="Arithmetic mean and population standard deviation over the selected range.")

    # ---------------- Download + custom chart (RIGHT column) ----------------
    btn_dl, btn_custom = st.columns([0.55, 0.45], gap="small")
    with btn_dl:
        st.download_button(
            "Download data (CSV)",
            data=series.to_csv(index=False).encode("utf-8"),
            file_name=f"{iso3_now}_{('country' if city_now in ('', 'Country (all)') else city_now.replace(' ','_'))}_temperature.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
    with btn_custom:
        if st.button("📈 Generate a custom chart", type="secondary", use_container_width=True):
            try:
                st.switch_page("pages/0_Custom_Chart.py")
            except Exception:
                st.switch_page("0_Custom_Chart.py")

    # --- Compact latest-year chart to fill space under buttons ---
    def _latest_year_chart(s: pd.DataFrame):
        if s.empty:
            return None
        end_dt = s["date"].max()
        start_dt = end_dt - pd.DateOffset(years=1)
        s1 = s[(s["date"] >= start_dt) & (s["date"] <= end_dt)].copy()
        if s1.empty:
            s1 = s.tail(12).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s1["date"], y=s1["avg"], mode="lines+markers",
                                 name="Avg Temp", line=dict(color=CBLIND["blue"])))
        fig.update_layout(
            height=260, margin=dict(l=30,r=30,t=40,b=40),
            title="Latest 12 months — Average Temperature",
            xaxis_title="Date", yaxis_title="°C"
        )
        return fig

    fig_last_year = _latest_year_chart(series)
    if fig_last_year is not None:
        st.plotly_chart(fig_last_year, use_container_width=True)

# =========================
# CHARTS (own rows below) — story text (non-editable) on the LEFT
# =========================
st.markdown("---")

def chart_with_ribbon(title: str, s: pd.DataFrame, color_key="blue", ylab="°C"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["date"], y=s["avg"], mode="lines",
                             name="Average", line=dict(color=CBLIND[color_key], width=2)))
    if "var" in s.columns and s["var"].notna().any():
        sigma = np.sqrt(s["var"].clip(lower=0))
        upper = s["avg"] + sigma
        lower = s["avg"] - sigma
        fig.add_trace(go.Scatter(x=s["date"], y=upper, mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=s["date"], y=lower, mode="lines", fill='tonexty',
                                 line=dict(width=0), name="±1σ", hoverinfo='skip',
                                 fillcolor="rgba(0,114,178,0.18)"))
    fig.update_layout(
        height=320, margin=dict(l=30,r=30,t=40,b=40),
        title=title, xaxis_title="Date", yaxis_title=ylab,
        xaxis_title_standoff=34, yaxis_title_standoff=34, template=None
    )
    fig.update_xaxes(automargin=True, title_standoff=34)
    fig.update_yaxes(automargin=True, title_standoff=34)
    return fig

def story_block(title: str, text: str):
    st.markdown(f"**{title}**  \n{text}")

# Average Temperature
cL, cR = st.columns([0.2, 0.67], gap="large")
with cL:
    story_block("Story - Average Temperature",
                "Describe the headline signal or anomaly highlighted by the series. "
                "Update this text in the code as the narrative evolves.")
with cR:
    st.plotly_chart(chart_with_ribbon("Average Temperature", series, "blue"), use_container_width=True)

# Diurnal Temperature Range (placeholder)
cL, cR = st.columns([0.2, 0.67], gap="large")
with cL:
    story_block("Story - Diurnal Temperature Range",
                "Discuss range widening/narrowing and likely drivers (cloud cover, humidity, land use, etc.).")
with cR:
    st.plotly_chart(chart_with_ribbon("Diurnal Temperature Range (placeholder)", series, "orange"), use_container_width=True)

# Maximum Temperature (placeholder)
cL, cR = st.columns([0.2, 0.67], gap="large")
with cL:
    story_block("Story - Maximum Temperature",
                "Call out heat spikes, threshold exceedances, and any emerging extremes.")
with cR:
    st.plotly_chart(chart_with_ribbon("Maximum Temperature (placeholder)", series, "verm"), use_container_width=True)

# Minimum Temperature (placeholder)
cL, cR = st.columns([0.2, 0.67], gap="large")
with cL:
    story_block("Story - Minimum Temperature",
                "Discuss night-time warming, cold spells, frost risk, or lower-tail changes.")
with cR:
    st.plotly_chart(chart_with_ribbon("Minimum Temperature (placeholder)", series, "navy"), use_container_width=True)

# ===== Percentile selector + charts (selector FLUSH-LEFT under title) =====
def percentile_series(dates: pd.Series, values: pd.Series, pct: int) -> pd.DataFrame:
    dfp = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "val": pd.to_numeric(values, errors="coerce"),
    }).dropna()
    if dfp.empty:
        return pd.DataFrame(columns=["date", "p"])
    dfp["month"] = dfp["date"].dt.month
    ref = dfp.groupby("month")["val"].quantile(pct / 100.0)
    return pd.DataFrame({"date": dfp["date"], "p": dfp["month"].map(ref)})

def percentile_chart(title_base: str, s: pd.DataFrame, pct: int, color_key="green", ylab="°C"):
    ps = percentile_series(s["date"], s["avg"], pct)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ps["date"], y=ps["p"], mode="lines",
        name=f"P{pct}", line=dict(color=CBLIND[color_key], width=2)
    ))
    fig.update_layout(
        height=320, margin=dict(l=30, r=30, t=40, b=40),
        title=f"{title_base} — {pct}th Percentile",
        xaxis_title="Date", yaxis_title=ylab,
        xaxis_title_standoff=34, yaxis_title_standoff=34
    )
    st.plotly_chart(fig, use_container_width=True)

def _percentile_block():
    st.markdown("#### Percentile selection")
    if "opt_pct_t" not in st.session_state:
        st.session_state["opt_pct_t"] = 50
    st.radio(
        "Percentile selection",
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        horizontal=True,
        key="opt_pct_t",
        label_visibility="collapsed",
    )
    pct = int(st.session_state["opt_pct_t"])

    cL, cR = st.columns([0.2, 0.67], gap="large")
    with cL:
        st.markdown("**Story - Percentile (Avg Temp)**  \nDescribe what the chosen percentile shows.")
    with cR:
        percentile_chart("Average Temperature", series, pct, "green")

    cL, cR = st.columns([0.2, 0.67], gap="large")
    with cL:
        st.markdown("**Story - Percentile (Tmax)**  \nUpper-tail/heat risk context.")
    with cR:
        percentile_chart("Maximum Temperature (placeholder)", series, pct, "pink")

    cL, cR = st.columns([0.2, 0.67], gap="large")
    with cL:
        st.markdown("**Story - Percentile (Tmin)**  \nLower-tail/cold risk context.")
    with cR:
        percentile_chart("Minimum Temperature (placeholder)", series, pct, "sky")

_percentile_block()
