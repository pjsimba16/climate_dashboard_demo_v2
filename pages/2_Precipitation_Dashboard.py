# pages/2_Precipitation_Dashboard.py
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

st.set_page_config(page_title="Precipitation Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ====== SWITCH THIS WHEN READY ======
GEOJSON_SOURCE = "Hugging Face"   # change to "Hugging Face" after upload
# ====================================

# One-time note about caching
if "_shown_cache_note" not in st.session_state:
    st.info("First load may take a bit longer while data and map shapes are cached. Subsequent loads will be much faster.")
    st.session_state["_shown_cache_note"] = True

try:
    import pycountry
except Exception:
    pycountry = None

INDICATOR_LABELS = [
    "Temperature", "Temperature Thresholds", "Heatwaves", "Coldwaves",
    "Precipitation", "Dry Conditions", "Wet Conditions", "Humidity", "Windspeeds"
]
CBLIND = {
    "blue":"#0072B2","orange":"#E69F00","sky":"#56B4E9","green":"#009E73",
    "yellow":"#F0E442","navy":"#332288","verm":"#D55E00","pink":"#CC79A7","grey":"#999999","red":"#d62728"
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

# Custom display names for specific ISO3s
_CUSTOM_COUNTRY_DISPLAY = {
    "CHN": "People's Republic of China",
    "TWN": "Taipei,China",
    "HKG": "Hong Kong, China",
}

def display_country_name(iso: str) -> str:
    iso = (iso or "").upper().strip()
    if iso in _CUSTOM_COUNTRY_DISPLAY:
        return _CUSTOM_COUNTRY_DISPLAY[iso]
    # fall back to pycountry -> e.g., "France", "Philippines"
    return iso3_to_name(iso)


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

# Percentile helper (cached)
@st.cache_data(ttl=3600, show_spinner=False)
def percentile_series_cached(df_dates: pd.Series, df_values: pd.Series, pct: int) -> pd.DataFrame:
    dfp = pd.DataFrame({"date": pd.to_datetime(df_dates), "val": pd.to_numeric(df_values, errors="coerce")}).dropna()
    if dfp.empty:
        return pd.DataFrame(columns=["date","p"])
    dfp["month"] = dfp["date"].dt.month
    ref = dfp.groupby("month")["val"].quantile(pct/100.0)
    out = pd.DataFrame({"date": dfp["date"], "p": dfp["month"].map(ref)})
    return out

# Compact Plotly percentile selector (no 'config=' to support older component versions)
def percentile_selector_plotly(state_key: str, default_val: int = 50, height: int = 72, width: int = 520) -> int:
    """
    Compact 10-box selector (10..100). One click updates the selected box color
    and, via fragment, re-renders only the percentile block.
    """
    values = [10,20,30,40,50,60,70,80,90,100]
    if state_key not in st.session_state:
        st.session_state[state_key] = default_val
    current = st.session_state[state_key]

    import plotly.graph_objects as go
    x = list(range(len(values)))
    y = [0]*len(values)

    # Pre-color with current selection
    fill_colors = ["#2563eb" if v == current else "#e2e8f0" for v in values]
    line_colors = ["#1d4ed8" if v == current else "#94a3b8" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text",
        text=[str(v) for v in values],
        textposition="middle center",
        marker=dict(
            symbol="square",
            size=28,  # compact boxes
            color=fill_colors,
            line=dict(width=2, color=line_colors)
        ),
        hovertemplate="<b>%{text}th percentile</b><extra></extra>",
        name="percentiles"
    ))
    # Tight layout so boxes sit side-by-side with minimal whitespace
    fig.update_xaxes(visible=False, range=[-0.5, len(values)-0.5], fixedrange=True, constrain="domain")
    fig.update_yaxes(visible=False, range=[-1, 1], fixedrange=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )

    st.write("Select percentile")
    if plotly_events is None:
        st.warning("Install `streamlit-plotly-events` to enable the fast percentile selector.", icon="⚠️")
        st.plotly_chart(fig, use_container_width=False)
        return current

    # Render ONCE through the component (avoid duplicate rows)
    events = plotly_events(
        fig,
        click_event=True, hover_event=False, select_event=False,
        override_height=height, override_width=width,
        key=f"{state_key}_events"
    )
    if events:
        idx = events[0].get("pointIndex")
        if isinstance(idx, int) and 0 <= idx < len(values):
            new_val = values[idx]
            if new_val != current:
                st.session_state[state_key] = new_val
                st.rerun()  # recolors boxes + refreshes only the fragment

    return st.session_state[state_key]

# -------------------- Datasets --------------------
with st.spinner("Loading base datasets…"):
    CITY_PR     = read_parquet_from_hf("city_precipitation.snappy.parquet")
    COUNTRY_PR  = read_parquet_from_hf("country_precipitation.snappy.parquet")
    CITY_MAP    = read_parquet_from_hf("city_mapper_with_coords_v2.snappy.parquet")

# ------------------ Read URL query params ------------------
qp = st.query_params
iso3_q  = (qp.get("iso3") or st.session_state.get("nav_iso3") or "").upper()
city_q  = qp.get("city", "")
start_q = qp.get("start", "")
end_q   = qp.get("end", "")

# Title + Home
title_country = display_country_name(iso3_q) if iso3_q else "…"
top_l, top_r = st.columns([0.12, 0.88])
with top_l:
    if st.button("← Home", help="Back to Home map"):
        st.query_params.clear()
        try:
            st.switch_page("Home_Page.py")
        except Exception:
            st.rerun()
st.markdown(f"### Precipitation - {title_country}")

# --------------- Precompute lists ----------
countries_iso = sorted(COUNTRY_PR["Country"].dropna().astype(str).str.upper().unique().tolist())
country_options = ["—"] + countries_iso

if "opt_iso3_p" not in st.session_state:
    st.session_state["opt_iso3_p"] = iso3_q if iso3_q in country_options else "—"

# -------------------------- Layout -------------------------------
lc, rc = st.columns([0.34, 0.66], gap="large")

with lc:
    # Country selector
    iso3 = st.selectbox(
        "Select Country",
        options=country_options,
        key="opt_iso3_p",
        format_func=lambda v: ("Select…" if v=="—" else display_country_name(v)),
        help="Pick a country, or click one on the Home map to arrive here pre-selected."
    )
    if iso3 != (iso3_q if iso3_q in country_options else "—"):
        st.query_params.update({"iso3": "" if iso3=="—" else iso3, "city": ""})
        st.rerun()

    MAP_HEIGHT = 640

    # ---------- Map ----------
    if iso3 and iso3 != "—":
        df_iso = CITY_PR[CITY_PR["Country"].astype(str).str.upper() == iso3].copy()
        if not df_iso.empty:
            df_iso["Date"] = pd.to_datetime(df_iso["Date"], errors="coerce")
            df_iso = df_iso.dropna(subset=["Date"])
            df_iso["City"] = df_iso["City"].astype(str)

            try:
                grp = df_iso.sort_values("Date").groupby("City", observed=True)["Date"]
                idx = grp.idxmax()
                df_map = df_iso.loc[idx, ["City", "Precipitation (Sum)", "Date"]].copy()
            except Exception:
                df_map = (
                    df_iso.sort_values(["City", "Date"])
                          .drop_duplicates(subset=["City"], keep="last")[["City", "Precipitation (Sum)", "Date"]]
                          .copy()
                )

            if not df_map.empty:
                with st.spinner("Loading map & drawing choropleth…"):
                    try:
                        geojson_dict, bounds, n_features = load_country_adm1_geojson(iso3, GEOJSON_SOURCE)
                        minx, miny, maxx, maxy = bounds

                        fig_city = px.choropleth(
                            df_map,
                            geojson=geojson_dict,
                            locations="City",
                            featureidkey="properties.shapeName",
                            color="Precipitation (Sum)",
                            projection="mercator",
                            color_continuous_scale="YlGnBu"
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
                            coloraxis_colorbar=dict(title="Precipitation")
                        )
                        df_map["_date_str"] = df_map["Date"].dt.strftime("%Y-%m")
                        fig_city.data[0].customdata = np.stack(
                            [df_map["City"].values,
                             df_map["Precipitation (Sum)"].values,
                             df_map["_date_str"].values],
                            axis=-1
                        )
                        fig_city.data[0].hovertemplate = (
                            "<b>%{customdata[0]}</b><br>"
                            "Latest: %{customdata[1]:.1f} mm<br>"
                            "As of: %{customdata[2]}<extra></extra>"
                        )
                        fig_city.update_traces(marker_line_width=0.3, marker_line_color="#999")
                        st.plotly_chart(fig_city, use_container_width=True, config={"displayModeBar": False})

                        # ===== Country info chips =====
                        latest_month = df_map["Date"].max().strftime("%Y-%m")
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
                    except FileNotFoundError as e:
                        st.info(str(e), icon="ℹ️")
            else:
                st.info("No latest-by-region precipitation values to map for this country.", icon="ℹ️")
        else:
            st.info("No city-level precipitation data available for this country.", icon="ℹ️")

    # ---------- City dropdown (drives the charts) ----------
    if iso3 and iso3 != "—":
        _cm = CITY_MAP[CITY_MAP["Country"].astype(str).str.upper() == iso3]
        _cities = sorted(_cm["City"].dropna().astype(str).str.strip().unique().tolist())
    else:
        _cities = []
    city_options = ["Country (all)"] + _cities

    if "opt_city_p" not in st.session_state or st.session_state["opt_city_p"] not in city_options:
        st.session_state["opt_city_p"] = "Country (all)"

    sel_city = st.selectbox(
        "Select Province/City/State",
        options=city_options,
        key="opt_city_p",
        help="Use this dropdown to change the region focus for the charts."
    )

    # Methodology (left column)
    with st.expander("Methodology & Notes", expanded=False):
        st.markdown(
            """
            - **ADM1 matching:** Regions are matched by `properties.shapeName` in the country ADM1 GeoJSON.
            - **Map values:** The choropleth shows the **latest available** monthly value per ADM1.
            - **Percentiles:** For each calendar month (Jan…Dec), we compute quantiles from the historical monthly series. The line shows the selected percentile over time (mapped by month).
            - **Variance bands:** The ±1σ band uses the reported monthly variance when present; otherwise, only the mean line is shown.
            - **Bounds:** Map is clipped to the country’s ADM1 extent; outside areas are hidden.
            """
        )

    # Indicator selector (vertical)
    indicator = st.radio(
        "Select climate indicator",
        INDICATOR_LABELS,
        index=4,
        key="opt_indicator_prec",
        help="Switch indicators. Temperature opens its own page."
    )
    if indicator == "Temperature":
        st.query_params.update({"page": "1_Temperature_Dashboard"})
        st.switch_page("pages/1_Temperature_Dashboard.py")

with rc:
    st.markdown("#### Options")
    with st.form("options_form_prec", clear_on_submit=False):
        colA, colB, colC = st.columns(3)
        with colA:
            data_type = st.radio(
                "Type",
                ["Historical Observations", "Projections (SSPs)"],
                index=0, key="opt_type_p",
                help="Historical: reanalysis/observed datasets.\nProjections: scenario-based model outputs (SSPs)."
            )
        with colB:
            freq = st.radio(
                "Frequency",
                ["Monthly", "Seasonal", "Annual"],
                index=0, key="opt_freq_p",
                help="Temporal aggregation of the time series shown in all charts."
            )
        with colC:
            source = st.radio(
                "Data Source",
                ["CDS/CCKP", "CRU", "ERA5"],
                index=2, key="opt_source_p",
                help="Current demo uses ERA5. CRU/CDS/CCKP will be enabled as data becomes available."
            )
        st.form_submit_button("Apply changes", type="primary")

    # Chips (right side)
    st.markdown(
        f"""
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:-.35rem; margin-bottom:.7rem;">
          <span style="font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid #cbd5e1;color:#334155;">Data: {source}</span>
          <span style="font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid #cbd5e1;color:#334155;">Frequency: {freq}</span>
          <span style="font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid #cbd5e1;color:#334155;">Type: {"Historical" if data_type.startswith("Historical") else "Projections"}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Scope & data prep
    iso3_now = st.query_params.get("iso3", (st.session_state.get("opt_iso3_p") or "")) or ""
    city_now = st.query_params.get("city", "") or st.session_state.get("opt_city_p", "")

    if iso3_now:
        using_city = (city_now not in ("", "Country (all)"))
        df_scope = (CITY_PR if using_city else COUNTRY_PR)
        df_scope = df_scope[df_scope["Country"].astype(str).str.upper() == iso3_now].copy()
    else:
        df_scope = COUNTRY_PR.copy()

    df_scope = df_scope.rename(columns={
        "Precipitation (Sum)":"sum",
        "Precipitation (Variance)":"var",
        "Date":"date",
        "City":"City",
        "Country":"Country"
    })
    df_scope["date"] = pd.to_datetime(df_scope["date"], errors="coerce")
    df_scope = df_scope.dropna(subset=["date"]).sort_values("date")
    if city_now not in ("", "Country (all)"):
        df_scope = df_scope[df_scope["City"].astype(str).str.strip() == city_now.strip()]

    if df_scope.empty:
        st.warning("No precipitation data found for this selection.")
        st.stop()

    # Mini sparkline (last 12 months)
    with st.spinner("Computing mini trend…"):
        s_scope = df_scope[["date","sum"]].dropna().copy()
        last12 = s_scope.tail(12)
        if not last12.empty:
            fig_sp = go.Figure()
            fig_sp.add_trace(go.Scatter(x=last12["date"], y=last12["sum"], mode="lines+markers",
                                        line=dict(width=2), marker=dict(size=4), name="Last 12 mo"))
            fig_sp.update_layout(height=120, margin=dict(l=10,r=10,t=10,b=10),
                                 xaxis_title=None, yaxis_title=None, showlegend=False)
            st.plotly_chart(fig_sp, use_container_width=True, config={"displayModeBar": False})

    dmin = df_scope["date"].min().date(); dmax = df_scope["date"].max().date()
    try:
        dstart = pd.to_datetime(start_q).date() if start_q else dmin
        dend   = pd.to_datetime(end_q).date() if end_q else dmax
    except Exception:
        dstart, dend = dmin, dmax
    dstart, dend = st.slider("Date range", min_value=dmin, max_value=dmax, value=(dstart, dend), format="YYYY-MM")

    st.query_params.update({
        "iso3": iso3_now,
        "city": "" if city_now in ("", "Country (all)") else city_now,
        "start": dstart.isoformat(),
        "end": dend.isoformat(),
        "page": "2_Precipitation_Dashboard",
    })

    mask = (df_scope["date"]>=pd.to_datetime(dstart)) & (df_scope["date"]<=pd.to_datetime(dend))
    series = df_scope.loc[mask, ["date","sum","var"]].reset_index(drop=True)

    # ===== KPIs =====
    k1, k2, k3, k4 = st.columns(4)
    latest = series["sum"].iloc[-1] if not series.empty else np.nan
    prev   = series["sum"].iloc[-2] if len(series) > 1 else np.nan
    with k1:
        st.metric("Latest Total Precip.", f"{latest:.1f} mm" if np.isfinite(latest) else "—",
                  help="Most recent monthly total precipitation in the selected range/area.")
    with k2:
        dprev = f"{(latest - prev):+0.1f} mm" if np.isfinite(prev) and np.isfinite(latest) else "—"
        st.metric("Δ vs previous point", dprev,
                  help="Change from the immediately previous data point (e.g., previous month).")
    # Δ vs same month last year
    yoy_delta = "—"
    if not series.empty:
        last_date = pd.to_datetime(series["date"].iloc[-1])
        target = (last_date - pd.DateOffset(years=1)).strftime("%Y-%m")
        prev_year_row = series[series["date"].dt.strftime("%Y-%m") == target]
        if not prev_year_row.empty and np.isfinite(latest):
            last_year_val = prev_year_row["sum"].iloc[-1]
            if np.isfinite(last_year_val):
                yoy_delta = f"{(latest - last_year_val):+0.1f} mm"
    with k3:
        st.metric("Δ vs same month LY", yoy_delta,
                  help="Difference relative to the same calendar month in the previous year.")
    with k4:
        mean_v = series["sum"].mean() if not series.empty else np.nan
        std_v  = series["sum"].std(ddof=0) if not series.empty else np.nan
        st.metric("Mean / σ in range",
                  f"{mean_v:.1f} mm • {std_v:.1f}" if np.isfinite(mean_v) and np.isfinite(std_v) else "—",
                  help="Mean and standard deviation over the selected date range.")

    st.download_button(
        "Download data (CSV)",
        data=series.to_csv(index=False).encode("utf-8"),
        file_name=f"{iso3_now}_{('country' if city_now in ('', 'Country (all)') else city_now.replace(' ','_'))}_precipitation.csv",
        mime="text/csv",
        type="primary"
    )

    # ===== Charts (one per row) =====
    def chart_sum_ribbon(title: str, s: pd.DataFrame, value_col="sum", color_key="blue", ylab="mm"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s["date"], y=s[value_col], mode="lines",
                                 name="Total", line=dict(color=CBLIND[color_key], width=2)))
        if "var" in s.columns and s["var"].notna().any():
            sigma = np.sqrt(s["var"].clip(lower=0))
            upper = s[value_col] + sigma
            lower = s[value_col] - sigma
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

    with st.spinner("Rendering time-series charts…"):
        st.plotly_chart(chart_sum_ribbon("Total Precipitation", series, "sum", "blue", "mm"),
                        use_container_width=True)
        st.plotly_chart(chart_sum_ribbon("Precipitation Intensity (placeholder)", series, "sum", "orange", "mm"),
                        use_container_width=True)
        st.plotly_chart(chart_sum_ribbon("Wet Conditions (placeholder)", series, "sum", "green", "mm"),
                        use_container_width=True)
        st.plotly_chart(chart_sum_ribbon("Dry Conditions (placeholder)", series, "sum", "verm", "mm"),
                        use_container_width=True)

    # ===== Percentile selector + charts in a fragment (fast) =====
    def _percentile_block():
        pct = percentile_selector_plotly("opt_pct_p", 50)

        def percentile_chart(title_base: str, s: pd.DataFrame, val_col="sum", color_key="sky", ylab="mm"):
            ps = percentile_series_cached(s["date"], s[val_col], pct)
            if ps.empty:
                st.warning(f"No data for {title_base.lower()}."); return
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ps["date"], y=ps["p"], mode="lines", name=f"P{pct}",
                                     line=dict(color=CBLIND[color_key], width=2)))
            fig.update_layout(
                height=320, margin=dict(l=30,r=30,t=40,b=40),
                title=f"{title_base} — {pct}th Percentile",
                xaxis_title="Date", yaxis_title=ylab,
                xaxis_title_standoff=34, yaxis_title_standoff=34
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.spinner("Updating percentile charts…"):
            percentile_chart("Total Precipitation", series, "sum", "sky", "mm")
            percentile_chart("Wet Conditions (placeholder)", series, "sum", "pink", "mm")
            percentile_chart("Dry Conditions (placeholder)", series, "sum", "navy", "mm")

    if _FRAGMENT_SUPPORTED:
        @fragment
        def _percentile_fragment():
            _percentile_block()
        _percentile_fragment()
    else:
        _percentile_block()
