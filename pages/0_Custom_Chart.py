# pages/0_Custom_Chart.py
import os, math
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from huggingface_hub import hf_hub_download

# Try plotly-resampler (auto-downsampling). Falls back if missing.
try:
    from plotly_resampler import FigureResampler
    _HAS_RESAMPLER = True
except Exception:
    _HAS_RESAMPLER = False

# =================== PAGE CONFIG ===================
st.set_page_config(page_title="Custom Chart Builder", layout="wide", initial_sidebar_state="collapsed")

# =================== STYLES ===================
st.markdown("""
<style>
h1.custom-title{ text-align:center; font-size:2.1rem; margin:0.6rem 0 0.35rem 0; }
.topbar{ margin-top:8px; }
.helper-text{ text-align:center; color:#4a4a4a; margin-bottom:0.8rem; }

/* badges */
.badge-row{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 10px 0; justify-content:center; }
.badge{ background:#f1f3f5; border:1px solid #e1e4e8; border-radius:999px; padding:4px 10px; font-size:.85rem; }

/* unify action buttons to the tag color (#f55551) */
div[data-testid="stFormSubmitButton"] button,
div[data-testid="stDownloadButton"] button,
button[kind="primary"]{
  background-color:#f55551 !important; color:white !important; border:0 !important; font-weight:600;
}

/* give the plot extra breathing room; keep y-titles clear */
</style>
""", unsafe_allow_html=True)

# =================== COLORS ===================
CBLIND = {
    "blue":"#0072B2","orange":"#E69F00","sky":"#56B4E9","green":"#009E73",
    "yellow":"#F0E442","navy":"#332288","verm":"#D55E00","pink":"#CC79A7","grey":"#999999","red":"#d62728"
}
TRACE_PALETTE = [CBLIND[k] for k in ["blue","verm","green","orange","navy","pink","sky","grey","yellow"]]
def _color(i:int)->str: return TRACE_PALETTE[i % len(TRACE_PALETTE)]

# =================== COUNTRY DISPLAY ===================
try:
    import pycountry
except Exception:
    pycountry = None

def iso3_to_name(iso:str)->str:
    iso=(iso or "").upper().strip()
    if pycountry:
        try:
            c=pycountry.countries.get(alpha_3=iso)
            if c and getattr(c,"name",None): return c.name
        except Exception: pass
    return iso

_CUSTOM = {"CHN":"People's Republic of China","TWN":"Taipei,China","HKG":"Hong Kong, China"}
def display_country_name(iso:str)->str: return _CUSTOM.get((iso or "").upper().strip(), iso3_to_name(iso))

# =================== HF HELPERS ===================
def _secret_or_env(k, default=""):
    try:
        if hasattr(st,"secrets") and k in st.secrets: return st.secrets[k]
    except Exception: pass
    return os.getenv(k, default)

HF_REPO_ID   = _secret_or_env("HF_REPO_ID","pjsimba16/adb_climate_dashboard_v1")
HF_REPO_TYPE = _secret_or_env("HF_REPO_TYPE","space")
HF_TOKEN     = _secret_or_env("HF_TOKEN","") or None

@st.cache_data(ttl=24*3600, show_spinner=False)
def _dl(filename:str)->str:
    return hf_hub_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=filename, token=HF_TOKEN)

@st.cache_data(ttl=24*3600, show_spinner=False)
def read_parquet_from_hf(filename:str)->pd.DataFrame:
    return pd.read_parquet(_dl(filename))

# =================== LOAD (FAST: countries only) ===================
with st.spinner("Loading country datasets…"):
    COUNTRY_TEMP  = read_parquet_from_hf("country_temperature.snappy.parquet")
    COUNTRY_PREC  = read_parquet_from_hf("country_precipitation.snappy.parquet")
with st.spinner("Loading ADM1 lookup…"):
    CITY_MAP      = read_parquet_from_hf("city_mapper_with_coords_v2.snappy.parquet")  # small

# Standardize country frames (ADM1 deferred)
def _norm_country_temp(df):
    d=df.rename(columns={"Country":"iso3","Date":"date","Temperature (Mean)":"temp","Temperature (Variance)":"tvar"}).copy()
    d["iso3"]=d["iso3"].astype(str).str.upper().str.strip(); d["date"]=pd.to_datetime(d["date"], errors="coerce")
    d=d.dropna(subset=["iso3","date"]); d["adm1"]=""; d["level"]="country"; return d[["level","iso3","adm1","date","temp","tvar"]]
def _norm_country_prec(df):
    d=df.rename(columns={"Country":"iso3","Date":"date","Precipitation (Sum)":"prcp","Precipitation (Variance)":"pvar"}).copy()
    d["iso3"]=d["iso3"].astype(str).str.upper().str.strip(); d["date"]=pd.to_datetime(d["date"], errors="coerce")
    d=d.dropna(subset=["iso3","date"]); d["adm1"]=""; d["level"]="country"; return d[["level","iso3","adm1","date","prcp","pvar"]]

with st.spinner("Standardizing & caching country data…"):
    COUN_T=_norm_country_temp(COUNTRY_TEMP); COUN_P=_norm_country_prec(COUNTRY_PREC)

ALL_ISOS = sorted(set(COUN_T["iso3"]).union(COUN_P["iso3"]))
ISO_NAME_MAP = {iso:display_country_name(iso) for iso in ALL_ISOS}

ADM1_ALL=(CITY_MAP.rename(columns={"Country":"iso3","City":"adm1"})[["iso3","adm1"]].dropna().astype(str))
ADM1_ALL["iso3"]=ADM1_ALL["iso3"].str.upper().str.strip(); ADM1_ALL["adm1"]=ADM1_ALL["adm1"].str.strip()
ADM1_CHOICES=sorted(ADM1_ALL.drop_duplicates().itertuples(index=False, name=None))

# =================== PRE-AGGREGATION (countries now; ADM1 lazy) ===================
@st.cache_data(ttl=24*3600, show_spinner=False)
def preaggregate(df:pd.DataFrame, value_cols:List[str])->Dict[str,pd.DataFrame]:
    """Return {'M':monthly,'Q':seasonal,'A':annual} pre-aggregates."""
    if df.empty: return {"M":df, "Q":df, "A":df}
    out={}
    for tag,rule in {"M":"M","Q":"Q","A":"A"}.items():
        d=df.copy()
        d["date"]=pd.to_datetime(d["date"], errors="coerce")
        agg={c:("mean" if (c.startswith("temp") or c.endswith("var")) else "sum") for c in value_cols}
        keys=["level","iso3","adm1"]
        out[tag]=(d.set_index("date")
                    .groupby(keys+[pd.Grouper(freq=rule)])[value_cols]
                    .agg(agg).reset_index().rename(columns={"date":"date"}))
    return out

with st.spinner("Pre-aggregating country series by frequency…"):
    PRE_COUN_T = preaggregate(COUN_T, ["temp","tvar"])
    PRE_COUN_P = preaggregate(COUN_P, ["prcp","pvar"])

# Lazy ADM1 loaders (only hit when ADM1s are selected; then cached)
@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_adm1_temp() -> pd.DataFrame:
    CITY_TEMP = read_parquet_from_hf("city_temperature.snappy.parquet")
    d=CITY_TEMP.rename(columns={"Country":"iso3","City":"adm1","Date":"date","Temperature (Mean)":"temp","Temperature (Variance)":"tvar"}).copy()
    d["iso3"]=d["iso3"].astype(str).str.upper().str.strip(); d["adm1"]=d["adm1"].astype(str).str.strip()
    d["date"]=pd.to_datetime(d["date"], errors="coerce"); d=d.dropna(subset=["iso3","adm1","date"]); d["level"]="adm1"
    return d[["level","iso3","adm1","date","temp","tvar"]]

@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_adm1_prec() -> pd.DataFrame:
    CITY_PREC = read_parquet_from_hf("city_precipitation.snappy.parquet")
    d=CITY_PREC.rename(columns={"Country":"iso3","City":"adm1","Date":"date","Precipitation (Sum)":"prcp","Precipitation (Variance)":"pvar"}).copy()
    d["iso3"]=d["iso3"].astype(str).str.upper().str.strip(); d["adm1"]=d["adm1"].astype(str).str.strip()
    d["date"]=pd.to_datetime(d["date"], errors="coerce"); d=d.dropna(subset=["iso3","adm1","date"]); d["level"]="adm1"
    return d[["level","iso3","adm1","date","prcp","pvar"]]

@st.cache_data(ttl=24*3600, show_spinner=False)
def _preagg_adm1_temp() -> Dict[str,pd.DataFrame]:
    return preaggregate(_load_adm1_temp(), ["temp","tvar"])

@st.cache_data(ttl=24*3600, show_spinner=False)
def _preagg_adm1_prec() -> Dict[str,pd.DataFrame]:
    return preaggregate(_load_adm1_prec(), ["prcp","pvar"])

# =================== HELPERS ===================
def _label_geo(level, iso3, adm1):
    base=display_country_name(iso3); return f"{base} — {adm1}" if level=="adm1" and adm1 else base
def _indicator_to_col(name:str): return ("temp","Temperature (°C)") if name.startswith("Temp") else ("prcp","Precipitation (mm)")
def _compose_title(indicators:List[str], countries:List[str], adm1_pairs:List[Tuple[str,str]])->str:
    inds=" & ".join(indicators) if indicators else ""
    parts=[]
    if countries: parts.append(", ".join([display_country_name(i) for i in countries]))
    if adm1_pairs:
        df=pd.DataFrame(adm1_pairs, columns=["iso3","adm1"])
        chunks=[f"{display_country_name(iso)} ({', '.join(g['adm1'].tolist())})" for iso,g in df.groupby("iso3")]
        parts.append("ADM1: " + "; ".join(chunks))
    bits=[b for b in [inds]+parts if b]; return " — ".join(bits)

def _make_subplots(rows=1, cols=1, titles=None):
    specs=[[{"secondary_y":True}]*cols for _ in range(rows)]
    return make_subplots(rows=rows, cols=cols, shared_xaxes=True, vertical_spacing=0.12, specs=specs, subplot_titles=titles)

def _apply_layout(fig, title_text:str, x_title:str, y_title:Optional[str], show_title:bool, is_bar:bool, stacked:bool, dual_units:bool):
    if not show_title: title_text=""
    fig.update_layout(
        height=780,
        margin=dict(l=48,r=48,t=64 if show_title else 24,b=80),
        title=title_text,
        legend=dict(orientation="h", y=1.04, x=0, title=dict(text="Units: °C & mm" if dual_units else "")),
        hovermode="x unified" if not is_bar else "x",
        barmode=("stack" if stacked else "group") if is_bar else None,
        template=None,
    )
    fig.update_xaxes(title_text=x_title, title_standoff=36, automargin=True)
    if y_title:
        fig.update_yaxes(title_text=y_title, title_standoff=86, automargin=True, secondary_y=False)
    fig.update_yaxes(title_standoff=86, automargin=True, secondary_y=True)

def _modebar_cfg():
    return {"displaylogo":False, "toImageButtonOptions":{"format":"png","scale":2}, "modeBarButtonsToAdd":["toImage"]}

# =================== HEADER ===================
top = st.container()
with top:
    c1,c2,c3 = st.columns([0.12,0.76,0.12], gap="small")
    with c1:
        st.markdown('<div class="topbar"></div>', unsafe_allow_html=True)
        if st.button("← Home"):
            st.query_params.clear()
            try: st.switch_page("Home_Page.py")
            except Exception: st.rerun()
    with c2:
        st.markdown("<h1 class='custom-title'>Custom Chart Builder</h1>", unsafe_allow_html=True)
        st.markdown(
            "<div class='helper-text'>Build and compare custom climate charts across countries and ADM1s. "
            "Choose indicators, chart types (including anomalies), frequency, date windows, and facets; "
            "apply smoothing, normalization, and climatology overlays; then export the figure and data. "
            "<br><em>Tip:</em> Use the Quick Select to start with the most recent years, and add ADM1 only when needed for faster loading.</div>",
            unsafe_allow_html=True
        )

# Defaults
if "chart_type" not in st.session_state: st.session_state["chart_type"]="Line"

# =================== CHART TYPE ===================
chart_type = st.segmented_control(
    "Chart type",
    options=["Line","Area","Bar","Scatter","Anomaly"],
    selection_mode="single",
    key="chart_type",
    help="Choose how to visualize: Anomaly shows deviation from a baseline climatology; Scatter compares two indicators."
)

# =================== OPTIONS (Collapsible Form) ===================
with st.expander("Options", expanded=True):
    with st.form("controls"):
        st.markdown("#### Selections")

        # Geographies
        iso_opts = sorted(ALL_ISOS, key=lambda x: ISO_NAME_MAP.get(x,x))
        sel_countries = st.multiselect(
            "Countries (optional)", options=iso_opts, format_func=lambda x: ISO_NAME_MAP.get(x,x),
            help="Add country-level series. You can also add ADM1 series below and compare across both."
        )
        def _adm1label(pair): return f"{display_country_name(pair[0])} — {pair[1]}"
        sel_adm1 = st.multiselect(
            "ADM1 (optional)", options=ADM1_CHOICES, format_func=_adm1label,
            help="Add province/state series from any country. Works alone or with country-level series."
        )

        # Date quick selector (default last 30y)
        st.markdown("**Date range**")
        quick = st.segmented_control(
            "Quick select", options=["10y","20y","30y","50y","All"],
            selection_mode="single", key="quick_sel",
            help="Select a preset window. These refer to the most recent N years available."
        )

        # Compute min/max from the chosen geos (using country tables; ADM1 is lazy)
        def _minmax():
            t_c = COUN_T[COUN_T["iso3"].isin(sel_countries)] if sel_countries else COUN_T.iloc[0:0]
            p_c = COUN_P[COUN_P["iso3"].isin(sel_countries)] if sel_countries else COUN_P.iloc[0:0]
            dmin=pd.concat([t_c["date"],p_c["date"]],ignore_index=True).min()
            dmax=pd.concat([t_c["date"],p_c["date"]],ignore_index=True).max()
            return (pd.to_datetime(dmin), pd.to_datetime(dmax))
        dmin, dmax = _minmax()
        if pd.isna(dmin) or pd.isna(dmax):
            st.info("Select at least one Country or ADM1 to enable date range and chart.", icon="ℹ️")
            dstart=dend=None
        else:
            q_years = {"10y":10,"20y":20,"30y":30,"50y":50}.get(quick, None)
            default_start = (dmax - pd.DateOffset(years=q_years)).date() if q_years else dmin.date()
            default_start = max(default_start, dmin.date())
            dstart, dend = st.slider(
                "Custom range", min_value=dmin.date(), max_value=dmax.date(),
                value=(default_start, dmax.date()), format="YYYY-MM",
                help="Drag the handles to refine the window."
            )

        st.markdown("---")
        freq = st.radio(
            "Frequency", ["Monthly","Seasonal","Annual"], index=0, horizontal=True,
            help="Aggregate raw data to the selected frequency: Monthly (M), Seasonal/Quarterly (Q), or Annual (A)."
        )
        facet_by = st.selectbox(
            "Facet", ["None","Geography (Country/ADM1)","Indicator"], index=0,
            help="Create small multiples either by geography (up to 6 panels) or by indicator."
        )

        # Indicators & type-specific advanced options
        stack_by_adm1 = False
        show_points = st.checkbox("Show markers", value=False, help="Overlay point markers on lines.")
        hide_title  = st.checkbox("Hide chart title", value=False, help="Turn off the chart title.")

        global_window = 0
        enable_per_trace = False
        normalize_ix = False
        clim_overlay = False

        if chart_type in ("Line","Area","Bar"):
            indicators = st.multiselect(
                "Indicator(s)", ["Temperature (°C)","Precipitation (mm)"], default=["Temperature (°C)"],
                help="Select one or both. If both, precipitation is placed on a secondary Y (unless you normalize)."
            )
            if chart_type=="Bar":
                stack_by_adm1 = st.checkbox(
                    "Stack bars by ADM1", value=False,
                    help="Best when comparing several ADM1 within a single country."
                )
            with st.expander("Advanced (time-series)"):
                global_window = st.number_input(
                    "Smoothing window (points, 0 = off)", min_value=0, value=0, step=1,
                    help="Apply a centered rolling mean. Per-trace overrides are available below."
                )
                enable_per_trace = st.checkbox("Custom smoothing per trace", help="Edit smoothing per series after applying.")
                normalize_ix = st.checkbox(
                    "Normalize to index=100 at start date (disables dual axes)",
                    help="Rebase each series to 100 at the first visible date; useful for relative growth."
                )
                clim_overlay = st.checkbox(
                    "Overlay seasonal climatology (monthly mean by calendar month)",
                    help="Adds a dotted line showing the long-run typical seasonal cycle."
                )

        elif chart_type=="Scatter":
            x_series = st.selectbox("X-axis series", ["Temperature (°C)","Precipitation (mm)"], index=0, help="Horizontal values.")
            y_series = st.selectbox("Y-axis series", ["Temperature (°C)","Precipitation (mm)"], index=1, help="Vertical values.")
            with st.expander("Advanced (scatter)"):
                global_window = st.number_input(
                    "Smoothing window for connecting lines (0 = off)", min_value=0, value=0, step=1,
                    help="If >0, draw faint connecting lines smoothed by this window."
                )

        else:  # ===== Anomaly options (updated) =====
            an_indicator = st.selectbox(
                "Indicator", ["Temperature (°C)", "Precipitation (mm)"], index=0,
                help="Which variable to compute anomalies for."
            )

            # Baseline year bounds based on FULL available data for the selection (not just visible range)
            @st.cache_data(ttl=1800, show_spinner=False)
            def _baseline_bounds(freq: str, countries: Tuple[str, ...], adm1s: Tuple[Tuple[str, str], ...]) -> Tuple[int, int]:
                tag = {"Monthly":"M","Seasonal":"Q","Annual":"A"}[freq]
                c_frames = []
                if countries:
                    if an_indicator.startswith("Temp"):
                        c_frames.append(PRE_COUN_T[tag][PRE_COUN_T[tag]["iso3"].isin(countries)][["date"]])
                    else:
                        c_frames.append(PRE_COUN_P[tag][PRE_COUN_P[tag]["iso3"].isin(countries)][["date"]])
                if adm1s:
                    sel_df = pd.DataFrame(list(adm1s), columns=["iso3","adm1"])
                    if an_indicator.startswith("Temp"):
                        PRE_ADM1_T = _preagg_adm1_temp()
                        a = PRE_ADM1_T[tag].merge(sel_df, on=["iso3","adm1"], how="inner")[["date"]]
                        c_frames.append(a)
                    else:
                        PRE_ADM1_P = _preagg_adm1_prec()
                        a = PRE_ADM1_P[tag].merge(sel_df, on=["iso3","adm1"], how="inner")[["date"]]
                        c_frames.append(a)
                if not c_frames:
                    src = PRE_COUN_T[tag] if an_indicator.startswith("Temp") else PRE_COUN_P[tag]
                    c_frames = [src[["date"]]]
                all_dates = pd.concat(c_frames, ignore_index=True) if c_frames else pd.DataFrame({"date":[]})
                if all_dates.empty:
                    return (1991, 2020)
                return (all_dates["date"].dt.year.min().item(), all_dates["date"].dt.year.max().item())

            bmin, bmax = _baseline_bounds(freq, tuple(sel_countries), tuple(sel_adm1))
            base_start, base_end = st.slider(
                "Baseline years",
                min_value=int(bmin), max_value=int(bmax),
                value=(max(int(bmin), 1991), min(int(bmax), 2020)),
                help="The baseline is computed from the full dataset (not limited to the visible date range)."
            )

            baseline_calc = st.selectbox(
                "Baseline calculation method",
                ["Mean", "Median", "Min", "Max"],
                help="How to summarize the baseline window per month/quarter/year for each geography."
            )

            an_method  = st.selectbox(
                "Anomaly calculation method",
                ["Absolute (value - baseline)", "Percent of baseline", "Z-score (std from baseline)"],
                help="Absolute departures, percent departures, or standardized anomalies."
            )

            with st.expander("Advanced (anomalies)"):
                clim_overlay = st.checkbox("Overlay baseline (=0) line", value=True, help="Draws a dotted zero line.")
                show_points   = st.checkbox("Show markers (anomaly)", value=False, help="Overlay point markers.")

        submitted = st.form_submit_button("Apply changes", use_container_width=True)

if not submitted or (dstart is None) or (dend is None) or (not sel_countries and not sel_adm1):
    st.stop()

# =================== DATA (fast path using pre-agg; ADM1 lazy) ===================
@st.cache_data(ttl=3600, show_spinner=False)
def compute_wide_fast(countries:Tuple[str,...], adm1s:Tuple[Tuple[str,str],...], d0:str, d1:str, freq:str) -> pd.DataFrame:
    tag = {"Monthly":"M","Seasonal":"Q","Annual":"A"}[freq]
    # Country blocks
    t_c = PRE_COUN_T[tag][PRE_COUN_T[tag]["iso3"].isin(countries)] if countries else PRE_COUN_T[tag].iloc[0:0]
    p_c = PRE_COUN_P[tag][PRE_COUN_P[tag]["iso3"].isin(countries)] if countries else PRE_COUN_P[tag].iloc[0:0]
    # ADM1 blocks (loaded lazily only if needed)
    if adm1s:
        PRE_ADM1_T = _preagg_adm1_temp()
        PRE_ADM1_P = _preagg_adm1_prec()
        sel_df = pd.DataFrame(list(adm1s), columns=["iso3","adm1"])
        t_a = PRE_ADM1_T[tag].merge(sel_df, on=["iso3","adm1"], how="inner")
        p_a = PRE_ADM1_P[tag].merge(sel_df, on=["iso3","adm1"], how="inner")
    else:
        t_a = pd.DataFrame(columns=["level","iso3","adm1","date","temp"])
        p_a = pd.DataFrame(columns=["level","iso3","adm1","date","prcp"])

    t_df = pd.concat([t_c,t_a], ignore_index=True)
    p_df = pd.concat([p_c,p_a], ignore_index=True)
    mask_t=(t_df["date"]>=pd.to_datetime(d0)) & (t_df["date"]<=pd.to_datetime(d1))
    mask_p=(p_df["date"]>=pd.to_datetime(d0)) & (p_df["date"]<=pd.to_datetime(d1))
    t_df=t_df.loc[mask_t].copy(); p_df=p_df.loc[mask_p].copy()
    wide=(t_df[["level","iso3","adm1","date","temp"]]
            .merge(p_df[["level","iso3","adm1","date","prcp"]],
                   on=["level","iso3","adm1","date"], how="outer")
            .sort_values(["level","iso3","adm1","date"]))
    return wide

with st.spinner("Preparing chart data…"):
    wide = compute_wide_fast(tuple(sel_countries), tuple(sel_adm1), str(dstart), str(dend), freq)
if wide.empty:
    st.warning("No data for the current selection / date range."); st.stop()

# =================== SMOOTHING / NORMALIZATION ===================
def _apply_smooth(s:pd.Series, win:int)->pd.Series:
    return s.rolling(win, min_periods=1, center=True).mean() if (win and win>0) else s

geos = wide[["level","iso3","adm1"]].drop_duplicates().reset_index(drop=True)
trace_names=[]
if chart_type!="Scatter" and chart_type!="Anomaly":
    base_inds = (locals().get("indicators") or ["Temperature (°C)"])
    for _,g in geos.iterrows():
        nm=_label_geo(g.level,g.iso3,g.adm1)
        if len(base_inds)==1: trace_names.append(nm)
        else:
            for ind in base_inds: trace_names.append(f"{nm} — {ind.split()[0]}")
elif chart_type=="Scatter":
    trace_names=[_label_geo(r.level,r.iso3,r.adm1) for _,r in geos.iterrows()]

per_trace_windows: Dict[str,int] = {}
if (locals().get("enable_per_trace") and trace_names):
    cfg_df = pd.DataFrame({"Trace":trace_names, "Window (0 = off)":[locals().get("global_window",0)]*len(trace_names)})
    cfg = st.data_editor(cfg_df, num_rows="fixed", use_container_width=True, key="pertrace_cfg")
    per_trace_windows = {row["Trace"]: int(row["Window (0 = off)"] or 0) for _,row in cfg.iterrows()}

if chart_type in ("Line","Area","Bar"):
    wide = wide.sort_values(["level","iso3","adm1","date"]).copy()
    gw = int(locals().get("global_window",0))
    chosen = (locals().get("indicators") or ["Temperature (°C)"])
    if (locals().get("enable_per_trace") and per_trace_windows) or (gw>0):
        out=[]
        for (_, g) in wide.groupby(["level","iso3","adm1"]):
            gg=g.copy(); base=_label_geo(gg.iloc[0]["level"], gg.iloc[0]["iso3"], gg.iloc[0]["adm1"])
            if "Temperature (°C)" in chosen or len(chosen)==1:
                nm = base if len(chosen)==1 and chosen[0].startswith("Temp") else f"{base} — Temperature"
                win = per_trace_windows.get(nm, gw)
                if "temp" in gg: gg["temp"]=_apply_smooth(gg["temp"], win)
            if "Precipitation (mm)" in chosen or (len(chosen)==1 and chosen[0].startswith("Precip")):
                nm = base if len(chosen)==1 and chosen[0].startswith("Precip") else f"{base} — Precipitation"
                win = per_trace_windows.get(nm, gw)
                if "prcp" in gg: gg["prcp"]=_apply_smooth(gg["prcp"], win)
            out.append(gg)
        wide=pd.concat(out, ignore_index=True)
    if locals().get("normalize_ix"):
        def _norm(g:pd.DataFrame):
            for col in ["temp","prcp"]:
                if col in g and not g[col].dropna().empty:
                    base=g[col].iloc[0]
                    if pd.notna(base) and base!=0: g[col]=100*g[col]/base
            return g
        wide = wide.groupby(["level","iso3","adm1"], as_index=False).apply(_norm).reset_index(drop=True)

# =================== ANOMALY CALC HELPERS ===================
def _phase_key_from_freq(tag: str, s: pd.Series) -> pd.Series:
    """Return the grouping 'phase' for baseline stats given frequency tag."""
    if tag == "M":   return s.dt.month           # 1..12
    if tag == "Q":   return s.dt.quarter         # 1..4
    return pd.Series(1, index=s.index)           # Annual: single bucket

def _baseline_agg(df: pd.DataFrame, ycol: str, tag: str, method: str) -> pd.DataFrame:
    """Aggregate baseline stats per-geo per-phase."""
    phase = _phase_key_from_freq(tag, df["date"])
    df = df.assign(_phase=phase)
    agg_map = {"Mean":"mean", "Median":"median", "Min":"min", "Max":"max"}
    op = agg_map.get(method, "mean")
    bl = (df.groupby(["level","iso3","adm1","_phase"])[ycol]
            .agg(baseline=op, mean="mean", std="std")
            .reset_index())
    return bl

@st.cache_data(ttl=1800, show_spinner=False)
def _full_selection_for_baseline(tag: str, countries: Tuple[str,...], adm1s: Tuple[Tuple[str,str],...], indicator: str) -> pd.DataFrame:
    """Return full-span pre-aggregated series for the selected geos (no date filter), only the chosen indicator."""
    if indicator.startswith("Temp"):
        src_c = PRE_COUN_T[tag]
        src_a = _preagg_adm1_temp()[tag] if adm1s else None
        ycol = "temp"
    else:
        src_c = PRE_COUN_P[tag]
        src_a = _preagg_adm1_prec()[tag] if adm1s else None
        ycol = "prcp"
    frames = []
    if countries:
        frames.append(src_c[src_c["iso3"].isin(countries)][["level","iso3","adm1","date",ycol]])
    if adm1s and src_a is not None:
        sel_df = pd.DataFrame(list(adm1s), columns=["iso3","adm1"])
        frames.append(src_a.merge(sel_df, on=["iso3","adm1"], how="inner")[["level","iso3","adm1","date",ycol]])
    if not frames:
        frames.append(src_c[["level","iso3","adm1","date",ycol]])
    return pd.concat(frames, ignore_index=True)

def _compute_anomaly(
        wide_visible: pd.DataFrame,  # currently filtered to dstart..dend
        countries: List[str], adm1s: List[Tuple[str,str]],
        freq: str, indicator: str,
        base_start: int, base_end: int,
        baseline_calc: str, anomaly_calc: str
    ) -> pd.DataFrame:
    """Compute anomalies on the visible window using a baseline derived from FULL data within baseline years."""
    tag = {"Monthly":"M","Seasonal":"Q","Annual":"A"}[freq]
    ycol, _ = _indicator_to_col(indicator)

    # Full selection for baseline window (no visible-range filter)
    full = _full_selection_for_baseline(tag, tuple(countries), tuple(adm1s), indicator)
    if full.empty:
        return pd.DataFrame()

    # Clip to baseline years and build baseline stats per geo & phase
    base = full[(full["date"].dt.year >= base_start) & (full["date"].dt.year <= base_end)].copy()
    if base.empty:
        return pd.DataFrame()
    bl = _baseline_agg(base, ycol, tag, baseline_calc)

    # Now compute anomaly on the visible frame by joining baseline stats by geo & phase
    vis = wide_visible.dropna(subset=[ycol]).copy()
    vis["_phase"] = _phase_key_from_freq(tag, vis["date"])

    out = vis.merge(bl, on=["level","iso3","adm1","_phase"], how="left")
    if out.empty:
        return pd.DataFrame()

    if anomaly_calc.startswith("Absolute"):
        out["anom"] = out[ycol] - out["baseline"]; unit = "Δ" + ("°C" if ycol=="temp" else " mm")
    elif anomaly_calc.startswith("Percent"):
        out["anom"] = np.where(out["baseline"].abs()>0, 100*(out[ycol]-out["baseline"])/out["baseline"], np.nan); unit="%"
    else:  # Z-score
        out["anom"] = (out[ycol] - out["mean"]) / out["std"]
        unit = "σ"

    return out[["level","iso3","adm1","date","anom"]].assign(unit=unit)

# =================== FIGURE HELPERS ===================
def _add_ts(fig, df_geo, row, col, name, ycol, secondary=False, markers=False, is_bar=False, idx=0):
    if is_bar:
        tr = go.Bar(x=df_geo["date"], y=df_geo[ycol], name=name)
        fig.add_trace(tr, row=row, col=col, secondary_y=secondary)
        fig.data[-1].update(marker_color=_color(idx))  # Bars: color marker only
    else:
        mode = "lines+markers" if markers else "lines"
        tr = go.Scatter(x=df_geo["date"], y=df_geo[ycol], mode=mode, name=name, line=dict(width=2))
        fig.add_trace(tr, row=row, col=col, secondary_y=secondary)
        fig.data[-1].update(line=dict(color=_color(idx), width=2), marker_color=_color(idx))

def _add_xy(fig, df_geo, row, col, name, xcol, ycol, markers=False, idx=0):
    tr = go.Scatter(x=df_geo[xcol], y=df_geo[ycol],
                    mode="markers+lines" if markers else "markers",
                    text=df_geo["date"].dt.strftime("%Y-%m"),
                    hovertemplate="%{text}<br>X=%{x:.2f}, Y=%{y:.2f}<extra></extra>",
                    name=name)
    fig.add_trace(tr, row=row, col=col, secondary_y=False)
    fig.data[-1].update(marker_color=_color(idx), line=dict(color=_color(idx)))

def _new_fig(rows=1, cols=1, titles=None):
    base = _make_subplots(rows, cols, titles)
    if _HAS_RESAMPLER and st.session_state.get("chart_type") in ("Line","Area"):
        try:
            return FigureResampler(base, default_n_shown_samples=2500)
        except Exception:
            return base
    return base

# =================== TITLE ===================
inds_list = (
    [locals().get("x_series"), locals().get("y_series")] if chart_type=="Scatter"
    else ([locals().get("an_indicator")] if chart_type=="Anomaly" else (locals().get("indicators") or []))
)
title_txt = _compose_title(inds_list, sel_countries, sel_adm1)

# =================== BUILD CHART ===================
geos = wide[["level","iso3","adm1"]].drop_duplicates().reset_index(drop=True)
is_bar = (chart_type=="Bar")
stacked = (is_bar and (locals().get("stack_by_adm1") or False))

if facet_by=="None":
    fig = _new_fig(1,1); r=c=1

    if chart_type=="Scatter":
        xcol,_=_indicator_to_col(x_series); ycol,_=_indicator_to_col(y_series); k=0
        for _,g in geos.iterrows():
            df=wide[(wide["level"]==g["level"]) & (wide["iso3"]==g["iso3"]) & (wide["adm1"]==g["adm1"])].dropna(subset=[xcol,ycol])
            if df.empty: continue
            _add_xy(fig, df, r, c, _label_geo(g["level"],g["iso3"],g["adm1"]), xcol, ycol, markers=show_points, idx=k); k+=1
        _apply_layout(fig, title_txt, x_series, y_series, not hide_title, False, False, False)

    elif chart_type=="Anomaly":
        with st.spinner("Computing anomalies…"):
            an = _compute_anomaly(
                wide, sel_countries, sel_adm1, freq, an_indicator,
                int(base_start), int(base_end), baseline_calc, an_method
            )
        if an.empty:
            st.warning("No data for anomaly calculation with the selected baseline. Try expanding the baseline years or picking other geographies.", icon="⚠️")
            st.stop()
        ylab = f"Anomaly ({an['unit'].dropna().iloc[0]})"
        k=0
        for _,g in an[["level","iso3","adm1"]].drop_duplicates().iterrows():
            df=an[(an["level"]==g["level"]) & (an["iso3"]==g["iso3"]) & (an["adm1"]==g["adm1"])].copy()
            name=_label_geo(g["level"],g["iso3"],g["adm1"])
            tr = go.Scatter(x=df["date"], y=df["anom"], mode="lines+markers" if show_points else "lines",
                            name=name, line=dict(width=2, color=_color(k)))
            fig.add_trace(tr, row=r, col=c); k+=1
        if clim_overlay: fig.add_hline(y=0, line=dict(color="#999999", dash="dot"))
        _apply_layout(fig, title_txt, "Date", ylab, not hide_title, False, False, False)

    else:
        chosen = (locals().get("indicators") or ["Temperature (°C)"])
        auto_dual = (len(chosen)==2) and not (locals().get("normalize_ix"))
        k=0
        for _,g in geos.iterrows():
            df=wide[(wide["level"]==g["level"]) & (wide["iso3"]==g["iso3"]) & (wide["adm1"]==g["adm1"])].copy()
            base=_label_geo(g["level"],g["iso3"],g["adm1"])
            for ind in chosen:
                ycol,_=_indicator_to_col(ind)
                if ycol not in df or df[ycol].dropna().empty: continue
                secondary = (auto_dual and ind=="Precipitation (mm)")
                nm = base if len(chosen)==1 else f"{base} — {ind.split()[0]}"
                if _HAS_RESAMPLER and isinstance(fig, go.Figure) is False and st.session_state.get("chart_type") in ("Line","Area"):
                    fig.add_trace(go.Scatter(name=nm, mode="lines+markers" if show_points else "lines"),
                                  hf_x=df["date"], hf_y=df[ycol], row=r, col=c, secondary_y=secondary)  # type: ignore[attr-defined]
                    fig.data[-1].update(line=dict(color=_color(k), width=2), marker_color=_color(k))
                else:
                    _add_ts(fig, df, r, c, nm, ycol, secondary, show_points, is_bar=is_bar, idx=k)
                k+=1
        if locals().get("normalize_ix"):
            _apply_layout(fig, title_txt, "Date", "Index (start=100)", not hide_title, is_bar, stacked, False)
        elif len(chosen)==1:
            _apply_layout(fig, title_txt, "Date", chosen[0], not hide_title, is_bar, stacked, False)
        else:
            _apply_layout(fig, title_txt, "Date", "Temperature (°C)", not hide_title, is_bar, stacked, True)
            fig.update_yaxes(title_text="Precipitation (mm)", secondary_y=True, title_standoff=86, automargin=True)

elif facet_by=="Indicator":
    if chart_type in ("Scatter","Anomaly"):
        st.info("Indicator faceting is not supported for this chart type.", icon="ℹ️"); st.stop()
    chosen = (locals().get("indicators") or ["Temperature (°C)"])
    cols=min(2,len(chosen)); rows=math.ceil(len(chosen)/2)
    fig = _new_fig(rows, cols, titles=chosen); k=0
    for idx,ind in enumerate(chosen):
        rr=idx//2+1; cc=idx%2+1; ycol,_=_indicator_to_col(ind)
        for _,g in geos.iterrows():
            df=wide[(wide["level"]==g["level"]) & (wide["iso3"]==g["iso3"]) & (wide["adm1"]==g["adm1"])].copy()
            if ycol not in df or df[ycol].dropna().empty: continue
            nm=_label_geo(g["level"],g["iso3"],g["adm1"])
            if _HAS_RESAMPLER and isinstance(fig, go.Figure) is False and st.session_state.get("chart_type") in ("Line","Area"):
                fig.add_trace(go.Scatter(name=nm, mode="lines+markers" if show_points else "lines"),
                              hf_x=df["date"], hf_y=df[ycol], row=rr, col=cc, secondary_y=False)  # type: ignore[attr-defined]
                fig.data[-1].update(line=dict(color=_color(k), width=2), marker_color=_color(k))
            else:
                _add_ts(fig, df, rr, cc, nm, ycol, secondary=False, markers=show_points, is_bar=is_bar, idx=k)
            k+=1
    _apply_layout(fig, title_txt, "Date", None, not hide_title, is_bar, stacked, False)

else:  # Geography facets
    maxf=6
    if len(geos)>maxf:
        st.warning(f"Too many geographies selected for faceting ({len(geos)}). Showing the first {maxf}.", icon="⚠️")
        geos=geos.head(maxf)
    titles=[_label_geo(r.level,r.iso3,r.adm1) for _,r in geos.iterrows()]
    cols=3 if len(geos)>=3 else len(geos); rows=math.ceil(len(geos)/cols)
    fig = _new_fig(rows, cols, titles=titles); k=0

    if chart_type=="Scatter":
        xcol,_=_indicator_to_col(x_series); ycol,_=_indicator_to_col(y_series)
        for idx,(_,g) in enumerate(geos.iterrows()):
            rr=idx//cols+1; cc=idx%cols+1
            df=wide[(wide["level"]==g["level"]) & (wide["iso3"]==g["iso3"]) & (wide["adm1"]==g["adm1"])].dropna(subset=[xcol,ycol])
            if df.empty: continue
            _add_xy(fig, df, rr, cc, "", xcol, ycol, markers=show_points, idx=k); k+=1
        _apply_layout(fig, title_txt, x_series, y_series, not hide_title, False, False, False)

    elif chart_type=="Anomaly":
        with st.spinner("Computing anomalies…"):
            an = _compute_anomaly(
                wide, sel_countries, sel_adm1, freq, an_indicator,
                int(base_start), int(base_end), baseline_calc, an_method
            )
        if an.empty:
            st.warning("No data for anomaly calculation with the selected baseline. Try expanding the baseline years or picking other geographies.", icon="⚠️")
            st.stop()
        ylab = f"Anomaly ({an['unit'].dropna().iloc[0]})"
        for idx,(_,g) in enumerate(geos.iterrows()):
            rr=idx//cols+1; cc=idx%cols+1
            df=an[(an["level"]==g["level"]) & (an["iso3"]==g["iso3"]) & (an["adm1"]==g["adm1"])].copy()
            if df.empty: continue
            tr = go.Scatter(x=df["date"], y=df["anom"], mode="lines+markers" if show_points else "lines",
                            name="", line=dict(width=2, color=_color(idx)))
            fig.add_trace(tr, row=rr, col=cc)
        if clim_overlay: fig.add_hline(y=0, line=dict(color="#999999", dash="dot"))
        _apply_layout(fig, title_txt, "Date", ylab, not hide_title, False, False, False)

    else:
        chosen = (locals().get("indicators") or ["Temperature (°C)"])
        auto_dual = (len(chosen)==2) and not (locals().get("normalize_ix"))
        for idx,(_,g) in enumerate(geos.iterrows()):
            rr=idx//cols+1; cc=idx%cols+1
            df=wide[(wide["level"]==g["level"]) & (wide["iso3"]==g["iso3"]) & (wide["adm1"]==g["adm1"])].copy()
            for ind in chosen:
                ycol,_=_indicator_to_col(ind)
                if ycol not in df or df[ycol].dropna().empty: continue
                if _HAS_RESAMPLER and isinstance(fig, go.Figure) is False and st.session_state.get("chart_type") in ("Line","Area"):
                    fig.add_trace(go.Scatter(name=ind.split()[0], mode="lines+markers" if show_points else "lines"),
                                  hf_x=df["date"], hf_y=df[ycol], row=rr, col=cc,
                                  secondary_y=(auto_dual and ind=="Precipitation (mm)"))  # type: ignore[attr-defined]
                    fig.data[-1].update(line=dict(color=_color(idx), width=2), marker_color=_color(idx))
                else:
                    _add_ts(fig, df, rr, cc, ind.split()[0], ycol, secondary=(auto_dual and ind=="Precipitation (mm)"),
                            markers=show_points, is_bar=is_bar, idx=idx)
        if locals().get("normalize_ix"):
            _apply_layout(fig, title_txt, "Date", "Index (start=100)", not hide_title, is_bar, stacked, False)
        elif len(chosen)==1:
            _apply_layout(fig, title_txt, "Date", None, not hide_title, is_bar, stacked, False)
        else:
            _apply_layout(fig, title_txt, "Date", None, not hide_title, is_bar, stacked, True)

# Area fill (only Area charts)
if chart_type=="Area":
    for tr in fig.select_traces(type="scatter"):
        tr.update(fill="tozeroy", hoverinfo="x+y+name")

# Avoid y-title/tick clashes with generous standoff
fig.update_yaxes(title_standoff=86, automargin=True, secondary_y=False)
fig.update_yaxes(title_standoff=86, automargin=True, secondary_y=True)

# =================== BADGES ===================
badges=[]
badges.append(f"<span class='badge'>Type: {chart_type}</span>")
badges.append(f"<span class='badge'>Freq: {freq}</span>")
# Quick-select helper badge
if quick != "All":
    qs_text = f"Last {quick.replace('y',' years (most recent)')}"
else:
    qs_text = "All years"
badges.append(f"<span class='badge'>Window: {qs_text}</span>")
badges.append(f"<span class='badge'>Date: {dstart.isoformat()} → {dend.isoformat()}</span>")
if chart_type not in ("Scatter","Anomaly"):
    if locals().get("indicators"):
        badges.append(f"<span class='badge'>Indicators: {' + '.join(indicators)}</span>")
        if len(indicators)==2 and not (locals().get("normalize_ix")):
            badges.append("<span class='badge'>Secondary Y: Auto</span>")
elif chart_type=="Scatter":
    badges.extend([f"<span class='badge'>X: {x_series}</span>", f"<span class='badge'>Y: {y_series}</span>"])
else:
    badges.append(f"<span class='badge'>Baseline: {int(base_start)}-{int(base_end)}</span>")
    badges.append(f"<span class='badge'>Baseline calc: {baseline_calc}</span>")
    badges.append(f"<span class='badge'>Method: {an_method.split('(')[0].strip()}</span>")
if facet_by!="None": badges.append(f"<span class='badge'>Facet: {facet_by}</span>")
if locals().get("show_points"): badges.append("<span class='badge'>Markers: On</span>")
if locals().get("global_window",0)>0: badges.append(f"<span class='badge'>Smooth: {int(global_window)}</span>")
if locals().get("clim_overlay"): badges.append("<span class='badge'>Climatology: On</span>")
st.markdown(f"<div class='badge-row'>{''.join(badges)}</div>", unsafe_allow_html=True)
# Ensure all y-axis titles stay vertical with ample spacing (all subplots & both sides)
fig.for_each_yaxis(lambda a: a.update(title_standoff=90, automargin=True, ticks="outside", ticklabelposition="outside"))

# =================== RENDER ===================
st.plotly_chart(fig, use_container_width=True, config=_modebar_cfg())

# =================== EXPORT ===================
st.markdown("#### Export")
def _clean_export(df:pd.DataFrame)->pd.DataFrame:
    cols=["level","iso3","adm1","date","temp","prcp"]; keep=[c for c in cols if c in df.columns]
    return df[keep].sort_values(["level","iso3","adm1","date"])
st.download_button("Download data (CSV)", data=_clean_export(wide).to_csv(index=False).encode("utf-8"),
                   file_name="custom_chart_data.csv", mime="text/csv")
try:
    import plotly.io as pio
    base_export = fig if isinstance(fig, go.Figure) else getattr(fig, "figure", fig)
    st.download_button("Download chart image (PNG)", data=pio.to_image(base_export, format="png", scale=2),
                       file_name="custom_chart.png", mime="image/png")
except Exception:
    pass
