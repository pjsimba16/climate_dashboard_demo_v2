# Home_Page.py
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px  # for continent lookup & hist
import streamlit.components.v1 as components
from huggingface_hub import hf_hub_download

# Optional: fast click events (falls back gracefully)
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

# Prefer the user's local timezone (Asia/Manila) for the "Last Update" badge
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    LOCAL_TZ = ZoneInfo("Asia/Manila")
except Exception:
    LOCAL_TZ = None

# Optional: pycountry for country names if available
try:
    import pycountry
except Exception:
    pycountry = None

st.set_page_config(
    page_title="Home Page — Global Database of Subnational Climate Indicators",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styles ----------
st.markdown("""
<style>
:root { --muted:#64748b; }
h1, h2, h3 { letter-spacing:.2px; }
.subtitle { text-align:center; color:#64748b; margin-top:-.4rem; }
.card { border:1px solid #e5e7eb; border-radius:14px; padding:12px 14px; background:#fafafa; }
.hero { padding:12px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#f8fafc; margin-bottom:.75rem; }
.badge { padding:4px 8px; border-radius:999px; background:#eef2ff; border:1px solid #e0e7ff; font-size:12px; }
.footer-box { padding:16px; border-top:1px solid #e5e7eb; margin-top:1rem; color:#64748b; }
.full-bleed { width: 100vw; margin-left: calc(-50vw + 50%); }
@media (min-width: 1400px) {
  [data-testid="stAppViewContainer"] .main .block-container { padding-left: .5rem; padding-right: .5rem; }
}
.legend-chip { display:inline-flex; align-items:center; gap:8px;
  background:rgba(255,255,255,0.9); border:1px solid #e5e7eb; border-radius:10px; padding:6px 10px; font-size:12px; }
.legend-swatch { display:inline-block; width:10px; height:10px; background:#12a39a; border:1px solid rgba(0,0,0,.25); }
.align-with-input { height: 1.9rem; }
[data-testid="stPlotlyChart"] div, [data-testid="stPlotlyChart"] canvas { border-radius: 0 !important; }
.uc-card { border:1px solid #e5e7eb; border-radius:14px; padding:12px; background:white; height:100%; }
.uc-card h4 { margin:0 0 .25rem 0; font-size:16px; }
.uc-card p { margin:.15rem 0 0 0; font-size:13px; color:#475569; }
/* Subtle tinted hero-left panel */
.panel-left{
  position: relative; border: 1px solid rgba(235,92,86,0.25);
  border-radius:16px; padding:16px 18px; margin-bottom:10px; background: transparent !important; overflow: hidden;
}
.panel-left::before{ content:""; position:absolute; inset:0; background: rgba(235,92,86,0.06); border-radius: inherit; z-index: 0; }
.panel-left > *{ position: relative; z-index: 1; }
</style>
""", unsafe_allow_html=True)

# ---------- Secrets / env ----------
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

# Admin flag: show analytics panel if secrets say so, or ?admin=1
def _is_admin() -> bool:
    try:
        if hasattr(st, "secrets") and st.secrets.get("IS_ADMIN", False):
            return True
    except Exception:
        pass
    return st.query_params.get("admin", ["0"])[0] == "1"

IS_ADMIN = _is_admin()

# ---------- HF helpers with soft error capture ----------
if "hf_errors" not in st.session_state:
    st.session_state["hf_errors"] = []  # list[str]

def _note_err(msg: str):
    st.session_state["hf_errors"].append(msg)

@st.cache_data(ttl=24*3600, show_spinner=False)
def _download_from_hub(filename: str) -> str:
    try:
        return hf_hub_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=filename, token=_get_hf_token())
    except Exception as e:
        _note_err(f"Hub fetch failed for {filename}: {e}")
        raise

def _read_any_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)

# ---------- Availability snapshot ----------
@st.cache_data(ttl=24*3600, show_spinner=False)
def _read_availability_snapshot_local() -> Optional[pd.DataFrame]:
    here = Path(__file__).parent.resolve()
    for p in [here/"availability_snapshot.parquet", here/"availability_snapshot.csv",
              here/"data/availability_snapshot.parquet", here/"data/availability_snapshot.csv"]:
        if p.exists():
            try:
                return _read_any_table(p)
            except Exception as e:
                _note_err(f"Local read failed for {p.name}: {e}")
    return None

@st.cache_data(ttl=24*3600, show_spinner=False)
def _read_availability_snapshot_hf() -> Optional[pd.DataFrame]:
    for fname in ("availability_snapshot.parquet", "availability_snapshot.csv"):
        try:
            p = Path(_download_from_hub(fname))
            return _read_any_table(p)
        except Exception as e:
            _note_err(f"Hub read failed for {fname}: {e}")
    return None

def _load_availability_snapshot() -> Optional[pd.DataFrame]:
    snap = _read_availability_snapshot_local()
    if isinstance(snap, pd.DataFrame) and not snap.empty:
        return snap
    snap = _read_availability_snapshot_hf()
    if isinstance(snap, pd.DataFrame) and not snap.empty:
        return snap
    return None

# ---------- City-level core (for freshness badge) ----------
@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_core_city_parquets():
    try:
        cit = pd.read_parquet(_download_from_hub("city_temperature.snappy.parquet"))
    except Exception:
        cit = None
    try:
        cip = pd.read_parquet(_download_from_hub("city_precipitation.snappy.parquet"))
    except Exception:
        cip = None
    return cit, cip

@st.cache_data(ttl=24*3600, show_spinner=False)
def _global_latest_month() -> Optional[pd.Timestamp]:
    cit, cip = _load_core_city_parquets()
    dates = []
    for df in (cit, cip):
        if isinstance(df, pd.DataFrame) and "Date" in df.columns:
            d = pd.to_datetime(df["Date"], errors="coerce").max()
            if pd.notnull(d): dates.append(d)
    if not dates:
        return None
    return max(dates)

# ---------- Load country coverage ----------
snap = _load_availability_snapshot()
if snap is not None and {"iso3","has_temp","has_prec"} <= set(snap.columns):
    iso_temp = set(snap.loc[snap["has_temp"] == 1, "iso3"].astype(str).str.upper())
    iso_prec = set(snap.loc[snap["has_prec"] == 1, "iso3"].astype(str).str.upper())
    iso_with_data = iso_temp | iso_prec
else:
    # fallback from parquet files
    def _safe_read(fname):
        try:
            return pd.read_parquet(_download_from_hub(fname))
        except Exception as e:
            _note_err(f"Hub read failed for {fname}: {e}")
            return None
    ct = _safe_read("country_temperature.snappy.parquet")
    cp = _safe_read("country_precipitation.snappy.parquet")
    cit = _safe_read("city_temperature.snappy.parquet")
    cip = _safe_read("city_precipitation.snappy.parquet")

    def _isos(df) -> set:
        if not isinstance(df, pd.DataFrame): return set()
        for c in ("iso3","iso_a3","Country"):
            if c in df.columns:
                s = df[c].dropna().astype(str).str.upper().str.strip().unique().tolist()
                return set(s)
        return set()

    iso_temp = _isos(ct) | _isos(cit)
    iso_prec = _isos(cp) | _isos(cip)
    iso_with_data = iso_temp | iso_prec

# Expose availability early (needed by pending nav)
st.session_state["iso_temp"] = iso_temp
st.session_state["iso_prec"] = iso_prec
st.session_state["iso_with_data"] = iso_with_data

# ---------- Title / subtitle / timestamps ----------
latest_global = _global_latest_month()
fresh_label = pd.to_datetime(latest_global).strftime("%b %Y") if pd.notnull(latest_global) else "—"

def _now_label() -> str:
    try:
        now = datetime.now(LOCAL_TZ) if LOCAL_TZ else datetime.now()
        return now.strftime("%b %d, %Y %H:%M %Z")
    except Exception:
        return datetime.now().strftime("%b %d, %Y %H:%M")

# ---------- Indicator routing ----------
AVAILABLE_INDICATORS = ["Temperature", "Precipitation"]
INDICATOR_TO_PAGE = {
    "Temperature": ("pages/1_Temperature_Dashboard.py", "1_Temperature_Dashboard"),
    "Precipitation": ("pages/2_Precipitation_Dashboard.py", "2_Precipitation_Dashboard"),
}

def _navigate_to_dashboard_immediate(iso3: str, indicator: str):
    """Execute navigation now (used by the pending-nav runner)."""
    ind = indicator or "Temperature"
    page_path, page_qp = INDICATOR_TO_PAGE.get(ind, INDICATOR_TO_PAGE["Temperature"])
    # availability fallback
    if ind == "Precipitation" and iso3 not in st.session_state.get("iso_prec", set()):
        ind = "Temperature"
        page_path, page_qp = INDICATOR_TO_PAGE["Temperature"]
    st.session_state["nav_iso3"] = iso3
    st.query_params.update({"page": page_qp, "iso3": iso3, "city": ""})
    try:
        st.switch_page(page_path)
    except Exception:
        st.rerun()

# ---------- Pending navigation handler (runs after availability is ready) ----------
def _perform_nav_if_pending():
    nav = st.session_state.get("_pending_nav")
    if not nav:
        return
    iso3 = nav.get("iso3")
    indicator = nav.get("indicator") or "Temperature"
    st.session_state["_pending_nav"] = None  # clear first to avoid loops
    _navigate_to_dashboard_immediate(iso3, indicator)

# Call once now (availability already set above)
_perform_nav_if_pending()

# ---------- Countries master (names) ----------
if pycountry:
    all_countries = pd.DataFrame(
        [{"iso3": c.alpha_3, "name": c.name} for c in pycountry.countries if hasattr(c, "alpha_3")]
    )
else:
    all_countries = pd.DataFrame({"iso3": sorted(list(iso_with_data))})
    all_countries["name"] = all_countries["iso3"]

all_countries["iso3"] = all_countries["iso3"].astype(str).str.upper().str.strip()

# Name remaps (per your requirements)
_name_overrides = {"CHN": "People's Republic of China", "TWN": "Taipe, China", "HKG": "Hong Kong, China"}
all_countries["name"] = all_countries.apply(lambda r: _name_overrides.get(r["iso3"], r.get("name", r["iso3"])), axis=1)

all_countries["has_data_temp"] = all_countries["iso3"].isin(iso_temp)
all_countries["has_data_prec"] = all_countries["iso3"].isin(iso_prec)
all_countries["has_data_any"]  = all_countries["has_data_temp"] | all_countries["has_data_prec"]

def _badges(iso):
    has_t = iso in iso_temp
    has_p = iso in iso_prec
    if has_t and has_p: return "Temperature • Precipitation"
    if has_t:          return "Temperature"
    if has_p:          return "Precipitation"
    return "—"
all_countries["badges"] = all_countries["iso3"].map(_badges)

# --- Continent membership helper (for filtering) ---
@st.cache_data(show_spinner=False)
def _continent_lookup() -> dict:
    gm = px.data.gapminder()
    base = dict(zip(gm["iso_alpha"], gm["continent"]))
    south_america = {"ARG","BOL","BRA","CHL","COL","ECU","GUY","PRY","PER","SUR","URY","VEN","FLK","GUF"}
    north_america = {"USA","CAN","MEX","GTM","BLZ","HND","SLV","NIC","CRI","PAN","CUB","DOM","HTI","JAM","TTO","BRB","BHS",
                     "ATG","DMA","GRD","KNA","LCA","VCT","ABW","BES","BMU","CUW","GLP","GRL","MTQ","MSR","PRI","SXM","SJM",
                     "TCA","VGB","VIR"}
    out = {}
    for iso, cont in base.items():
        if cont != "Americas":
            out[iso] = cont
        else:
            out[iso] = "South America" if iso in south_america else "North America"
    return out

CONTINENT_OF = _continent_lookup()

def _isos_for_region(region_name: str, all_isos: pd.Series) -> set:
    if region_name == "World":
        return set(all_isos.tolist())
    res = {iso for iso in all_isos if CONTINENT_OF.get(iso, None) == region_name}
    return res or set(all_isos.tolist())

# ---------- Title & subtitle ----------
st.markdown("<h1 style='text-align:center'>Global Database of Subnational Climate Indicators</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Built and Maintained by Roshen Fernando and Patrick Jaime Simba</div>", unsafe_allow_html=True)
st.divider()

# ===== HERO split: left (message) + right (Custom Chart Builder) =====
left, right = st.columns([0.62, 0.38], gap="large")
with left:
    st.markdown(
        f"""
        <div class="panel-left">
          <h2>🌍 Explore subnational climate indicators worldwide</h2>
          <p>Click a country on the map to open its dashboard, or use Quick search to jump directly.</p>
          <div class="badgerow">
            <span class="badge">Data through: <b>{fresh_label}</b></span>
            <span class="badge" style="margin-left:6px;">Last Update: <b>{_now_label()}</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with right:
    st.subheader("Custom Chart Builder")
    st.write(
        "Create bespoke charts across countries and ADM1s, compare indicators, "
        "combine countries/ADM1s, facet, smooth, normalize and export."
    )
    if st.button("📈 Generate a custom chart", key="hero_custom_chart"):
        try:
            st.switch_page("pages/0_Custom_Chart.py")
        except Exception:
            st.rerun()
    st.caption("Starts a flexible chart workspace with export options.")

# ---------- First-load cache hint ----------
if "first_load_hint" not in st.session_state:
    st.info("Tip: first load warms the cache; subsequent loads should be faster.", icon="💡")
    st.session_state["first_load_hint"] = True

# ---------- Controls ----------
if "region_scope" not in st.session_state:
    st.session_state["region_scope"] = "World"
if "default_indicator" not in st.session_state:
    st.session_state["default_indicator"] = "Temperature"

def _log_event(evt: str, payload: dict):
    if "analytics" not in st.session_state:
        st.session_state["analytics"] = []
    ts = datetime.now(LOCAL_TZ).isoformat() if LOCAL_TZ else datetime.now().isoformat()
    st.session_state["analytics"].append({"ts": ts, "event": evt, **payload})

def _reset_scope():
    st.session_state["region_scope"] = "World"
    _log_event("reset_view", {"to": "World"})

def _request_nav(iso3: str, indicator: str):
    """Queue navigation; actual switch happens at top via _perform_nav_if_pending()."""
    st.session_state["_pending_nav"] = {"iso3": iso3, "indicator": indicator}
    st.rerun()

quick_opts = ["— Type to search —"] + sorted(all_countries["name"].tolist())

# Quick search | Default Indicator | View | Reset
c1, c2, c3, c4 = st.columns([1, 0.32, 0.25, 0.15])
with c1:
    chosen = st.selectbox(
        "Quick search",
        options=quick_opts,
        index=0,
        help="Type a country name, or click a country on the map to navigate to its dashboard."
    )
with c2:
    st.selectbox(
        "Default Indicator",
        ["Temperature","Precipitation"],
        key="default_indicator",
        help="Which dashboard opens when you click a country or use Quick search. Does not change the map."
    )
with c3:
    st.selectbox(
        "View",
        ["World","Africa","Asia","Europe","North America","South America","Oceania"],
        key="region_scope",
        index=["World","Africa","Asia","Europe","North America","South America","Oceania"].index(st.session_state["region_scope"]),
        help="Change geographic scope."
    )
with c4:
    st.markdown('<div class="align-with-input"></div>', unsafe_allow_html=True)
    st.button("Reset view", use_container_width=True, on_click=_reset_scope)

# Legend
st.markdown("""
<div class="legend-chip" style="margin: 6px 0 4px 0;">
  <span class="legend-swatch"></span>
  Countries with available indicators
</div>
""", unsafe_allow_html=True)

# Log view changes
if "last_region_logged" not in st.session_state:
    st.session_state["last_region_logged"] = st.session_state["region_scope"]
if st.session_state["last_region_logged"] != st.session_state["region_scope"]:
    _log_event("view_change", {"to": st.session_state["region_scope"]})
    st.session_state["last_region_logged"] = st.session_state["region_scope"]

# Navigate if quick search used — respects Default Indicator
if chosen and chosen != "— Type to search —":
    row = all_countries.loc[all_countries["name"] == chosen].iloc[0]
    iso3_jump = row["iso3"]
    if iso3_jump in iso_with_data:
        _log_event("quick_search_open", {"iso3": iso3_jump, "indicator": st.session_state.get("default_indicator", "Temperature")})
        _request_nav(iso3_jump, st.session_state.get("default_indicator", "Temperature"))
    else:
        st.info(f"{chosen}: No available indicators.", icon="ℹ️")

# ---------- MAP ----------
vp = components.html("""
<script>
(function(){function send(){const p={width:window.innerWidth,height:window.innerHeight};
window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setComponentValue',value:p},'*');}
window.addEventListener('resize',send);send();})();
</script>
""", height=0)
vw = int(vp["width"]) if isinstance(vp, dict) and "width" in vp else 1280

# Taller for continents so small countries are easier to click
height_ratio = {"World":0.50,"Africa":0.82,"Asia":0.80,"Europe":0.90,"North America":0.82,"South America":0.94,"Oceania":0.86}
ratio = height_ratio.get(st.session_state["region_scope"], 0.70)
map_h = max(600, int(vw * ratio))

region_isos = _isos_for_region(st.session_state["region_scope"], all_countries["iso3"])
plot_df = all_countries[all_countries["iso3"].isin(region_isos)].copy()
plot_df["hovertext"] = plot_df.apply(lambda r: f"{r['name']}<br><span>Indicators: {r['badges']}</span>", axis=1)
plot_df["val"] = plot_df["has_data_any"].astype(float)

fig = go.Figure(go.Choropleth(
    locations=plot_df["iso3"], z=plot_df["val"], locationmode="ISO-3",
    colorscale=[[0.0, "#d4d4d8"], [1.0, "#12a39a"]],
    zmin=0.0, zmax=1.0, autocolorscale=False, showscale=False,
    hoverinfo="text", text=plot_df["hovertext"],
    customdata=plot_df[["iso3"]].to_numpy(),
    marker_line_width=1.6, marker_line_color="rgba(0,0,0,0.70)",
))
scope_map = {"World":"world","Africa":"africa","Asia":"asia","Europe":"europe","North America":"north america","South America":"south america","Oceania":"world"}
scope_val = scope_map.get(st.session_state["region_scope"], "world")
fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=map_h,
    geo=dict(projection_type="robinson", showframe=False, showcoastlines=False, showocean=False,
             bgcolor="rgba(0,0,0,0)", scope=scope_val,
             fitbounds="locations" if st.session_state["region_scope"]!="World" else None,
             lataxis_range=[-60,85] if st.session_state["region_scope"]=="World" else None))

st.markdown('<div class="full-bleed">', unsafe_allow_html=True)
if plotly_events:
    events = plotly_events(fig, click_event=True, hover_event=False, override_height=map_h, override_width="100%")
else:
    st.plotly_chart(fig, use_container_width=True)
    events = []
st.markdown('</div>', unsafe_allow_html=True)

# Click → Selected indicator dashboard (queue + next-run perform)
clicked_iso3 = None
if events:
    e = events[0]
    if isinstance(e, dict):
        idx = e.get("pointIndex", None)
        if idx is not None and 0 <= idx < len(plot_df):
            clicked_iso3 = str(plot_df.iloc[idx]["iso3"]).upper()

if clicked_iso3 and clicked_iso3 in iso_with_data:
    _log_event("map_click_open", {"iso3": clicked_iso3, "indicator": st.session_state.get("default_indicator", "Temperature")})
    _request_nav(clicked_iso3, st.session_state.get("default_indicator", "Temperature"))

# =========================
# GLOBAL SNAPSHOT (BETA)
# =========================
st.divider()
st.subheader("Global snapshot (beta)")

@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_country_temp() -> Optional[pd.DataFrame]:
    try:
        p = Path(_download_from_hub("country_temperature.snappy.parquet")); return pd.read_parquet(p)
    except Exception:
        for candidate in (Path("/mnt/data/country_temperature.csv"), Path("data/country_temperature.csv")):
            if candidate.exists():
                try: return pd.read_csv(candidate)
                except Exception: pass
    return None

@st.cache_data(ttl=24*3600, show_spinner=False)
def _load_country_prec() -> Optional[pd.DataFrame]:
    try:
        p = Path(_download_from_hub("country_precipitation.snappy.parquet")); return pd.read_parquet(p)
    except Exception:
        for candidate in (Path("/mnt/data/country_precipitation.csv"), Path("data/country_precipitation.csv")):
            if candidate.exists():
                try: return pd.read_csv(candidate)
                except Exception: pass
    return None

def _ensure_iso_date(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty: return None
    cols = {c.lower(): c for c in df.columns}
    iso_col = next((cols[k] for k in ("iso3","iso_a3","country_iso3","countrycode","country") if k in cols), None)
    date_col = next((cols[k] for k in ("date","month","time","period") if k in cols), None)
    if not iso_col or not date_col: return None
    d = df.rename(columns={iso_col:"iso3", date_col:"Date"}).copy()
    d["iso3"] = d["iso3"].astype(str).str.upper().str.strip()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["iso3","Date"])
    return d

def _pick_value_column(df: pd.DataFrame, indicator: str) -> Optional[str]:
    if df is None or df.empty: return None
    cols = {c.lower(): c for c in df.columns}
    if indicator == "temp":
        preferred = ["temperature_c","temp_c","tavg_c","tas_c","temperature","temp","tavg","tas","mean_temp"]
    else:
        preferred = ["precip_mm","prcp_mm","pr_mm","precipitation_mm","precipitation","prcp","pr","ppt","rain_mm"]
    for k in preferred:
        if k.lower() in cols and pd.api.types.is_numeric_dtype(df[cols[k.lower()]]):
            return cols[k.lower()]
    tokens = ["temp","tas"] if indicator=="temp" else ["precip","prcp","ppt","rain","pr"]
    candidates = [c for c in df.columns if any(t in c.lower() for t in tokens) and pd.api.types.is_numeric_dtype(df[c])]
    if candidates: return candidates[0]
    avoid = {"area","population","pop","lat","lon","latitude","longitude"}
    for c in df.columns:
        if c.lower() in avoid: continue
        if pd.api.types.is_numeric_dtype(df[c]): return c
    return None

def _kelvin_to_c_if_needed(s: pd.Series) -> pd.Series:
    if s is None or s.empty: return s
    try:
        med = float(s.dropna().median())
        return s - 273.15 if med > 200 else s
    except Exception:
        return s

def _latest_by_country(df: pd.DataFrame, value_col: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty or not {"iso3","Date", value_col} <= set(df.columns):
        return None
    d = df[["iso3","Date", value_col]].copy().sort_values(["iso3","Date"])
    out = d.drop_duplicates(subset=["iso3"], keep="last").reset_index(drop=True)
    return out

def _series_latest_for_hist(raw_df: Optional[pd.DataFrame], indicator: str) -> Optional[pd.Series]:
    df = _ensure_iso_date(raw_df)
    if df is None: return None
    val_col = _pick_value_column(df, indicator)
    if not val_col: return None
    latest = _latest_by_country(df, val_col)
    if latest is None or latest.empty: return None
    s = latest[val_col]
    if indicator == "temp":
        s = _kelvin_to_c_if_needed(s)
        s = s[(s > -80) & (s < 60)]
    else:
        s = s[(s >= 0) & (s < 2000)]
    return s.dropna()

# Load raw files
_ct_raw = _load_country_temp()
_cp_raw = _load_country_prec()

# (1) Global coverage over time
def _monthly_coverage(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    df = _ensure_iso_date(df)
    if df is None: return None
    d = df[["iso3","Date"]].copy()
    d["Date"] = d["Date"].dt.to_period("M").dt.to_timestamp()
    cov = d.groupby("Date")["iso3"].nunique().reset_index(name="countries")
    return cov.sort_values("Date")

cov_t = _monthly_coverage(_ct_raw)
cov_p = _monthly_coverage(_cp_raw)

if cov_t is not None or cov_p is not None:
    cov_fig = go.Figure()
    if cov_t is not None:
        cov_fig.add_trace(go.Scatter(x=cov_t["Date"], y=cov_t["countries"], mode="lines", name="Temperature"))
    if cov_p is not None:
        cov_fig.add_trace(go.Scatter(x=cov_p["Date"], y=cov_p["countries"], mode="lines", name="Precipitation"))
    cov_fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Month",
        yaxis_title="Countries with data",
    )
    st.plotly_chart(cov_fig, use_container_width=True)
else:
    st.info("Global coverage over time is unavailable at the moment.", icon="ℹ️")

# (2) Latest datapoint per country — histograms
hcol1, hcol2 = st.columns(2)

with hcol1:
    s = _series_latest_for_hist(_ct_raw, "temp")
    if s is not None and not s.empty:
        hist_t = px.histogram(pd.DataFrame({"value": s}), x="value", nbins=30, title="Latest datapoint per country — Temperature")
        hist_t.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=260, xaxis_title="Temperature (°C)", yaxis_title="Countries")
        st.plotly_chart(hist_t, use_container_width=True)
    else:
        st.info("Latest temperature distribution unavailable.", icon="ℹ️")

with hcol2:
    s = _series_latest_for_hist(_cp_raw, "precip")
    if s is not None and not s.empty:
        hist_p = px.histogram(pd.DataFrame({"value": s}), x="value", nbins=30, title="Latest datapoint per country — Precipitation")
        hist_p.update_layout(margin=dict(l=0,r=0,t=40,b=0), height=260, xaxis_title="Precipitation (mm)", yaxis_title="Countries")
        st.plotly_chart(hist_p, use_container_width=True)
    else:
        st.info("Latest precipitation distribution unavailable.", icon="ℹ️")

# ---------- COVERAGE & DATA SOURCES ----------
st.divider()
k1, k2 = st.columns([1,1])
with k1:
    st.markdown(f"""
    <div class="card">
      <div style="font-size:13px;color:#64748b;">Coverage</div>
      <div style="font-size:22px;margin:.15rem 0;">
        <strong>{int(all_countries['has_data_any'].sum())}</strong> countries with at least one indicator
      </div>
      <div style="font-size:13px;color:#475569;">
        Indicators shown: Temperature • Precipitation
      </div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="card">
      <div style="font-size:13px;color:#64748b; display:flex; align-items:center; gap:6px;">
        Data sources
        <span title="ERA5 reanalysis (ECMWF). Gridded climate fields; aggregated to country, region, city. License and terms apply at the provider.">
          ⓘ
        </span>
      </div>
      <div style="font-size:22px;margin:.15rem 0;"><strong>ERA5</strong></div>
      <div style="font-size:13px;color:#475569;">Additional integrations under consideration</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- spacing before expander ----------
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# ---------- "What’s inside" + Methods / QC ----------
with st.expander("What’s inside this dashboard?", expanded=False):
    st.markdown("""
- **Geography:** Countries and first-level administrative regions (ADM1); selected cities for context.
- **Indicators:** Air temperature and precipitation (aggregated from gridded sources).
- **Temporal frequency:** Monthly; global snapshot shown as “Data through”.
- **Latency:** Updates typically published within weeks of source release.
- **Method summary:** Area-weighted aggregation of grid cells to administrative boundaries.
- **Caveats:** Administrative boundary changes, data gaps, and reanalysis corrections can affect comparability over time.
""")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        try:
            st.page_link("pages/7_Methodology.py", label="📘 Methodology", icon="📘")
        except Exception:
            st.markdown("📘 **Methodology**: Contact maintainers for full documentation.")
    with mcol2:
        try:
            st.page_link("pages/8_Quality_Control.py", label="🛠️ Quality Control", icon="🛠️")
        except Exception:
            st.markdown("🛠️ **Quality Control**: Summary of checks available upon request.")

# ---------- Soft offline / hub error banner ----------
if st.session_state.get("hf_errors"):
    with st.expander("Notice: Some resources failed to load from the data hub (using available fallbacks)."):
        for msg in st.session_state["hf_errors"]:
            st.write("•", msg)

# ---------- ADMIN ANALYTICS (hidden unless admin) ----------
if IS_ADMIN:
    st.divider()
    st.subheader("Admin: Session analytics")
    a = st.session_state.get("analytics", [])
    st.caption("Lightweight, in-session logs. Export below. (Counts reset per session.)")
    if a:
        df_log = pd.DataFrame(a)
        st.dataframe(df_log, use_container_width=True)
        st.download_button(
            "Download logs (CSV)",
            data=df_log.to_csv(index=False).encode("utf-8"),
            file_name=f"home_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        st.markdown(f"- **View changes:** {(df_log['event']=='view_change').sum()}")
        st.markdown(f"- **Quick-search opens:** {(df_log['event']=='quick_search_open').sum()}")
        st.markdown(f"- **Map click opens:** {(df_log['event']=='map_click_open').sum()}")
        st.markdown(f"- **Resets:** {(df_log['event']=='reset_view').sum()}")
    else:
        st.info("No events logged yet in this session.", icon="ℹ️")

# ---------- Footer ----------
st.markdown("""
<div class="footer-box">
  <em>Note:</em> This page time-stamps the "Last Update" at render time (Asia/Manila).
</div>
""", unsafe_allow_html=True)
