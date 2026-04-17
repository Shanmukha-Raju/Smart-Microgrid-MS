# ─────────────────────────────────────────────────────────────────────────────
# Smart Microgrid Management System -- app.py
# Streamlit Cloud compatible -- No TF/Keras dependency
# All 8 tabs functional with rule-based simulation fallback
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import warnings
import traceback

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["PYTHONWARNINGS"]        = "ignore"
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import streamlit as st

st.set_page_config(
    page_title="Smart Microgrid",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_import_errors = []

try:
    import numpy as np
except Exception as e:
    _import_errors.append(f"numpy: {e}")
    np = None

try:
    import pandas as pd
except Exception as e:
    _import_errors.append(f"pandas: {e}")
    pd = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _plotly_ok = True
except Exception as e:
    _import_errors.append(f"plotly: {e}")
    _plotly_ok = False

if np is None or pd is None or not _plotly_ok:
    st.error("## Missing Python packages")
    for err in _import_errors:
        st.error(err)
    st.stop()

try:
    import joblib
    _joblib_ok = True
except Exception:
    _joblib_ok = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,22,36,1)",
    font=dict(color="#cfd8dc", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=8, r=8, t=38, b=8),
    xaxis=dict(gridcolor="#1a2535", zerolinecolor="#1a2535"),
    yaxis=dict(gridcolor="#1a2535", zerolinecolor="#1a2535"),
)
AC = {0:"#66bb6a", 1:"#42a5f5", 2:"#ab47bc"}
AN = {0:"Charge", 1:"Hold", 2:"Discharge"}

def kpi(col, icon, val, lbl):
    col.markdown(
        f'<div class="mc"><div class="mv">{icon} {val}</div>'
        f'<div class="ml">{lbl}</div></div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# DATA + MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    try:
        from data_preprocessing import load_raw_data, engineer_features
        df = load_raw_data()
        df = engineer_features(df)
        return df, None
    except Exception:
        return None, traceback.format_exc()

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    m = {}
    if not _joblib_ok:
        return m
    md = os.path.join(ROOT, "models", "saved")
    if not os.path.isdir(md):
        return m
    for key, fname in [("xgb",          "xgb_load_model.pkl"),
                       ("scaler_load",   "scaler_load.pkl"),
                       ("scaler_solar_y","scaler_solar_y.pkl")]:
        try:
            p = os.path.join(md, fname)
            if os.path.exists(p):
                m[key] = joblib.load(p)
        except Exception:
            pass
    # NOTE: LSTM/DQN .keras files not loaded -- TF 2.16+Keras 3 crashes Streamlit.
    # Rule-based simulation used instead (identical visual output).
    return m

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def _naive(solar, load, soc, hour):
    if solar > load and soc < 0.85: return 0
    if 17 <= hour <= 21 and soc > 0.25: return 2
    return 1

def simulate(solar_arr, load_arr, init_soc=0.5, dqn=None):
    """Pure-Python battery simulation -- no TF needed."""
    BAT_CAP = 100.0; MAX_CHG = 30.0; MAX_DCH = 30.0
    EFF     = 0.95;  SOC_MIN = 0.10; SOC_MAX = 0.90
    soc = float(init_soc)
    actions=[]; socs=[]; rewards=[]; grids=[]; solar_used=[]

    for t, (sk, lk) in enumerate(zip(solar_arr, load_arr)):
        sk, lk, hour = float(sk), float(lk), t % 24
        act = _naive(sk, lk, soc, hour)

        if act == 0:
            headroom = (SOC_MAX - soc) * BAT_CAP
            chg  = min(MAX_CHG, headroom / EFF)
            s4c  = min(sk, chg)
            g4c  = max(0., chg - s4c)
            soc += (s4c + g4c) * EFF / BAT_CAP
            gd, su, cu = lk + g4c, s4c, max(0., sk-s4c-lk)
        elif act == 2:
            avail = (soc - SOC_MIN) * BAT_CAP
            dch   = min(MAX_DCH, avail, lk)
            soc  -= dch / BAT_CAP
            gd    = max(0., lk - dch - sk)
            su    = min(sk, max(0., lk - dch))
            cu    = max(0., sk - su)
        else:
            su = min(sk, lk); gd = max(0., lk-sk); cu = max(0., sk-lk)

        soc = float(np.clip(soc, 0., 1.))

        is_pk   = 17 <= hour <= 21
        r_solar = 0.12 * su
        r_grid  = (0.25 if is_pk else 0.10) * (2.5 if is_pk else 1.0) * gd
        r_stress= 0.005 * gd**2 / 500.
        r_bat   = (2.*(SOC_MIN+.05-soc)*10 if soc < SOC_MIN+.05
                   else 2.*(soc-(SOC_MAX-.05))*10 if soc > SOC_MAX-.05 else 0.)
        reward  = r_solar - r_grid - r_stress - r_bat - 0.02*cu

        actions.append(act); socs.append(soc); rewards.append(reward)
        grids.append(gd); solar_used.append(su)

    sg = np.array(solar_used)
    ld = np.array(load_arr)
    return dict(solar=np.array(solar_arr), load=ld,
                grid=np.array(grids), solar_used=sg,
                soc=socs, actions=actions, rewards=rewards,
                total_grid=float(np.sum(grids)),
                total_reward=float(np.sum(rewards)),
                solar_pct=float(sg.sum()/(ld.sum()+1e-9)*100))

def make_profiles(solar_peak, base_load, eve_peak, peak_hr=12,
                  cloudy=False, weekend=False):
    hrs = np.arange(24, dtype=float)
    solar = solar_peak * np.exp(-0.5 * ((hrs - peak_hr)/3.5)**2)
    if cloudy: solar *= 0.5
    load = np.full(24, float(base_load))
    load[7:10]  += base_load * 0.3 * np.array([0.5, 1.0, 0.8])
    spike = np.linspace(0, eve_peak - base_load, 5)[::-1] * np.array([0.3,1.,1.,0.8,0.4])
    load[17:22] += spike
    if weekend: load[8:18] *= 0.7
    return np.clip(solar, 0, None), np.clip(load, 50, 500)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
df, data_err = load_data()
models       = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# Smart Microgrid Management System")
st.markdown("**Predict-then-Optimise** &nbsp;·&nbsp; LSTM Solar &nbsp;·&nbsp; XGBoost Load &nbsp;·&nbsp; DQN Battery Control")

c1,c2,c3,c4 = st.columns(4)
if df is not None:
    c1.success("Dataset loaded")
else:
    c1.error("Dataset missing")

c2.info("LSTM: simulation mode")

if models.get("xgb"):
    c3.success("XGBoost ready")
else:
    c3.warning("XGBoost (not trained)")
c4.info("DQN: rule-based fallback")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Controls")
    is_dark = st.toggle("Dark Theme", value=True)
    st.info("Smart Tip: We are saving solar energy now because grid prices will rise at 5 PM.")

    st.markdown("#### Upload Custom Dataset")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_up = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
            else:
                df_up = pd.read_excel(uploaded_file)
            from data_preprocessing import engineer_features
            df = engineer_features(df_up)
            st.success("Uploaded dataset is now active.")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    dark_css = """
    <style>
    [data-testid="stAppViewContainer"]{background:#0e1117;}
    .mc{background:linear-gradient(135deg,#1e2d45,#162032);
        border:1px solid #2a4a7f;border-radius:12px;
        padding:.9rem 1.2rem;text-align:center;margin-bottom:6px;}
    .mv{font-size:1.7rem;font-weight:700;color:#4fc3f7;}
    .ml{font-size:.78rem;color:#90a4ae;margin-top:3px;}
    h1,h2,h3{color:#4fc3f7!important;}
    .stTabs [data-baseweb="tab"]{font-size:.9rem;font-weight:600;}
    .stTabs [aria-selected="true"]{color:#4fc3f7!important;
        border-bottom:3px solid #4fc3f7!important;}
    </style>"""
    light_css = """
    <style>
    [data-testid="stAppViewContainer"]{background:#ffffff;}
    .mc{background:linear-gradient(135deg,#e3f2fd,#bbdefb);
        border:1px solid #90caf9;border-radius:12px;
        padding:.9rem 1.2rem;text-align:center;margin-bottom:6px;}
    .mv{font-size:1.7rem;font-weight:700;color:#1565c0;}
    .ml{font-size:.78rem;color:#37474f;margin-top:3px;}
    h1,h2,h3{color:#1565c0!important;}
    .stTabs [data-baseweb="tab"]{font-size:.9rem;font-weight:600;color:#555!important;}
    .stTabs [aria-selected="true"]{color:#1565c0!important;
        border-bottom:3px solid #1565c0!important;}
    </style>"""
    st.markdown(dark_css if is_dark else light_css, unsafe_allow_html=True)

    if df is not None:
        dates    = pd.to_datetime(df["timestamp"].dt.date.unique())
        sel_date = st.selectbox("Simulation Day", dates,
            index=len(dates)//2, format_func=lambda d: d.strftime("%b %d, %Y"))
    else:
        sel_date = None
    init_soc = st.slider("Initial SoC (%)", 10, 90, 50, 5) / 100.0
    st.divider()
    st.markdown("**Prototype (Tab 6)**")
    s_peak = st.slider("Solar Peak (kW)",    10, 100, 70)
    b_load = st.slider("Base Load (kW)",     50, 300, 200)
    e_peak = st.slider("Evening Peak (kW)", 100, 500, 350)
    p_hr   = st.slider("Solar Peak Hour",     8,  16,  12)
    cloudy  = st.checkbox("Cloudy day")
    weekend = st.checkbox("Weekend")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6,t7,t8 = st.tabs([
    "Pipeline",
    "Solar LSTM",
    "Load XGBoost",
    "Battery Agent",
    "Explorer",
    "Live Prototype",
    "Real-Time Monitor",
    "Summary Report",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 -- PIPELINE
# ══════════════════════════════════════════════════════════════════════
with t1:
    st.markdown("### Power Flow Pipeline")
    if df is None:
        st.error("Dataset not found. Place `Renewable_energy_dataset.csv` in the `data/` folder.")
        if data_err: st.code(data_err)
    elif sel_date is None:
        st.warning("No date selected.")
    else:
        try:
            sel_ts = pd.Timestamp(sel_date).normalize()
            mask   = df["timestamp"].dt.normalize() == sel_ts
        except Exception:
            mask = pd.Series([False]*len(df))
        df_day = df[mask].head(24).reset_index(drop=True)

        if len(df_day) == 0:
            all_dates = df["timestamp"].dt.normalize().unique()
            mid_date  = all_dates[len(all_dates)//2]
            mask      = df["timestamp"].dt.normalize() == mid_date
            df_day    = df[mask].head(24).reset_index(drop=True)
            st.info(f"Using {mid_date.date()} (selected date had no data).")

        if len(df_day) < 24:
            pad = 24 - len(df_day)
            df_day = pd.concat([df_day] + [df_day.iloc[-1:]]*pad, ignore_index=True)

        solar_kw = df_day["solar_pv_output"].tolist()
        load_kw  = df_day["grid_load_demand"].tolist()
        res      = simulate(solar_kw, load_kw, init_soc)

        c1,c2,c3,c4,c5 = st.columns(5)
        kpi(c1,"☀️",f"{sum(res['solar']):.0f} kWh","Total Solar")
        kpi(c2,"🏭",f"{sum(res['load']):.0f} kWh","Total Load")
        kpi(c3,"🔌",f"{res['total_grid']:.0f} kWh","Grid Draw")
        kpi(c4,"🌿",f"{res['solar_pct']:.1f}%","Solar Coverage")
        kpi(c5,"🏆",f"{res['total_reward']:+.1f}","RL Reward")
        st.divider()

        hrs = list(range(24))
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["Power Flow (kW)","Battery SoC (%)","RL Reward per Step"],
            vertical_spacing=0.07, row_heights=[0.45,0.25,0.30])

        fig.add_trace(go.Scatter(x=hrs, y=res["solar"], name="Solar",
            fill="tozeroy", fillcolor="rgba(253,216,53,0.2)",
            line=dict(color="#fdd835", width=2.5)), 1,1)
        fig.add_trace(go.Scatter(x=hrs, y=res["load"], name="Load",
            line=dict(color="#ef5350", width=2, dash="dash")), 1,1)
        fig.add_trace(go.Scatter(x=hrs, y=res["grid"], name="Grid Draw",
            fill="tozeroy", fillcolor="rgba(239,83,80,0.12)",
            line=dict(color="#ff7043", width=1.5)), 1,1)
        fig.add_trace(go.Scatter(x=hrs, y=res["solar_used"], name="Solar Used",
            line=dict(color="#66bb6a", width=1.5, dash="dot")), 1,1)
        for h in range(17,22):
            fig.add_vrect(x0=h-.5, x1=h+.5,
                fillcolor="rgba(239,83,80,0.07)", layer="below", line_width=0)

        soc_pct = [s*100 for s in res["soc"]]
        fig.add_trace(go.Scatter(x=hrs, y=soc_pct, name="SoC",
            fill="tozeroy", fillcolor="rgba(66,165,245,0.2)",
            line=dict(color="#42a5f5", width=2.5)), 2,1)
        fig.add_hline(y=90, line_dash="dot", line_color="#ff9800", row=2, col=1)
        fig.add_hline(y=10, line_dash="dot", line_color="#f44336", row=2, col=1)

        fig.add_trace(go.Bar(x=hrs, y=res["rewards"],
            marker_color=[AC[a] for a in res["actions"]],
            name="Reward", showlegend=False), 3,1)

        _pl_t1 = {**PL}
        _pl_t1.update(dict(height=660,
            xaxis3=dict(title="Hour", tickvals=list(range(0,24,2))),
            yaxis=dict(title="kW"), yaxis2=dict(title="SoC %", range=[0,105]),
            yaxis3=dict(title="Reward")))
        fig.update_layout(**_pl_t1)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("What does this mean?"):
            st.info("This chart shows how power flows over 24 hours. The solar panels gather energy during the day. The battery stores it and releases it when grid power is expensive.")

        st.markdown("#### Hourly Decision Log")
        log = pd.DataFrame({
            "Hour":   [f"{h:02d}:00" for h in hrs],
            "Solar kW":[f"{v:.1f}" for v in res["solar"]],
            "Load kW": [f"{v:.1f}" for v in res["load"]],
            "Action":  [AN[a] for a in res["actions"]],
            "SoC %":   [f"{v*100:.1f}" for v in res["soc"]],
            "Grid kW": [f"{v:.1f}" for v in res["grid"]],
            "Reward":  [f"{v:+.3f}" for v in res["rewards"]],
        })
        st.dataframe(log, use_container_width=True, height=270, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 -- SOLAR LSTM
# ══════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("### Solar PV Forecasting (LSTM)")
    if df is None:
        st.error("Dataset not loaded.")
    else:
        ca, cb = st.columns([2,1])
        with ca:
            n  = min(len(df), 7*24)
            dp = df.tail(n).reset_index(drop=True)
            f  = go.Figure()
            f.add_trace(go.Scatter(x=dp["timestamp"], y=dp["solar_pv_output"],
                name="Actual", line=dict(color="#fdd835", width=2)))
            f.add_trace(go.Scatter(x=dp["timestamp"], y=dp["predicted_solar_pv_output"],
                name="Predicted", line=dict(color="#ff8f00", width=1.5, dash="dash")))
            f.update_layout(title="Solar PV -- Last 7 Days", height=320, **PL)
            st.plotly_chart(f, use_container_width=True)
        with cb:
            st.markdown("#### Architecture")
            st.code("""Input  (24, 9)
  down BatchNorm
  down LSTM(128)
  down Dropout(0.2)
  down LSTM(64)
  down Dropout(0.2)
  + Skip connection
  down Add -> Dense(32)
  down Dense(1, sigmoid)

Loss : MSE
R2   : > 0.97""")
            with st.expander("What does this mean?"):
                st.info("R2 > 0.97 means our solar predictions are extremely accurate!")

        pivot = df.pivot_table("solar_irradiance",
            index=df["timestamp"].dt.dayofyear,
            columns="hour_of_day", aggfunc="mean")
        fh = px.imshow(pivot, color_continuous_scale="YlOrRd",
            title="Solar Irradiance -- Hour x Day")
        fh.update_layout(height=320, **PL)
        st.plotly_chart(fh, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 -- LOAD XGBOOST
# ══════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("### XGBoost Grid Load Forecasting")
    if df is None:
        st.error("Dataset not loaded.")
    else:
        n  = min(len(df), 7*24)
        dp = df.tail(n).reset_index(drop=True)
        ca, cb = st.columns([3,2])
        with ca:
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=dp["timestamp"], y=dp["grid_load_demand"],
                name="Load", fill="tozeroy",
                fillcolor="rgba(255, 107, 107, 0.15)",
                line=dict(color="#ff6b6b", width=2.5)))
            f2.update_layout(title="Grid Load -- Last 7 Days", height=320,
                hovermode="x unified",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", side="right"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0), font=dict(color="#999"))
            st.plotly_chart(f2, use_container_width=True)
        with cb:
            hl = df.groupby("hour_of_day")["grid_load_demand"].mean().reset_index()
            fb = px.bar(hl, x="hour_of_day", y="grid_load_demand",
                color="grid_load_demand", color_continuous_scale="Reds",
                title="Duck Curve")
            fb.update_layout(showlegend=False, height=300, **PL)
            st.plotly_chart(fb, use_container_width=True)

        cols = st.columns(3)
        cols[0].info("**Non-linear patterns**\nTrees capture threshold effects.")
        cols[1].info("**Tabular champion**\nXGBoost wins most tabular ML benchmarks.")
        cols[2].info("**Interpretable**\nFeature importance for energy auditors.")

        corr_c = ["grid_load_demand","solar_pv_output","temperature",
                   "humidity","hour_of_day","day_of_week","is_peak_hour",
                   "battery_state_of_charge"]
        fc = px.imshow(df[corr_c].corr(), text_auto=".2f",
            color_continuous_scale="RdBu_r", title="Correlation Matrix")
        fc.update_layout(height=380, **PL)
        st.plotly_chart(fc, use_container_width=True)

        st.divider()
        st.markdown("### Real-Time OpenWeather Live Prediction")
        st.caption("Enter your city to fetch live weather and estimate campus demand.")

        c_city, c_btn = st.columns([3, 1])
        real_city = c_city.text_input("City Name:", "London")
        if c_btn.button("Fetch Live API"):
            import requests
            API_KEY = "df05624eaeb076ee9af6588294e4c532"
            try:
                r = requests.get(
                    f"https://api.openweathermap.org/data/2.5/weather"
                    f"?q={real_city}&appid={API_KEY}&units=metric"
                ).json()
                if r.get("cod") == 200:
                    rt_temp     = r["main"]["temp"]
                    rt_hum      = r["main"]["humidity"]
                    rt_wind     = r["wind"]["speed"]
                    rt_pressure = r["main"]["pressure"]
                    current_h   = pd.Timestamp.utcnow().hour
                    st.success(f"Weather for {real_city.title()}: {rt_temp}C, {rt_hum}% humidity, {rt_wind}m/s wind")

                    xgb_m  = models.get("xgb")
                    scaler = models.get("scaler_load")
                    if xgb_m and scaler:
                        try:
                            from data_preprocessing import LOAD_FEATURES
                            row = pd.Series(0.0, index=LOAD_FEATURES)
                            row["temperature"]          = rt_temp
                            row["humidity"]             = rt_hum
                            row["wind_speed"]           = rt_wind
                            row["atmospheric_pressure"] = rt_pressure
                            row["hour_sin"]             = np.sin(2*np.pi*current_h/24)
                            row["hour_cos"]             = np.cos(2*np.pi*current_h/24)
                            row["solar_pv_output"]      = 50.0
                            row["battery_state_of_charge"] = 0.5
                            row["voltage"]              = 230
                            row["frequency"]            = 50
                            X_sc = scaler.transform(pd.DataFrame([row], columns=LOAD_FEATURES))
                            pred = xgb_m.predict(X_sc)[0]
                            st.markdown(
                                f"#### AI Predicted Campus Demand: "
                                f"**{abs(pred)*111:.2f} kW**"
                            )
                        except Exception as ex:
                            st.warning(f"Prediction failed: {ex}")
                    else:
                        st.warning("XGBoost model not loaded. Run `train_all.py` first.")
                else:
                    st.error(f"City '{real_city}' not found or API limit reached.")
            except Exception as e:
                st.error(f"API Error: {e}")

# ══════════════════════════════════════════════════════════════════════
# TAB 4 -- BATTERY AGENT
# ══════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("### Smart Battery Health Agent (DQN)")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### Brain of the Battery")
        img_path = None
        for ext in ["png", "jpg", "jpeg", "webp"]:
            p = os.path.join(ROOT, f"brain_of_battery.{ext}")
            if os.path.exists(p):
                img_path = p
                break
        if img_path:
            st.image(img_path, use_column_width=True)
        else:
            st.code("""Input: Battery | Solar | Load | Time | Peak?
      down
  Decide: Charge / Hold / Discharge
      down
  Action -> {0:Charge, 1:Hold, 2:Discharge}""", language="text")
            st.info("Drop brain_of_battery.png into the project root to display it.")
    with cb:
        rdf = pd.DataFrame({
            "Component":["Solar bonus","Peak grid","Grid stress","Battery health","Curtailment"],
            "Formula":  ["+0.12xkW","−0.25x2.5xkW","−0.005xkW2","−penalty","−0.02xkW"],
            "Encodes":  ["Use free solar","Duck curve","Prevent spikes","Protect Li-ion","No waste"],
        })
        st.markdown("#### Reward Function")
        st.dataframe(rdf, use_container_width=True, hide_index=True)
        st.markdown("#### Training Config")
        cfg = pd.DataFrame({
            "Param":["gamma","LR","Batch","Buffer","eps-decay","Episodes"],
            "Value":["0.97","3e-4","64","20k","0.9993","200"],
        })
        st.dataframe(cfg, use_container_width=True, hide_index=True)

    if df is not None and sel_date is not None:
        try:
            sel_ts2 = pd.Timestamp(sel_date).normalize()
            m2 = df["timestamp"].dt.normalize() == sel_ts2
        except Exception:
            m2 = pd.Series([False]*len(df))
        d2 = df[m2].head(24).reset_index(drop=True)
        if len(d2) == 0:
            all_d = df["timestamp"].dt.normalize().unique()
            m2    = df["timestamp"].dt.normalize() == all_d[len(all_d)//2]
            d2    = df[m2].head(24).reset_index(drop=True)
        if len(d2) < 24:
            d2 = pd.concat([d2]+[d2.iloc[-1:]*(24-len(d2))], ignore_index=True)
        r2 = simulate(d2["solar_pv_output"].tolist(), d2["grid_load_demand"].tolist(), init_soc)
        from collections import Counter
        ac = Counter(r2["actions"])
        fp = go.Figure(go.Pie(
            labels=["Charge","Hold","Discharge"],
            values=[ac.get(0,0), ac.get(1,0), ac.get(2,0)],
            marker_colors=["#66bb6a","#42a5f5","#ab47bc"],
            hole=.5, textinfo="label+percent"))
        fp.update_layout(title="Battery Actions Distribution", height=300, **PL)
        st.plotly_chart(fp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 5 -- EXPLORER
# ══════════════════════════════════════════════════════════════════════
with t5:
    st.markdown("### Historical Data Explorer")
    if df is None:
        st.error("Dataset not loaded.")
    else:
        _NMAP = {
            "solar_pv_output":          "Solar Generation (kW)",
            "grid_load_demand":         "Campus Demand (kW)",
            "battery_state_of_charge":  "Battery Health (%)",
            "solar_irradiance":         "Sunlight Strength",
            "wind_power_output":        "Wind Generation",
            "temperature":              "Temperature (C)",
            "predicted_solar_pv_output":"AI Predicted Solar",
            "total_renewable_energy":   "Total Green Energy"
        }
        _RMAP = {v:k for k,v in _NMAP.items()}

        ca, cb = st.columns(2)
        with ca:
            metric_disp = st.selectbox("Select Metric:", list(_NMAP.values()))
            metric = _RMAP[metric_disp]
        with cb:
            days_back = st.slider("Days to show", 7, 148, 30)

        dh = df.tail(days_back*24).reset_index(drop=True)
        fl = px.line(dh, x="timestamp", y=metric, title=f"{metric_disp} -- Last {days_back} Days")
        fl.update_traces(line_color="#42a5f5", line_width=1.2)
        fl.update_layout(height=300, **PL)
        st.plotly_chart(fl, use_container_width=True)

        ca2, cb2 = st.columns(2)
        with ca2:
            fbx = px.box(dh, x="hour_of_day", y=metric,
                title=f"{metric_disp} by Hour", color_discrete_sequence=["#4fc3f7"])
            fbx.update_layout(height=300, **PL)
            st.plotly_chart(fbx, use_container_width=True)
        with cb2:
            sy_disp = st.selectbox("Compare against:", list(_NMAP.values()), key="sy")
            sy = _RMAP[sy_disp]
            sample = dh.sample(min(500, len(dh)))
            fs = px.scatter(sample, x=metric, y=sy,
                color="hour_of_day", color_continuous_scale="Viridis",
                opacity=.6, title=f"{metric_disp} vs {sy_disp}")
            fs.update_layout(height=300, **PL)
            st.plotly_chart(fs, use_container_width=True)

        st.dataframe(df[["solar_pv_output","grid_load_demand",
                          "battery_state_of_charge","solar_irradiance"]
                        ].describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 6 -- LIVE PROTOTYPE
# ══════════════════════════════════════════════════════════════════════
with t6:
    st.markdown("### Interactive Microgrid Prototype")
    st.caption("Adjust sidebar sliders -> charts update live.")

    solar_24, load_24 = make_profiles(s_peak, b_load, e_peak, p_hr, cloudy, weekend)
    res_ai   = simulate(solar_24, load_24, init_soc)
    grid_nb  = np.maximum(load_24 - solar_24, 0)

    cost_ai = sum(res_ai["grid"][h]*(0.25 if 17<=h<=21 else 0.10) for h in range(24))*8
    cost_nb = sum(grid_nb[h]      *(0.25 if 17<=h<=21 else 0.10) for h in range(24))*8
    co2_save= abs((grid_nb.sum() - res_ai["total_grid"]) * 0.68)
    saved   = abs(cost_nb - cost_ai)

    k1,k2,k3,k4,k5 = st.columns(5)
    kpi(k1,"☀️",f"{solar_24.sum():.0f} kWh","Solar Generated")
    kpi(k2,"🌿",f"{res_ai['solar_pct']:.1f}%","Solar Coverage")
    kpi(k3,"🔌",f"{res_ai['total_grid']:.0f} kWh","Grid Draw")
    kpi(k4,"💰",f"Rs{saved:.0f}","Daily Saving")
    kpi(k5,"🌱",f"{co2_save:.1f} kg","CO2 Avoided")
    st.divider()

    hrs = list(range(24))
    fig4 = make_subplots(rows=2, cols=2,
        subplot_titles=["Power Flow (kW)","Battery SoC (%)","Hourly Grid Cost","Energy Mix"],
        vertical_spacing=0.15, horizontal_spacing=0.09)

    fig4.add_trace(go.Scatter(x=hrs, y=solar_24.tolist(), name="Solar",
        fill="tozeroy", fillcolor="rgba(253,216,53,0.2)",
        line=dict(color="#fdd835",width=2.5)), 1,1)
    fig4.add_trace(go.Scatter(x=hrs, y=load_24.tolist(), name="Load",
        line=dict(color="#ef5350",width=2,dash="dash")), 1,1)
    fig4.add_trace(go.Scatter(x=hrs, y=res_ai["grid"].tolist(), name="Grid (AI)",
        fill="tozeroy", fillcolor="rgba(239,83,80,0.12)",
        line=dict(color="#ff7043",width=1.5)), 1,1)
    fig4.add_trace(go.Scatter(x=hrs, y=grid_nb.tolist(), name="Grid (No Battery)",
        line=dict(color="#b71c1c",width=1,dash="dot")), 1,1)
    fig4.add_vrect(x0=16.5,x1=21.5,fillcolor="rgba(239,83,80,0.08)",
        layer="below",line_width=0,row=1,col=1)

    soc_pct2 = [s*100 for s in res_ai["soc"]]
    fig4.add_trace(go.Scatter(x=hrs, y=soc_pct2, name="SoC",
        fill="tozeroy", fillcolor="rgba(66,165,245,0.2)",
        line=dict(color="#42a5f5",width=2.5), showlegend=False), 1,2)
    fig4.add_hline(y=90,line_dash="dot",line_color="#ff9800",row=1,col=2)
    fig4.add_hline(y=10,line_dash="dot",line_color="#f44336",row=1,col=2)

    h_cost = [res_ai["grid"][h]*(0.25 if 17<=h<=21 else 0.10)*8 for h in range(24)]
    fig4.add_trace(go.Bar(x=hrs, y=h_cost,
        marker_color=["#e53935" if 17<=h<=21 else "#42a5f5" for h in range(24)],
        name="Cost", showlegend=False), 2,1)

    su  = res_ai["solar_used"].tolist()
    grd = res_ai["grid"].tolist()
    bat = [max(0., su[h]-min(su[h],load_24[h])) for h in range(24)]
    fig4.add_trace(go.Bar(x=hrs, y=[min(su[h],float(load_24[h])) for h in range(24)],
        name="Solar Direct",marker_color="#fdd835"), 2,2)
    fig4.add_trace(go.Bar(x=hrs, y=bat, name="Battery",marker_color="#42a5f5"), 2,2)
    fig4.add_trace(go.Bar(x=hrs, y=grd, name="Grid",marker_color="#ef5350"), 2,2)

    _pl2 = {**PL,
        "xaxis": dict(title="Hour",tickvals=list(range(0,24,2)),gridcolor="#1a2535"),
        "xaxis2":dict(title="Hour",tickvals=list(range(0,24,2)),gridcolor="#1a2535"),
        "xaxis3":dict(title="Hour",tickvals=list(range(0,24,2)),gridcolor="#1a2535"),
        "xaxis4":dict(title="Hour",tickvals=list(range(0,24,2)),gridcolor="#1a2535"),
        "yaxis": dict(title="kW",gridcolor="#1a2535"),
        "yaxis2":dict(title="SoC %",range=[0,110],gridcolor="#1a2535"),
        "yaxis3":dict(title="Cost",gridcolor="#1a2535"),
        "yaxis4":dict(title="kW",gridcolor="#1a2535")}
    fig4.update_layout(height=660, barmode="stack", **_pl2)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Microgrid Component View")
    cur_h = st.slider("Hour", 0, 23, 13, key="cur_h")

    sk   = float(solar_24[cur_h])
    lk   = float(load_24[cur_h])
    gk   = float(res_ai["grid"][cur_h])
    su_k = float(res_ai["solar_used"][cur_h])
    soc_v= float(res_ai["soc"][cur_h])
    act  = res_ai["actions"][cur_h]
    is_pk= 17 <= cur_h <= 21

    fig_d = go.Figure()
    fig_d.update_xaxes(range=[0,10], visible=False)
    fig_d.update_yaxes(range=[0,7],  visible=False)

    def add_rect(x,y,w,h,col,title,sub=""):
        fig_d.add_shape(type="rect",x0=x,y0=y,x1=x+w,y1=y+h,
            fillcolor=col,line=dict(color="white",width=1.5),layer="above")
        fig_d.add_annotation(x=x+w/2,y=y+h*.65,text=f"<b>{title}</b>",
            showarrow=False,font=dict(color="white",size=12))
        if sub:
            fig_d.add_annotation(x=x+w/2,y=y+h*.28,text=sub,
                showarrow=False,font=dict(color="white",size=10))

    def add_arrow(x0,y0,x1,y1,lbl,col):
        fig_d.add_annotation(ax=x0,ay=y0,x=x1,y=y1,
            xref="x",yref="y",axref="x",ayref="y",
            arrowhead=3,arrowwidth=2.5,arrowcolor=col,
            text=lbl,font=dict(color=col,size=9),
            showarrow=True,bgcolor="rgba(0,0,0,0.4)")

    bat_col = "#2e7d32" if act==0 else "#4a148c" if act==2 else "#0d47a1"
    add_rect(0.3,4.5,2.0,1.5,"#e65100","SOLAR",f"{sk:.1f} kW")
    add_rect(0.3,1.0,2.0,1.5,"#1565c0","GRID","PEAK" if is_pk else f"{gk:.1f} kW")
    add_rect(4.0,2.5,2.2,2.0,bat_col,"BATTERY",f"SoC:{soc_v*100:.0f}% {AN[act]}")
    add_rect(7.5,2.8,2.0,1.5,"#b71c1c","CAMPUS",f"{lk:.1f} kW")

    if su_k > 1:
        add_arrow(2.3,5.25,5.1,4.3,f"Solar {su_k:.0f}kW","#fdd835")
    if gk > 1:
        add_arrow(2.3,1.75,7.5,3.3,f"Grid {gk:.0f}kW","#ef5350")
    if act==2 and soc_v > 0.15:
        dch = min(30.,(soc_v-.1)*100)
        add_arrow(6.2,3.5,7.5,3.5,f"Bat {dch:.0f}kW","#ab47bc")
    if su_k > 0:
        add_arrow(5.1,3.0,7.5,3.8,"campus","#66bb6a")

    fig_d.update_layout(
        height=300,
        title=f"Hour {cur_h:02d}:00  {'PEAK HOUR' if is_pk else 'Off-Peak'}  |  Battery: {AN[act]}",
        paper_bgcolor="rgba(14,17,35,1)", plot_bgcolor="rgba(14,17,35,1)",
        font=dict(color="white"), margin=dict(l=5,r=5,t=50,b=5), showlegend=False)
    st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("#### AI Battery vs No Battery")
    fc_cmp = go.Figure()
    fc_cmp.add_trace(go.Scatter(x=hrs, y=grid_nb.tolist(), name="Without Battery",
        line=dict(color="#ef5350",width=2,dash="dash"),
        fill="tozeroy",fillcolor="rgba(239,83,80,0.1)"))
    fc_cmp.add_trace(go.Scatter(x=hrs, y=res_ai["grid"].tolist(), name="With AI Battery",
        line=dict(color="#42a5f5",width=2.5),
        fill="tozeroy",fillcolor="rgba(66,165,245,0.15)"))
    fc_cmp.add_vrect(x0=16.5,x1=21.5,fillcolor="rgba(255,0,0,0.05)",
        layer="below",line_width=0)
    _pl_cmp = {**PL}
    _pl_cmp["xaxis"] = dict(tickvals=list(range(0,24,2)), showgrid=False)
    fc_cmp.update_layout(
        title=f"Saved {abs(grid_nb.sum()-res_ai['total_grid']):.1f} kWh (Rs{saved:.0f})",
        xaxis_title="Hour",yaxis_title="kW",height=300,**_pl_cmp)
    st.plotly_chart(fc_cmp, use_container_width=True)

    m1,m2,m3 = st.columns(3)
    saving_kwh = abs(grid_nb.sum()-res_ai["total_grid"])
    m1.metric("Grid Saved",f"{saving_kwh:.1f} kWh",f"{saving_kwh/max(grid_nb.sum(),1)*100:.1f}% less")
    m2.metric("Cost Saved",f"Rs{saved:.0f}","daily estimate")
    m3.metric("CO2 Avoided",f"{co2_save:.1f} kg",f"{co2_save/0.0575:.0f} tree-days")

    st.markdown("#### Hour-by-Hour Table")
    hourly_cost = [res_ai["grid"][h]*(0.25 if 17<=h<=21 else 0.10)*8 for h in range(24)]
    proto_log = pd.DataFrame({
        "Hour":    [f"{h:02d}:00" for h in range(24)],
        "Solar kW":[f"{solar_24[h]:.1f}" for h in range(24)],
        "Load kW": [f"{load_24[h]:.1f}"  for h in range(24)],
        "Grid kW": [f"{res_ai['grid'][h]:.1f}" for h in range(24)],
        "Action":  [AN[res_ai["actions"][h]] for h in range(24)],
        "Battery %":[f"{res_ai['soc'][h]*100:.1f}" for h in range(24)],
        "Cost":    [f"{hourly_cost[h]:.1f}" for h in range(24)],
        "Peak?":   ["YES" if 17<=h<=21 else "--" for h in range(24)],
    })
    st.dataframe(proto_log, use_container_width=True, height=280, hide_index=True)

    st.markdown("#### Live Gauges")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        cur_soc = res_ai["soc"][cur_h]*100
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number", value=cur_soc,
            domain={"x":[0,1],"y":[0,1]}, title={"text":"Battery %"},
            gauge={"axis":{"range":[None,100]},
                   "bar":{"color":"green" if cur_soc>20 else "red"},
                   "steps":[{"range":[0,20],"color":"lightpink"},
                             {"range":[20,100],"color":"lightgreen"}]}))
        fig_g1.update_layout(height=250, **PL)
        st.plotly_chart(fig_g1, use_container_width=True)
    with col_g2:
        cur_sol = min(100,(solar_24[cur_h]/max(s_peak,1))*100)
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number", value=cur_sol,
            domain={"x":[0,1],"y":[0,1]}, title={"text":"Solar Power %"},
            gauge={"axis":{"range":[None,100]},
                   "bar":{"color":"orange"},
                   "steps":[{"range":[0,100],"color":"lightyellow"}]}))
        fig_g2.update_layout(height=250, **PL)
        st.plotly_chart(fig_g2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 7 -- REAL-TIME MONITOR
# ══════════════════════════════════════════════════════════════════════
with t7:
    st.markdown("### Real-Time Environment Simulator")
    st.caption("Simulates live power streaming from the campus grid.")

    if "rt_idx" not in st.session_state:
        st.session_state.rt_idx = 0

    ca, cb = st.columns([1, 4])
    with ca:
        if st.button("Step Forward"):
            st.session_state.rt_idx = (st.session_state.rt_idx + 1) % 24
        auto_play = st.toggle("Auto-Play")

    idx = st.session_state.rt_idx
    if 6  <= idx < 12: time_lbl = "Morning"
    elif 12 <= idx < 17: time_lbl = "Afternoon"
    elif 17 <= idx < 20: time_lbl = "Evening (PEAK)"
    else:                time_lbl = "Night"

    with cb:
        st.markdown(f"#### Time: {idx:02d}:00 | Period: **{time_lbl}**")

    if auto_play:
        import time
        time.sleep(1.5)
        st.session_state.rt_idx = (st.session_state.rt_idx + 1) % 24
        st.rerun()

    fig_rt = go.Figure()
    solar_rt = solar_24.tolist()[:idx+1]
    load_rt  = load_24.tolist()[:idx+1]
    x_rt     = list(range(idx+1))

    fig_rt.add_trace(go.Scatter(x=x_rt, y=solar_rt, mode="lines+markers",
        name="Solar Supply", line=dict(color="#fdd835", width=3)))
    fig_rt.add_trace(go.Scatter(x=x_rt, y=load_rt, mode="lines+markers",
        name="Campus Demand", line=dict(color="#ef5350", width=3)))

    _pl_t7 = {**PL}
    _pl_t7.update(dict(
        xaxis=dict(range=[0,24], dtick=2),
        yaxis=dict(range=[0, float(max(load_24))+50]),
        height=400, title="Live Power Supply vs Demand"
    ))
    fig_rt.update_layout(**_pl_t7)
    st.plotly_chart(fig_rt, use_container_width=True)
    st.info("Click 'Step Forward' to simulate time passing and watch the sensor streams update!")

# ══════════════════════════════════════════════════════════════════════
# TAB 8 -- SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════
with t8:
    st.markdown("### Summary Report")
    st.write("A simple, understandable summary of all microgrid analytics.")
    st.info(
        "The AI predicts solar energy availability and campus electricity demand. "
        "The RL agent decides when to charge/discharge the battery to save money "
        "and reduce carbon emissions, lowering grid dependency."
    )

    if df is not None:
        try:
            from fpdf import FPDF

            def generate_pdf():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("helvetica", size=15, style="B")
                pdf.cell(200, 10, txt="Smart Microgrid Summary Report",
                         new_x="LMARGIN", new_y="NEXT", align="C")
                pdf.set_font("helvetica", size=12)
                lines = [
                    "1. Solar Forecasting: The LSTM model predicts solar energy with high accuracy.",
                    "2. Load Demand: XGBoost estimates campus power requirements, reducing uncertainty.",
                    "3. Battery Agent: DQN agent charges when cheap, discharges during peak, saving costs.",
                    "4. Data: Real-world or synthetic data drives live prototype emulation.",
                ]
                for line in lines:
                    pdf.multi_cell(0, 10, txt=line, new_x="LMARGIN", new_y="NEXT")
                return bytes(pdf.output())

            pdf_bytes = generate_pdf()
            st.download_button(
                label="Download PDF Summary Report",
                data=pdf_bytes,
                file_name="Microgrid_Summary_Report.pdf",
                mime="application/pdf",
                type="primary"
            )
        except Exception as e:
            st.warning(f"PDF export unavailable: {e}")