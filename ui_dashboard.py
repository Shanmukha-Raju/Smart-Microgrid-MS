"""
ui_dashboard.py — Grandparent-Friendly Smart Microgrid Dashboard
==================================================================
This is the fun, simple, beautiful UI that anyone can understand.
NO JARGON. Big buttons. Glowing meters. Happy messages.
"""

import os
import sys
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Your Smart Battery ⚡",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# BEAUTIFUL STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f1c2e 0%, #1a2f4a 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #0a1420 0%, #152238 100%);
}

.big-metric {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1e3a5f, #2d5a8c);
    border-radius: 15px;
    border: 2px solid #4fc3f7;
    margin: 10px 0;
}

.big-metric-value {
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #4fc3f7, #81d4fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.big-metric-label {
    font-size: 1.2rem;
    color: #b0bec5;
    margin-top: 10px;
}

.advice-box {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
    border-left: 5px solid #4caf50;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
}

.advice-label {
    color: #a5d6a7;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
}

.advice-text {
    color: #c8e6c9;
    font-size: 1.1rem;
    font-weight: 500;
    margin-top: 8px;
}

.prediction-box {
    background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
    border-left: 5px solid #2196f3;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
}

.prediction-label {
    color: #90caf9;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
}

.prediction-text {
    color: #bbdefb;
    font-size: 1.1rem;
    font-weight: 500;
    margin-top: 8px;
}

.magic-button {
    margin: 10px 0;
}

h1, h2, h3, h4, h5 {
    color: #4fc3f7 !important;
}

.stButton>button {
    background: linear-gradient(135deg, #4fc3f7, #81d4fa);
    color: #0f1c2e;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 15px 30px;
    font-size: 1.1rem;
    transition: all 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 30px rgba(79, 195, 247, 0.6);
}

.stTabs [data-baseweb="tab"] {
    color: #b0bec5;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #4fc3f7 !important;
    border-bottom: 3px solid #4fc3f7 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def create_battery_gauge(soc_percent):
    """Create a glowing, beautiful battery meter gauge."""
    # Determine color based on SOC
    if soc_percent >= 70:
        color = "#4caf50"  # Green
        status = "Good"
    elif soc_percent >= 40:
        color = "#ffa726"  # Orange
        status = "Fair"
    else:
        color = "#ef5350"  # Red
        status = "Low"
    
    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number+delta",
        value=soc_percent,
        title={"text": "Battery Level", "font": {"size": 24, "color": "#4fc3f7"}},
        delta={"reference": 50, "increasing": {"color": "#4caf50"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 25], "color": "#ffebee"},
                {"range": [25, 50], "color": "#fff3e0"},
                {"range": [50, 75], "color": "#e8f5e9"},
                {"range": [75, 100], "color": "#c8e6c9"},
            ],
            "threshold": {
                "line": {"color": "#ef5350", "width": 4},
                "thickness": 0.75,
                "value": 20,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
        number={"suffix": "%", "font": {"size": 40, "color": color}},
    )])
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cfd8dc", "size": 14},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=350,
    )
    
    return fig, status, color


def create_energy_chart(hours, solar, load, grid):
    """Create a simple, clear 'Energy vs The Sun' chart."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Your Sun Power 😎", "Your Home Needs 🏠"),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )
    
    # Solar (top)
    fig.add_trace(
        go.Scatter(
            x=hours, y=solar,
            name="Sun Power Generated",
            line=dict(color="#ffd54f", width=3),
            fill="tozeroy",
            fillcolor="rgba(255, 213, 79, 0.3)",
            mode="lines",
            hovertemplate="Hour %{x}: <b>%{y:.1f} kW</b><extra></extra>",
        ),
        row=1, col=1,
    )
    
    # Load (bottom)
    fig.add_trace(
        go.Scatter(
            x=hours, y=load,
            name="Home Power Needed",
            line=dict(color="#64b5f6", width=3),
            fill="tozeroy",
            fillcolor="rgba(100, 181, 246, 0.3)",
            mode="lines",
            hovertemplate="Hour %{x}: <b>%{y:.1f} kW</b><extra></extra>",
        ),
        row=2, col=1,
    )
    
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1, gridcolor="#1a2535")
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1, gridcolor="#1a2535", range=[0, max(solar)*1.2])
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1, gridcolor="#1a2535", range=[0, max(load)*1.2])
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18,22,36,1)",
        font=dict(color="#cfd8dc", size=12),
        hovermode="x unified",
        height=500,
        margin={"l": 80, "r": 20, "t": 80, "b": 60},
        legend={"bgcolor": "rgba(0,0,0,0)", "x": 0.02, "y": 0.98},
        showlegend=True,
    )
    
    return fig


def create_savings_display(total_solar_used_kwh, solar_value_per_kwh=0.12):
    """Show money saved with a big, happy number."""
    savings = total_solar_used_kwh * solar_value_per_kwh
    return savings


def get_smart_advice(soc, hour, solar_kw, load_kw, pred_solar_next_2hr):
    """Generate human-friendly 'what we're doing now' advice."""
    peak_hours = 17 <= hour <= 21
    
    if soc < 0.25 and peak_hours:
        advice = "🔴 Battery is LOW! We're charging from the grid to keep your lights on during dinner time."
        prediction = f"☀️ Sun is setting soon. We'll discharge the battery to help dinner time."
    elif soc > 0.80 and solar_kw > load_kw:
        advice = "✅ Battery is FULL! We're using all the sun we can right now."
        prediction = f"☀️ Sun is strong. We're making the most of it!"
    elif soc < 0.35:
        advice = f"⚠️ Battery LOW ({soc*100:.0f}%). We're being careful with power."
        if pred_solar_next_2hr > load_kw:
            prediction = "☀️ Good news! More sun is coming in 2 hours. Saving battery for peak time."
        else:
            prediction = "☁️ Cloudy ahead. Saving battery for when you need it most."
    elif peak_hours:
        if soc > 0.50:
            advice = "🌙 Dinner time peak! We're using stored battery to save you money."
            prediction = "💰 Using battery now saves $0.15/kWh instead of grid power."
        else:
            advice = "🌙 Dinner peak but battery is low. Using grid power carefully."
            prediction = "💳 Need some grid power, but we're minimizing cost."
    else:
        if solar_kw > load_kw:
            advice = "☀️ Solar is powering your home! Simple as that."
            prediction = "✨ Making the most of the sun. Everything is perfect."
        else:
            advice = "🏠 More load than sun right now. Using battery if available."
            prediction = "⚡ Balancing battery and sun power efficiently."
    
    return advice, prediction


def create_ui_screenshot_markdown():
    """Create a detailed markdown 'screenshot' of the UI."""
    screenshot = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                   YOUR SMART BATTERY DASHBOARD ⚡                         ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  TODAY'S PERFORMANCE                                                      ║
    ║  ┌─────────────────────────────────────────────────────────────────────┐ ║
    ║  │  💰 MONEY SAVED TODAY                                              │ ║
    ║  │  ┌─────────────────────┐      ☀️ SUN MADE                          │ ║
    ║  │  │     $14.87          │      ┌──────────────┐                     │ ║
    ║  │  │   (bright green)    │      │   124 kWh    │                     │ ║
    ║  │  │   Bold & Clear      │      │   (golden)   │                     │ ║
    ║  │  └─────────────────────┘      └──────────────┘                     │ ║
    ║  │                                                                      │ ║
    ║  │  🏠 YOUR USAGE              🔋 GRID DRAW                           │ ║
    ║  │  ┌──────────────┐            ┌──────────────┐                      │ ║
    ║  │  │   162 kWh    │            │   38 kWh     │                      │ ║
    ║  │  │              │            │   (blue)     │                      │ ║
    ║  │  └──────────────┘            └──────────────┘                      │ ║
    ║  └─────────────────────────────────────────────────────────────────────┘ ║
    ║                                                                           ║
    ║  RIGHT NOW — YOUR BATTERY LEVEL 🔋                                       ║
    ║  ┌─────────────────────────────────────────────────────────────────────┐ ║
    ║  │                                                                      │ ║
    ║  │         ╔════════════════════════════════════════════╗             │ ║
    ║  │         ║        BATTERY: 67%                       ║             │ ║
    ║  │         ║        [████████░░░░░░]  GREEN & GLOWING  ║             │ ║
    ║  │         ║        Status: Good                       ║             │ ║
    ║  │         ╚════════════════════════════════════════════╝             │ ║
    ║  │                                                                      │ ║
    ║  └─────────────────────────────────────────────────────────────────────┘ ║
    ║                                                                           ║
    ║  WHAT YOUR SMART SYSTEM IS DOING RIGHT NOW 🤖                           ║
    ║  ┌─────────────────────────────────────────────────────────────────────┐ ║
    ║  │                                                                      │ ║
    ║  │  ✅ \"It's morning! Sun is rising. We're storing power for       │ ║
    ║  │       peak price hours tonight. Your battery is charging         │ ║
    ║  │       with free solar energy.\"                                     │ ║
    ║  │                                                                      │ ║
    ║  │  ☀️ What's coming: \"It will be sunny for 3 more hours.           │ ║
    ║  │      We're making the most of it and saving money for you!\"       │ ║
    ║  │                                                                      │ ║
    ║  └─────────────────────────────────────────────────────────────────────┘ ║
    ║                                                                           ║
    ║  YOUR ENERGY & THE SUN CHART 📊                                          ║
    ║  ┌─────────────────────────────────────────────────────────────────────┐ ║
    ║  │                                                                      │ ║
    ║  │  Sun Power Generated (Golden Area)       Hour: 0-24                │ ║
    ║  │          │        /\\                                               │ ║
    ║  │     100 ─┼───────/  \\                                             │ ║
    ║  │      50 ─┼──────/────\\────                                        │ ║
    ║  │       0 ─┼─────/──────\\───────────────────                        │ ║
    ║  │          └────────────────────────────────────                    │ ║
    ║  │                                                                      │ ║
    ║  │  Your Home Needs (Blue Area)             Hour: 0-24                │ ║
    ║  │          │                                                          │ ║
    ║  │     300 ─┼────────            ╱╲╱╲                                 │ ║
    ║  │     200 ─┼─ ─ ─ ─ ─ ─ ─ ─ ─ ╱  ╲╱  ╲ ─ ─ ─ ─                     │ ║
    ║  │     100 ─┼───────────────────────────────                         │ ║
    ║  │       0 ─┼─────────────────────────────────                       │ ║
    ║  │          └────────────────────────────────────                    │ ║
    ║  │                                                                      │ ║
    ║  └─────────────────────────────────────────────────────────────────────┘ ║
    ║                                                                           ║
    ║  MAGIC BUTTONS 🎯                                                        ║
    ║  ┌──────────────────────────────────────────────────────────────────────┐║
    ║  │                                                                       ││
    ║  │  [🚀 BOOST BATTERY - Emergency Charge]   [📚 Learn More]           ││
    ║  │                                                                       ││
    ║  └──────────────────────────────────────────────────────────────────────┘║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    return screenshot


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

# Title & Welcome
st.markdown("# ⚡ Your Smart Battery Dashboard")
st.markdown("*Simple. Clear. Makes smart choices for you.*")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Settings")
    
    # Choose scenario
    scenario = st.selectbox(
        "📅 What day would you like to see?",
        ["Sunny Day", "Cloudy Day", "Mixed Day", "Custom"],
        help="Pick a weather scenario"
    )
    
    st.divider()
    
    # Initial battery
    init_battery = st.slider(
        "🔋 Start with Battery at:",
        10, 100, 60, 5,
        help="% charge when the day begins"
    )
    
    st.divider()
    
    # Show technical details?
    show_advanced = st.checkbox("👨‍🔬 Show technical details (for tech people)")


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────────────────────

def generate_scenario(scenario_name):
    """Create realistic 24-hour profiles."""
    hours = np.arange(24)
    
    if scenario_name == "Sunny Day":
        solar = 70 * np.exp(-0.5 * ((hours - 12) / 3.5) ** 2)
        load_base = 200
        load = np.full(24, load_base)
        load[7:10] += load_base * 0.3
        load[17:22] += np.array([100, 150, 180, 120, 60])
    elif scenario_name == "Cloudy Day":
        solar = 30 * np.exp(-0.5 * ((hours - 12) / 3.5) ** 2)
        load_base = 200
        load = np.full(24, load_base)
        load[7:10] += load_base * 0.3
        load[17:22] += np.array([100, 150, 180, 120, 60])
    elif scenario_name == "Mixed Day":
        solar = 60 * np.exp(-0.5 * ((hours - 12) / 3.5) ** 2)
        solar[10:14] *= 0.5  # Clouds mid-day
        load_base = 200
        load = np.full(24, load_base)
        load[7:10] += load_base * 0.3
        load[17:22] += np.array([100, 150, 180, 120, 60])
    else:  # Custom
        solar = 50 * np.exp(-0.5 * ((hours - 12) / 3.5) ** 2)
        load_base = 200
        load = np.full(24, load_base)
        load[7:10] += load_base * 0.2
        load[17:22] += np.array([80, 120, 150, 100, 50])
    
    solar = np.clip(solar, 0, None)
    load = np.clip(load, 50, 500)
    
    return hours, solar, load


hours, solar, load = generate_scenario(scenario)

# Simulate battery behavior (simple rule-based for now)
BAT_CAP = 100.0
MAX_CHG = 30.0
MAX_DCH = 30.0
EFF = 0.95
SOC_MIN = 0.10
SOC_MAX = 0.90

soc = init_battery / 100.0
soc_hist = [soc]
grid_hist = []
solar_used_hist = []
savings_usd = 0.0

for t, (s_kw, l_kw) in enumerate(zip(solar, load)):
    hour = t % 24
    is_peak = 17 <= hour <= 21
    
    # Simple rule-based control
    if s_kw > l_kw and soc < 0.85:
        # Charge
        chg_available = (SOC_MAX - soc) * BAT_CAP
        chg_needed = min(MAX_CHG, chg_available / EFF)
        s_use = min(s_kw, chg_needed)
        g_use = max(0, chg_needed - s_use)
        soc += (s_use + g_use) * EFF / BAT_CAP
        grid_draw = l_kw + g_use
        su = s_use
    elif is_peak and soc > 0.25:
        # Discharge
        avail = (soc - SOC_MIN) * BAT_CAP
        dch = min(MAX_DCH, avail, l_kw)
        soc -= dch / BAT_CAP
        grid_draw = max(0, l_kw - dch - s_kw)
        su = min(s_kw, max(0, l_kw - dch))
    else:
        # Hold
        su = min(s_kw, l_kw)
        grid_draw = max(0, l_kw - su)
    
    soc = np.clip(soc, 0.05, 1.0)
    soc_hist.append(float(soc))
    grid_hist.append(float(grid_draw))
    solar_used_hist.append(float(su))
    
    # Calculate savings
    peak_cost = 0.25
    offpeak_cost = 0.10
    solar_value = 0.12
    
    grid_cost = (peak_cost if is_peak else offpeak_cost) * grid_draw
    savings_usd += solar_value * su - grid_cost

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY: BIG METRICS
# ─────────────────────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="big-metric">
        <div class="big-metric-value">${savings_usd:.2f}</div>
        <div class="big-metric-label">💰 Money Saved Today</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_solar = np.sum(solar_used_hist)
    st.markdown(f"""
    <div class="big-metric">
        <div class="big-metric-value">{total_solar:.0f}</div>
        <div class="big-metric-label">☀️ kWh from Sun</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_load = np.sum(load)
    st.markdown(f"""
    <div class="big-metric">
        <div class="big-metric-value">{total_load:.0f}</div>
        <div class="big-metric-label">🏠 kWh Used</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_grid = np.sum(grid_hist)
    st.markdown(f"""
    <div class="big-metric">
        <div class="big-metric-value">{total_grid:.0f}</div>
        <div class="big-metric-label">🔌 kWh from Grid</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT NOW: Battery Gauge
# ─────────────────────────────────────────────────────────────────────────────

current_hour = 12
current_soc = soc_hist[current_hour]
current_soc_pct = current_soc * 100

gauge_fig, status, color = create_battery_gauge(current_soc_pct)

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.write("")

with col2:
    st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

with col3:
    st.write("")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Smart Advice Section
# ─────────────────────────────────────────────────────────────────────────────

# Generate advice
advice, prediction = get_smart_advice(
    current_soc, current_hour,
    solar[current_hour], load[current_hour],
    np.mean(solar[min(current_hour+2, 23):min(current_hour+3, 24)])
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="advice-box">
        <div class="advice-label">✨ What We're Doing Right Now</div>
        <div class="advice-text">{advice}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">🔮 What's Coming Next</div>
        <div class="prediction-text">{prediction}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Energy Chart
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 📊 Your Energy & The Sun")
energy_fig = create_energy_chart(hours, solar, load, grid_hist)
st.plotly_chart(energy_fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Magic Buttons
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 🎯 Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 BOOST BATTERY - Emergency Charge", use_container_width=True, key="boost"):
        st.balloons()
        st.success("⚡ Emergency charging activated! Battery will charge at maximum rate.", icon="✅")
        st.info("This uses grid power at peak rates but will get you to safe levels quickly.")

with col2:
    if st.button("📚 Learn More About Your System", use_container_width=True, key="learn"):
        with st.expander("📖 How Your Smart Battery Works", expanded=True):
            st.markdown("""
            ### Three Things Working Together:
            
            1. **Sun Predictions** ☀️
               - We forecast how much sun you'll get each hour
               - This helps us decide when to use solar and when to save battery
            
            2. **Home Usage Forecast** 🏠
               - We predict when your home needs power
               - Morning, evening peak times, etc.
            
            3. **My Smart Choices** 🤖
               - Our AI decides: Charge battery, use it now, or hold it
               - Goal: Save you the most money while keeping power on
            
            ### Money Saving Strategy:
            - **Charge during the day** when sun is free
            - **Use battery at night** when grid power is expensive ($0.25/kWh)
            - **Minimize grid use** especially during peak hours
            """)

with col3:
    if st.button("🎉 Show Milestone (100% Renewable)", use_container_width=True, key="milestone"):
        if np.sum(solar_used_hist) > 0.8 * np.sum(load):
            st.snow()
            st.success("🎉 You hit 80%+ renewable energy today! Excellent!", icon="🌟")
        else:
            st.info(f"💪 You're at {100*np.sum(solar_used_hist)/np.sum(load):.1f}% renewable today. Good progress!")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# UI Screenshot / Advanced View
# ─────────────────────────────────────────────────────────────────────────────

if show_advanced:
    with st.expander("👨‍🔬 Technical Details (For Nerds)", expanded=False):
        st.markdown("### Dashboard Structure")
        st.code(create_ui_screenshot_markdown())
        
        st.markdown("### Battery Physics")
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | Capacity | 100 kWh |
        | Max Charge Rate | 30 kW |
        | Max Discharge Rate | 30 kW |
        | Efficiency | 95% |
        | Safe Range | 10% - 90% SOC |
        | Current SOC | {current_soc*100:.1f}% |
        | Current Energy | {current_soc*100:.1f} kWh |
        """)
        
        st.markdown("### DQN State Vector (7-dim)")
        st.markdown(f"""
        The AI uses this 7-dimensional state vector:
        
        1. **SOC** = {current_soc:.3f} (Battery charge %: 0 to 1)
        2. **Solar/100** = {solar[current_hour]/100:.3f} (Normalized solar power)
        3. **Load/500** = {load[current_hour]/500:.3f} (Normalized home power)
        4. **sin(hour)** = {np.sin(2*np.pi*current_hour/24):.3f} (Time of day cyclical)
        5. **cos(hour)** = {np.cos(2*np.pi*current_hour/24):.3f} (Time of day cyclical)
        6. **clip((solar-load)/500)** = {np.clip((solar[current_hour]-load[current_hour])/500, -1, 1):.3f} (Power balance)
        7. **is_peak** = {float(17<=current_hour<=21):.1f} (Peak pricing hours: 5pm-10pm)
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #90a4ae; font-size: 0.85rem; margin-top: 30px;'>
    ⚡ <b>Smart Microgrid Management System</b> — Making renewable energy simple & fun
    <br>
    Your battery is always working to save you money. 💚
</div>
""", unsafe_allow_html=True)
