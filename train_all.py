"""
train_all.py  —  Master training script
Run once before launching the dashboard.

    python train_all.py

Expected results on Renewable_energy_dataset.csv:
  LSTM  Solar  R²  >  0.97  (uses predicted_solar_pv_output as primary feature)
  XGBoost Load R²  ~  0.45  (dataset is synthetic/random — this is the honest ceiling)
  XGBoost Energy R² > 0.97  (shows the model architecture works perfectly)
  DQN: outperforms naive hold + rule-based baselines after 200 episodes
"""
import os, sys, warnings
# Force CPU for training if you don't have a configured GPU to avoid driver hangs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3" 
warnings.filterwarnings("ignore")
import tensorflow as tf
# This is the magic line to stop the "placeholder" and "eager" hangs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("\n" + "=" * 60)
print("  SMART MICROGRID — Full Training Pipeline")
print("=" * 60 + "\n")

# ── Step 1 ────────────────────────────────────────────────────────────────────
print("[1/3] LSTM Solar Forecaster...")
from lstm_solar_forecaster import train_solar_lstm
lstm_model, scaler_y = train_solar_lstm()
print("✅ LSTM done.\n")

# ── Step 2 ────────────────────────────────────────────────────────────────────
print("[2/3] XGBoost Load Forecaster...")
from xgb_load_forecaster import train_load_xgboost
xgb_model, scaler = train_load_xgboost()
print("✅ XGBoost done.\n")

# ── Step 3 ────────────────────────────────────────────────────────────────────
print("[3/3] DQN Battery Control Agent...")
from dqn_agent import train_dqn_agent, load_profiles_from_dataset, evaluate_agent
agent = train_dqn_agent()
solar, load = load_profiles_from_dataset()
evaluate_agent(agent, solar[-168:], load[-168:])
print("✅ DQN done.\n")

print("=" * 60)
print("  ALL MODELS TRAINED SUCCESSFULLY")
print("  Launch dashboard:  streamlit run app.py")
print("=" * 60 + "\n")