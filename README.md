# Smart Microgrid Management System
### Predict-then-Optimize Framework | College Project (150 Marks)

---

## Project Overview
This system solves the **Duck Curve Problem** using a three-stage AI pipeline:
1. **LSTM** → Predicts Solar PV output (next 24 hrs)
2. **XGBoost** → Predicts Campus Grid Load Demand
3. **DQN (RL Agent)** → Optimizes Battery: Charge / Discharge / Hold

---

## File Structure
```
smart_microgrid/
├── data/
│   └── Renewable_energy_dataset.csv      # Raw dataset (place here)
├── models/
│   └── saved/                            # Auto-saved after training
│       ├── lstm_solar_model.keras
│       ├── xgb_load_model.pkl
│       ├── dqn_agent.keras
│       └── scaler_solar.pkl / scaler_load.pkl
├── src/
│   ├── data_preprocessing.py             # Shared data pipeline
│   ├── lstm_solar_forecaster.py          # LSTM training script
│   ├── xgb_load_forecaster.py            # XGBoost training script
│   ├── rl_microgrid_env.py               # Custom Gym Environment
│   └── dqn_agent.py                      # DQN Agent + Training
├── app.py                                # Streamlit Dashboard
├── requirements.txt
└── README.md
```

---

## Setup & Run
```bash
pip install -r requirements.txt

# Step 1: Train Solar LSTM
python src/lstm_solar_forecaster.py

# Step 2: Train Load XGBoost
python src/xgb_load_forecaster.py

# Step 3: Train RL Agent
python src/dqn_agent.py

# Step 4: Launch Dashboard
streamlit run app.py
```
