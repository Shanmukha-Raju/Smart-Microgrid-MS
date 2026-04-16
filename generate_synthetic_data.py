import pandas as pd
import numpy as np
import os

def create_synthetic_dataset(output_path="data/synthetic_dataset.csv", num_days=90):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    today = pd.Timestamp.now().normalize()
    start_time = today - pd.Timedelta(days=num_days)
    periods = num_days * 24
    
    timestamps = pd.date_range(start_time, periods=periods, freq="h")
    df = pd.DataFrame({"timestamp": timestamps})
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    # Generative dummy patterns
    df["solar_pv_output"] = np.clip(np.sin((df["hour_of_day"] - 6) * np.pi / 12) * 80, 0, 1) + np.random.normal(0, 5, periods)
    df["solar_pv_output"] = np.clip(df["solar_pv_output"], 0, None)
    df["predicted_solar_pv_output"] = df["solar_pv_output"] + np.random.normal(0, 2, periods)
    
    df["grid_load_demand"] = 200 + 50 * np.sin((df["hour_of_day"] - 12) * np.pi / 12) + np.random.normal(0, 10, periods)
    df["total_renewable_energy"] = df["solar_pv_output"] * 1.5
    df["total_predicted_energy"] = df["predicted_solar_pv_output"] * 1.5
    
    df["wind_power_output"] = np.clip(np.random.normal(50, 10, periods), 0, None)
    df["predicted_wind_power_output"] = df["wind_power_output"] + np.random.normal(0, 2, periods)
    
    df["solar_irradiance"] = df["solar_pv_output"] * 10
    df["wind_speed"] = np.clip(np.random.normal(10, 2, periods), 0, None)
    df["temperature"] = 20 + 10 * np.sin((df["hour_of_day"] - 8) * np.pi / 12) + np.random.normal(0, 2, periods)
    df["humidity"] = np.clip(50 + np.random.normal(0, 10, periods), 0, 100)
    df["atmospheric_pressure"] = 1013 + np.random.normal(0, 5, periods)
    df["frequency"] = 50 + np.random.normal(0, 0.05, periods)
    df["voltage"] = 230 + np.random.normal(0, 2, periods)
    
    # Random walk for state of charge
    df["battery_state_of_charge"] = 0.5 + np.cumsum(np.random.normal(0, 0.01, periods))
    df["battery_state_of_charge"] = np.clip(df["battery_state_of_charge"], 0.1, 0.9)
    
    df["battery_charging_rate"] = np.clip(np.random.normal(10, 1, periods), 0, None)
    df["battery_discharging_rate"] = np.clip(np.random.normal(10, 1, periods), 0, None)
    df["power_exchange"] = np.random.normal(0, 5, periods)
    
    # Save un-engineered features just like raw data
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset created successfully at {output_path}")

if __name__ == '__main__':
    create_synthetic_dataset("data/synthetic_dataset.csv", 90)
