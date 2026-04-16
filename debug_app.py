"""
debug_app.py — Run this INSTEAD of app.py to see the exact crash error
Usage: python debug_app.py
"""
import sys, os, traceback

print("=" * 60)
print("MICROGRID DEBUG - Finding the crash")
print("=" * 60)

# Step 1
print("\n[1] Python version:", sys.version)

# Step 2
print("\n[2] Testing basic imports...")
for mod in ["os","sys","numpy","pandas","plotly","streamlit"]:
    try:
        __import__(mod)
        print(f"    ✅ {mod}")
    except Exception as e:
        print(f"    ❌ {mod}: {e}")

# Step 3
print("\n[3] Testing src imports...")
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

for mod in ["data_preprocessing","rl_microgrid_env"]:
    try:
        __import__(mod)
        print(f"    ✅ {mod}")
    except Exception as e:
        print(f"    ❌ {mod}: {e}")
        traceback.print_exc()

# Step 4
print("\n[4] Testing dataset load...")
try:
    from data_preprocessing import load_raw_data, engineer_features
    df = load_raw_data()
    df = engineer_features(df)
    print(f"    ✅ Dataset loaded: {df.shape}")
except Exception as e:
    print(f"    ❌ Dataset error: {e}")
    traceback.print_exc()

# Step 5
print("\n[5] Testing simulation...")
try:
    import numpy as np
    from rl_microgrid_env import MicrogridEnv
    env = MicrogridEnv(np.ones(24)*50, np.ones(24)*200)
    obs, _ = env.reset()
    for _ in range(24):
        obs, r, done, _, info = env.step(1)
    print(f"    ✅ Simulation works")
except Exception as e:
    print(f"    ❌ Simulation error: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("If you see this line, the crash is INSIDE app.py itself")
print("Share the full output above with the developer")
print("=" * 60)
input("\nPress ENTER to exit...")