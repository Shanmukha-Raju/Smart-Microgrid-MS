"""
uncertainty_forecaster.py  —  Advanced Feature 3: Uncertainty-Aware Forecasting
================================================================================
Replaces point-estimate LSTM with a PROBABILISTIC forecast that outputs
prediction intervals (10th, 50th, 90th percentile).

Method: Monte Carlo Dropout (MC Dropout)
  - Train LSTM with Dropout layers (same as existing model)
  - At INFERENCE time, keep Dropout ACTIVE (training=True)
  - Run N forward passes → get distribution over predictions
  - 10th/90th percentile = confidence interval

Why it matters for the RL agent:
  - Conservative planning: feed 90th-percentile LOAD + 10th-percentile SOLAR
    to the RL agent → it plans for the WORST CASE (maximally cautious)
  - This is Robust Optimization Under Uncertainty (used in energy industry)

Usage:
    model = build_mc_dropout_lstm(seq_len, n_features)
    mean, lower, upper = mc_predict(model, sequence, n_samples=50)
"""

import os, sys
import numpy as np
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import (
    load_raw_data, prepare_solar_sequences, time_split,
    SEQ_LEN, MODEL_DIR, SOLAR_FEATURES
)

MC_MODEL_PATH = os.path.join(MODEL_DIR, "mc_dropout_lstm.keras")
N_MC_SAMPLES  = 50   # number of stochastic forward passes


# ─── MODEL WITH ALWAYS-ON DROPOUT ────────────────────────────────────────────
class MCDropout(layers.Dropout):
    """Dropout that stays ON during inference (Monte Carlo Dropout)."""
    def call(self, inputs):
        return super().call(inputs, training=True)   # always active


def build_mc_dropout_lstm(seq_len: int, n_features: int) -> keras.Model:
    """
    Same architecture as production LSTM but with MCDropout layers.
    The stochasticity during inference provides uncertainty estimates.
    """
    inp = keras.Input(shape=(seq_len, n_features), name="solar_sequence")
    x   = layers.BatchNormalization()(inp)

    x   = layers.LSTM(128, return_sequences=True)(x)
    x   = MCDropout(0.2)(x)                        # ← always-on dropout
    x   = layers.LSTM(64,  return_sequences=False)(x)
    x   = MCDropout(0.2)(x)

    # Skip connection
    last   = layers.Lambda(lambda t: t[:, -1, :])(inp)
    skip   = layers.Dense(64, activation="relu")(last)
    lstm_d = layers.Dense(64, activation="relu")(x)
    merged = layers.Add()([lstm_d, skip])
    merged = layers.Dense(32, activation="relu")(merged)
    out    = layers.Dense(1, activation="sigmoid")(merged)

    model  = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss="mse", metrics=["mae"])
    return model


# ─── MC INFERENCE ────────────────────────────────────────────────────────────
def mc_predict(
    model,
    sequence:   np.ndarray,
    scaler_y,
    n_samples:  int = N_MC_SAMPLES,
    n_steps:    int = 24,
) -> tuple:
    """
    Run N stochastic forward passes and return:
        mean_pred  : (n_steps,)  — expected forecast
        lower_10   : (n_steps,)  — 10th percentile (optimistic solar)
        upper_90   : (n_steps,)  — 90th percentile (pessimistic solar)

    For ROBUST RL: pass lower_10 (conservative solar) to the agent so it
    plans to charge MORE during the day as insurance against over-optimism.
    """
    all_preds = np.zeros((n_samples, n_steps), dtype=np.float32)

    for s in range(n_samples):
        seq = sequence.copy()
        for step in range(n_steps):
            inp   = seq[np.newaxis]
            p_sc  = model(inp, training=True).numpy()[0, 0]   # stochastic pass
            all_preds[s, step] = p_sc
            new_step    = seq[-1].copy()
            new_step[0] = p_sc
            seq = np.vstack([seq[1:], new_step])

    # Inverse-transform
    def inv(arr):
        return np.clip(
            scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten(),
            0, None
        )

    mean_pred = inv(all_preds.mean(axis=0))
    lower_10  = inv(np.percentile(all_preds, 10, axis=0))
    upper_90  = inv(np.percentile(all_preds, 90, axis=0))
    std_pred  = inv(all_preds.std(axis=0))

    return mean_pred, lower_10, upper_90, std_pred


# ─── TRAINING ────────────────────────────────────────────────────────────────
def train_mc_lstm():
    print("=" * 60)
    print("  MC-Dropout LSTM — Uncertainty-Aware Solar Forecasting")
    print("=" * 60)

    from tensorflow.keras import callbacks
    df                        = load_raw_data()
    X, y, scaler_X, scaler_y  = prepare_solar_sequences(df, SEQ_LEN)
    X_tr, X_te, y_tr, y_te   = time_split(X, y, test_ratio=0.2)

    model = build_mc_dropout_lstm(SEQ_LEN, X_tr.shape[2])
    model.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=7, min_lr=1e-6),
        callbacks.ModelCheckpoint(MC_MODEL_PATH, save_best_only=True, monitor="val_loss"),
    ]
    model.fit(X_tr, y_tr, validation_split=0.15, epochs=100, batch_size=32,
              callbacks=cbs, verbose=1)

    # Evaluate with uncertainty
    print("\nRunning MC inference on test set (first 24 steps)...")
    mean_p, low_p, high_p, std_p = mc_predict(model, X_te[0], scaler_y, n_samples=50, n_steps=24)
    y_true = scaler_y.inverse_transform(y_te[:24].reshape(-1, 1)).flatten()

    coverage = np.mean((y_true >= low_p) & (y_true <= high_p)) * 100
    print(f"\n  Mean prediction ±std:  {mean_p.mean():.1f} ± {std_p.mean():.1f} kW")
    print(f"  80% interval coverage: {coverage:.1f}%  (target ≥ 80%)")
    print(f"  MC model saved → {MC_MODEL_PATH}")
    return model, scaler_y


# ─── ROBUST RL STATE BUILDER ─────────────────────────────────────────────────
def build_robust_state(
    mean_solar:   float,
    lower_solar:  float,
    mean_load:    float,
    upper_load:   float,
    soc:          float,
    hour:         int,
) -> np.ndarray:
    """
    Build a CONSERVATIVE 7-dim RL state using worst-case forecast bounds.
    Use this instead of point estimates to make the agent plan robustly.

    Args:
        mean_solar  : Expected solar output (kW)
        lower_solar : 10th percentile solar (kW) — worst case for solar
        mean_load   : Expected load (kW)
        upper_load  : 90th percentile load (kW) — worst case for load
        soc         : Current battery state of charge [0,1]
        hour        : Current hour of day
    """
    # Agent sees CONSERVATIVE solar and HIGH load → will pre-charge more
    conservative_solar = lower_solar
    pessimistic_load   = upper_load
    net_norm           = (conservative_solar - pessimistic_load) / 500.0
    return np.array([
        soc,
        conservative_solar / 100.0,
        pessimistic_load   / 500.0,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.clip(net_norm, -1, 1),
        float(17 <= hour <= 21),
    ], dtype=np.float32)
if __name__ == "__main__":
    train_mc_lstm()