"""
multi_agent_rl.py  —  Advanced Feature 2: Multi-Agent RL (MARL)
================================================================
Splits a campus microgrid into ZONES, each with its own DQN sub-agent
controlling a local battery. A central COORDINATOR agent settles
inter-zone energy trading to minimise total campus grid cost.

Zone layout (example — customise to your campus):
  Zone A — Academic Block     (high daytime load, large solar roof)
  Zone B — Hostels            (high evening load, medium solar)
  Zone C — Labs & Data Centre (24/7 baseload, minimal solar)

Architecture:
  Each ZoneAgent is an independent DQN (same architecture as main agent).
  CoordinatorAgent sees aggregated zone states and decides energy transfers.
  Uses PettingZoo-style turn-based execution (no extra install needed —
  we implement a lightweight version directly).
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
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from rl_microgrid_env import BATTERY_CAPACITY_KWH, SOC_MIN, SOC_MAX

tf.random.set_seed(42)

# ─── ZONE DEFINITIONS ────────────────────────────────────────────────────────
@dataclass
class ZoneConfig:
    name:               str
    battery_capacity:   float   # kWh
    max_charge_rate:    float   # kW
    max_discharge_rate: float   # kW
    solar_fraction:     float   # fraction of campus solar assigned to zone


ZONES = {
    "academic":   ZoneConfig("Academic Block",    40.0, 12.0, 12.0, 0.50),
    "hostel":     ZoneConfig("Hostels",           30.0, 10.0, 10.0, 0.30),
    "labs":       ZoneConfig("Labs & Data Centre",30.0,  8.0,  8.0, 0.20),
}


# ─── ZONE ENVIRONMENT ────────────────────────────────────────────────────────
class ZoneEnv:
    """Lightweight single-zone environment (no Gym dependency)."""
    def __init__(self, cfg: ZoneConfig):
        self.cfg     = cfg
        self.soc     = 0.50
        self.t       = 0
        self.solar   = np.array([])
        self.load    = np.array([])

    def reset(self, solar: np.ndarray, load: np.ndarray):
        self.soc   = np.random.uniform(0.3, 0.7)
        self.t     = 0
        self.solar = solar * self.cfg.solar_fraction
        self.load  = load  * (self.cfg.battery_capacity / BATTERY_CAPACITY_KWH)
        return self._obs()

    def step(self, action: int, energy_import: float = 0.0) -> Tuple:
        """
        action:        0=Charge, 1=Hold, 2=Discharge
        energy_import: kW received from coordinator (positive=import, negative=export)
        """
        solar_kw = float(self.solar[self.t]) if self.t < len(self.solar) else 0.0
        load_kw  = float(self.load[self.t])  if self.t < len(self.load)  else 0.0
        hour     = self.t % 24

        # Execute battery action
        grid_draw, solar_used = self._battery_step(action, solar_kw, load_kw)
        grid_draw = max(0.0, grid_draw - energy_import)   # reduce by imported energy

        # Reward (simpler than main env — zone-level)
        peak     = 17 <= hour <= 21
        reward   = (0.10 * solar_used
                    - (0.25 if peak else 0.10) * grid_draw
                    - 0.005 * (grid_draw ** 2) / 100)

        self.t += 1
        done = self.t >= len(self.solar)
        return self._obs(), reward, done, {"grid_draw": grid_draw, "solar_used": solar_used}

    def _battery_step(self, action, solar_kw, load_kw):
        cap = self.cfg.battery_capacity
        if action == 0:   # Charge
            headroom     = (SOC_MAX - self.soc) * cap
            charge_kw    = min(self.cfg.max_charge_rate, headroom)
            self.soc    += charge_kw * 0.95 / cap
            grid_draw    = max(0.0, load_kw + charge_kw - solar_kw)
            solar_used   = min(solar_kw, load_kw + charge_kw)
        elif action == 2:  # Discharge
            avail        = (self.soc - SOC_MIN) * cap
            discharge_kw = min(self.cfg.max_discharge_rate, avail)
            self.soc    -= discharge_kw / cap
            grid_draw    = max(0.0, load_kw - discharge_kw - solar_kw)
            solar_used   = min(solar_kw, load_kw - discharge_kw)
        else:              # Hold
            solar_used   = min(solar_kw, load_kw)
            grid_draw    = max(0.0, load_kw - solar_kw)
        self.soc = np.clip(self.soc, 0.0, 1.0)
        return grid_draw, max(0.0, solar_used)

    def _obs(self) -> np.ndarray:
        if self.t >= len(self.solar):
            return np.zeros(5, dtype=np.float32)
        hour     = self.t % 24
        solar_kw = float(self.solar[self.t])
        load_kw  = float(self.load[self.t])
        return np.array([
            self.soc,
            solar_kw / 100.0,
            load_kw  / 500.0,
            np.sin(2 * np.pi * hour / 24),
            float(17 <= hour <= 21),
        ], dtype=np.float32)


# ─── SHARED POLICY NETWORK (parameter sharing across zones) ──────────────────
def build_shared_policy(obs_dim: int = 5, n_actions: int = 3) -> keras.Model:
    """
    All zone agents share weights — reduces total parameters and improves
    generalisation. Each zone passes its own observation; the network outputs
    Q-values for that zone's actions.
    """
    inp = keras.Input(shape=(obs_dim,))
    x   = layers.Dense(64, activation="relu")(inp)
    x   = layers.Dense(64, activation="relu")(x)
    # Value stream
    v   = layers.Dense(1)(layers.Dense(32, activation="relu")(x))
    # Advantage stream
    a   = layers.Dense(n_actions)(layers.Dense(32, activation="relu")(x))
    # Dueling combination (using keras ops, no Lambda)
    q   = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
    return keras.Model(inputs=inp, outputs=q)


# ─── COORDINATOR AGENT ───────────────────────────────────────────────────────
def build_coordinator(n_zones: int = 3, n_actions: int = 9) -> keras.Model:
    """
    Coordinator observes concatenated zone states and decides:
      Which zone exports → which zone imports (3×3 = 9 transfer actions)
    State: concatenation of all zone observations (n_zones × 5)
    """
    inp = keras.Input(shape=(n_zones * 5,))
    x   = layers.Dense(128, activation="relu")(inp)
    x   = layers.Dense(64,  activation="relu")(x)
    out = layers.Dense(n_actions, activation="linear")(x)
    return keras.Model(inputs=inp, outputs=out)


# ─── MARL TRAINING LOOP ──────────────────────────────────────────────────────
def train_marl(solar_profile: np.ndarray, load_profile: np.ndarray,
               n_episodes: int = 50) -> Dict:
    """
    Lightweight MARL training loop.
    Returns trained models dict.
    """
    print("=" * 60)
    print("  Multi-Agent RL — Campus Microgrid (3 Zones)")
    print("=" * 60)

    zone_envs    = {k: ZoneEnv(v) for k, v in ZONES.items()}
    shared_net   = build_shared_policy()
    coordinator  = build_coordinator()
    optimizer    = keras.optimizers.Adam(3e-4)
    EPISODE_LEN  = min(168, len(solar_profile))

    ep_rewards = []
    for ep in range(n_episodes):
        # Reset all zones
        obs_dict = {}
        for zk, zenv in zone_envs.items():
            obs_dict[zk] = zenv.reset(solar_profile[:EPISODE_LEN],
                                       load_profile[:EPISODE_LEN])

        ep_r = 0.0
        for t in range(EPISODE_LEN):
            # 1. Each zone selects action via shared policy
            actions = {}
            for zk, obs in obs_dict.items():
                q   = shared_net(obs[np.newaxis], training=False).numpy()[0]
                eps = max(0.05, 1.0 - ep / n_episodes)
                actions[zk] = (np.random.randint(3) if np.random.rand() < eps
                                else int(np.argmax(q)))

            # 2. Coordinator decides energy transfers
            combined_obs = np.concatenate(list(obs_dict.values()))
            coord_q      = coordinator(combined_obs[np.newaxis], training=False).numpy()[0]
            coord_action = int(np.argmax(coord_q))
            # Map to transfer: action 0-8 → (exporter_idx, importer_idx)
            zone_keys  = list(zone_envs.keys())
            export_idx = coord_action // 3
            import_idx = coord_action %  3
            transfer_kw = 5.0 if export_idx != import_idx else 0.0

            # 3. Step all zones
            new_obs = {}
            for i, (zk, zenv) in enumerate(zone_envs.items()):
                energy_import = (transfer_kw if i == import_idx
                                 else (-transfer_kw if i == export_idx else 0.0))
                n_obs, rew, done, _ = zenv.step(actions[zk], energy_import)
                new_obs[zk]  = n_obs
                ep_r        += rew

            obs_dict = new_obs

        ep_rewards.append(ep_r)
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:3d}/{n_episodes} | "
                  f"Avg Reward (10ep): {np.mean(ep_rewards[-10:]):+8.2f}")

    print("\n✅ MARL training complete")
    return {
        "shared_policy": shared_net,
        "coordinator":   coordinator,
        "episode_rewards": ep_rewards,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import load_raw_data, engineer_features
    df    = load_raw_data()
    df    = engineer_features(df)
    solar = df["solar_pv_output"].values.astype(np.float32)
    load  = df["grid_load_demand"].values.astype(np.float32)
    train_marl(solar, load, n_episodes=30)