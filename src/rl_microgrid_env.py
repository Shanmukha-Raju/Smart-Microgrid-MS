"""
rl_microgrid_env.py
====================
Custom Microgrid Battery Environment.
Works with OR without gymnasium installed — uses a lightweight fallback.
"""

import numpy as np

# ── Gymnasium optional import with fallback ───────────────────────────────────
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    # Minimal fallback so MicrogridEnv still works for the dashboard
    class _FakeSpace:
        def __init__(self, n=None, low=None, high=None, dtype=None):
            self.n = n
        def contains(self, x): return True
        def sample(self): return np.random.randint(3)
    class spaces:
        @staticmethod
        def Discrete(n): return _FakeSpace(n=n)
        @staticmethod
        def Box(low, high, dtype=None): return _FakeSpace()
    class gym:
        class Env:
            metadata = {}
            def __init_subclass__(cls, **kw): pass

# ─── PHYSICAL CONSTANTS ───────────────────────────────────────────────────────
BATTERY_CAPACITY_KWH  = 100.0
MAX_CHARGE_RATE_KW    = 30.0
MAX_DISCHARGE_RATE_KW = 30.0
BATTERY_EFFICIENCY    = 0.95
SOC_MIN               = 0.10
SOC_MAX               = 0.90
GRID_COST_PEAK        = 0.25
GRID_COST_OFFPEAK     = 0.10
SOLAR_VALUE           = 0.12
MAX_SOLAR_KW          = 100.0
MAX_LOAD_KW           = 500.0


class MicrogridEnv(gym.Env):
    """
    Smart Microgrid Battery Environment.
    Actions: 0=Charge, 1=Hold, 2=Discharge
    State:   7-dim vector [soc, solar_norm, load_norm, hour_sin, hour_cos, net_norm, is_peak]
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, solar_profile: np.ndarray, load_profile: np.ndarray,
                 initial_soc: float = 0.5, render_mode: str = None):
        if GYM_AVAILABLE:
            super().__init__()
        self.solar_profile  = np.asarray(solar_profile, dtype=np.float32)
        self.load_profile   = np.asarray(load_profile,  dtype=np.float32)
        self.initial_soc    = initial_soc
        self.T              = len(solar_profile)
        self.render_mode    = render_mode
        self.action_space   = spaces.Discrete(3)
        low  = np.array([0., 0., 0., -1., -1., -1., 0.], dtype=np.float32)
        high = np.array([1., 1., 1.,  1.,  1.,  1., 1.], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.t           = 0
        self.soc         = initial_soc
        self.episode_log = []

    def reset(self, seed=None, options=None):
        if GYM_AVAILABLE:
            super().reset(seed=seed)
        self.t           = 0
        self.soc         = self.initial_soc
        self.episode_log = []
        return self._get_obs(), {}

    def step(self, action: int):
        action = int(action)
        solar_kw = float(self.solar_profile[self.t])
        load_kw  = float(self.load_profile[self.t])
        hour     = self.t % 24
        grid_draw_kw, solar_used_kw, curtailed_kw = self._execute_action(action, solar_kw, load_kw)
        reward = self._compute_reward(grid_draw_kw, solar_used_kw, curtailed_kw, hour)
        self.episode_log.append({
            "t": self.t, "hour": hour, "action": action,
            "solar_kw": solar_kw, "load_kw": load_kw,
            "soc": self.soc, "grid_draw": grid_draw_kw,
            "solar_used": solar_used_kw, "reward": reward
        })
        self.t += 1
        done = self.t >= self.T
        return self._get_obs(), reward, done, False, {
            "grid_draw_kw":  grid_draw_kw,
            "solar_used_kw": solar_used_kw,
            "soc":           self.soc,
            "hour":          hour,
        }

    def _execute_action(self, action, solar_kw, load_kw):
        if action == 0:   # CHARGE
            headroom       = (SOC_MAX - self.soc) * BATTERY_CAPACITY_KWH
            max_charge     = min(MAX_CHARGE_RATE_KW, headroom / BATTERY_EFFICIENCY)
            solar_for_chg  = min(solar_kw, max_charge)
            grid_for_chg   = max(0., max_charge - solar_for_chg)
            self.soc      += (solar_for_chg + grid_for_chg) * BATTERY_EFFICIENCY / BATTERY_CAPACITY_KWH
            self.soc       = float(np.clip(self.soc, 0., 1.))
            grid_draw      = load_kw + grid_for_chg
            solar_used     = solar_for_chg
            curtailed      = max(0., solar_kw - solar_for_chg - load_kw)
        elif action == 2:  # DISCHARGE
            available      = (self.soc - SOC_MIN) * BATTERY_CAPACITY_KWH
            discharge_kw   = min(MAX_DISCHARGE_RATE_KW, available, load_kw)
            self.soc      -= discharge_kw / BATTERY_CAPACITY_KWH
            self.soc       = float(np.clip(self.soc, 0., 1.))
            residual       = max(0., load_kw - discharge_kw - solar_kw)
            grid_draw      = residual
            solar_used     = min(solar_kw, max(0., load_kw - discharge_kw))
            curtailed      = max(0., solar_kw - solar_used)
        else:              # HOLD
            solar_used     = min(solar_kw, load_kw)
            grid_draw      = max(0., load_kw - solar_kw)
            curtailed      = max(0., solar_kw - load_kw)
        return grid_draw, solar_used, curtailed

    def _compute_reward(self, grid_draw_kw, solar_used_kw, curtailed_kw, hour) -> float:
        is_peak     = 17 <= hour <= 21
        solar_bonus = SOLAR_VALUE * solar_used_kw
        grid_rate   = GRID_COST_PEAK if is_peak else GRID_COST_OFFPEAK
        peak_factor = 2.5 if is_peak else 1.0
        peak_pen    = grid_rate * peak_factor * grid_draw_kw
        grid_stress = 0.005 * (grid_draw_kw ** 2) / MAX_LOAD_KW
        battery_abuse = 0.
        if self.soc < SOC_MIN + 0.05:
            battery_abuse = 2. * (SOC_MIN + 0.05 - self.soc) * 10
        elif self.soc > SOC_MAX - 0.05:
            battery_abuse = 2. * (self.soc - (SOC_MAX - 0.05)) * 10
        curtailment_pen = 0.02 * curtailed_kw
        return float(solar_bonus - peak_pen - grid_stress - battery_abuse - curtailment_pen)

    def _get_obs(self) -> np.ndarray:
        if self.t >= self.T:
            return np.zeros(7, dtype=np.float32)
        solar_kw = float(self.solar_profile[self.t])
        load_kw  = float(self.load_profile[self.t])
        hour     = self.t % 24
        return np.array([
            self.soc,
            solar_kw / MAX_SOLAR_KW,
            load_kw  / MAX_LOAD_KW,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            float(np.clip((solar_kw - load_kw) / MAX_LOAD_KW, -1, 1)),
            float(17 <= hour <= 21),
        ], dtype=np.float32)

    def render(self):
        if self.episode_log:
            last = self.episode_log[-1]
            names = ["CHARGE", "HOLD  ", "DISCH "]
            print(f"t={last['t']:3d} h={last['hour']:02d} "
                  f"Act={names[last['action']]} SoC={last['soc']:.2f} "
                  f"Solar={last['solar_kw']:5.1f} Load={last['load_kw']:5.1f} "
                  f"Grid={last['grid_draw']:5.1f} R={last['reward']:+.3f}")