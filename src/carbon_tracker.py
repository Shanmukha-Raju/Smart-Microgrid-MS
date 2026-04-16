import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
# ─── CARBON INTENSITY CONSTANTS (gCO2eq / kWh) ───────────────────────────────
GRID_CARBON_INTENSITY   = 700.0   # India avg grid (gCO2/kWh)
SOLAR_CARBON_INTENSITY  =  20.0   # Solar PV lifecycle (gCO2/kWh)
BATTERY_CARBON_INTENSITY =  0.0   # At point of discharge (already charged)

# Time-of-day carbon multipliers (grid is dirtier at evening peak when
# coal plants ramp up to meet duck-curve demand)
HOURLY_CARBON_MULTIPLIER = {
    **{h: 0.85 for h in range(0, 6)},    # midnight–6am: cleaner (low demand)
    **{h: 1.00 for h in range(6, 17)},   # daytime: average
    **{h: 1.30 for h in range(17, 22)},  # peak 17–21h: dirtiest (coal ramp-up)
    **{h: 0.90 for h in range(22, 24)},  # late night: slightly cleaner
}


@dataclass
class CarbonLedger:
    """Accumulates per-step carbon metrics over an episode or real session."""
    total_grid_kwh:          float = 0.0
    total_solar_kwh:         float = 0.0
    total_battery_kwh:       float = 0.0
    total_co2_emitted_g:     float = 0.0
    total_co2_avoided_g:     float = 0.0
    hourly_log: List[dict]         = field(default_factory=list)

    def record_step(
        self,
        hour:            int,
        grid_draw_kw:    float,
        solar_used_kw:   float,
        battery_used_kw: float,
    ) -> float:
        """
        Record one timestep (1 hour assumed).
        Returns: carbon_penalty — a float to subtract from RL reward.
        """
        multiplier   = HOURLY_CARBON_MULTIPLIER.get(hour % 24, 1.0)
        effective_ci = GRID_CARBON_INTENSITY * multiplier

        # CO2 from grid this hour (grams)
        co2_grid    = grid_draw_kw    * effective_ci
        co2_solar   = solar_used_kw   * SOLAR_CARBON_INTENSITY
        co2_battery = battery_used_kw * BATTERY_CARBON_INTENSITY

        # CO2 avoided = what grid would have emitted if no solar/battery
        co2_avoided = (solar_used_kw + battery_used_kw) * effective_ci

        self.total_grid_kwh      += grid_draw_kw
        self.total_solar_kwh     += solar_used_kw
        self.total_battery_kwh   += battery_used_kw
        self.total_co2_emitted_g += co2_grid + co2_solar
        self.total_co2_avoided_g += co2_avoided

        self.hourly_log.append({
            "hour":           hour,
            "grid_draw_kw":   grid_draw_kw,
            "solar_used_kw":  solar_used_kw,
            "battery_kw":     battery_used_kw,
            "co2_emitted_g":  co2_grid,
            "co2_avoided_g":  co2_avoided,
            "carbon_intensity": effective_ci,
        })

        # RL carbon penalty: penalise high-carbon peak draws
        # Normalised to be small relative to main reward (~0.001–0.05 range)
        carbon_penalty = (co2_grid / 1000.0) * 0.005   # convert g→kg, scale
        return carbon_penalty

    # ── Summary ────────────────────────────────────────────────────────────────
    @property
    def co2_emitted_kg(self) -> float:
        return self.total_co2_emitted_g / 1000.0

    @property
    def co2_avoided_kg(self) -> float:
        return self.total_co2_avoided_g / 1000.0

    @property
    def renewable_fraction(self) -> float:
        total = self.total_grid_kwh + self.total_solar_kwh + self.total_battery_kwh
        return (self.total_solar_kwh + self.total_battery_kwh) / (total + 1e-9)

    @property
    def trees_equivalent(self) -> float:
        """1 tree absorbs ~21 kg CO2/year → daily = 21/365 ≈ 0.0575 kg/day"""
        return self.co2_avoided_kg / 0.0575

    def summary(self) -> dict:
        return {
            "CO2 Emitted (kg)":       round(self.co2_emitted_kg, 2),
            "CO2 Avoided (kg)":       round(self.co2_avoided_kg, 2),
            "Renewable Fraction (%)": round(self.renewable_fraction * 100, 1),
            "Trees Equivalent":       round(self.trees_equivalent, 1),
            "Solar Used (kWh)":       round(self.total_solar_kwh, 1),
            "Grid Used (kWh)":        round(self.total_grid_kwh, 1),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.hourly_log)

    def reset(self):
        self.__init__()


# ─── REWARD AUGMENTATION ─────────────────────────────────────────────────────
def carbon_aware_reward_bonus(
    hour:          int,
    solar_used_kw: float,
    grid_draw_kw:  float,
) -> float:
    """
    Drop-in addition to the RL reward function.
    Returns a small bonus/penalty based on carbon impact.
    Positive: using solar during high-carbon hours (saves more CO2)
    Negative: drawing grid during peak carbon hours
    """
    multiplier = HOURLY_CARBON_MULTIPLIER.get(hour % 24, 1.0)
    bonus  =  solar_used_kw * (multiplier - 1.0) * 0.002   # extra reward for solar at peak
    penalty = grid_draw_kw  * (multiplier - 1.0) * 0.003   # extra penalty for grid at peak
    return bonus - penalty


# ─── EXAMPLE USAGE ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ledger = CarbonLedger()
    # Simulate one day
    for hour in range(24):
        solar = 60.0 if 8 <= hour <= 16 else 0.0
        grid  = max(0, 200 - solar)
        batt  = 30.0 if 17 <= hour <= 21 else 0.0
        penalty = ledger.record_step(hour, grid, solar, batt)

    print("\n── Carbon Summary ──────────────────────────────")
    for k, v in ledger.summary().items():
        print(f"  {k:30s}: {v}")
    print("────────────────────────────────────────────────")