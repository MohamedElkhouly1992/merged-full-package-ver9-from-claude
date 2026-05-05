"""
HVAC ROM-Degradation Suite — Effects Comparison Runner
=======================================================
Purpose
-------
Run each publication-level coupled effect (and the BASE with all effects OFF)
across all four strategies S0–S3 and report the impact on the four PhD KPIs:

    1. Energy consumption  (Total Energy MWh)
    2. Comfort deviation   (Mean Comfort Deviation °C)
    3. Carbon emissions    (Total CO2 tonne)
    4. Degradation index   (Mean Degradation Index)

Speed notes
-----------
•  TIME_STEP_HOURS = 24.0   → one calculation step per day (fastest).
•  years = 5                → enough to see degradation trends, completes in ~10 s per run.
•  APO_POP = 8, APO_ITERS = 4 → S3 optimizer is light. For a publication run use
   APO_POP = 18, APO_ITERS = 10, years = 20.
•  axis_mode = "one_strategy" → runs all four strategies at once, one climate,
   one severity.  Change to "two_axis" or "three_axis" for the full matrix.

How to run
----------
    pip install -r requirements.txt
    python run_effects_comparison.py

Outputs (in ./effects_output/)
-------------------------------
•  effects_comparison_summary.csv   — KPI table: rows = effect variant × strategy
•  effects_delta_vs_base.csv        — % delta relative to BASE (all effects OFF)
•  figures/                         — bar charts for each of the 4 KPIs
•  <effect>_<strategy>_output/      — full CSVs/Excel/PDF for each run
"""

from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hvac_v3_engine import (
    BuildingSpec,
    HVACConfig,
    run_scenario_model,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Building & baseline HVAC configuration (New Mansoura University defaults)
# ──────────────────────────────────────────────────────────────────────────────

BLDG = BuildingSpec(
    building_type="Educational / University building",
    location="New Mansoura, Egypt",
    conditioned_area_m2=2000.0,
    floors=3,
    n_spaces=20,
    occupancy_density_p_m2=0.08,
    lighting_w_m2=10.0,
    equipment_w_m2=8.0,
    airflow_m3h_m2=4.0,
    infiltration_ach=0.5,
    sensible_w_per_person=75.0,
    cooling_intensity_w_m2=100.0,
    heating_intensity_w_m2=55.0,
    wall_u=0.6,
    roof_u=0.35,
    window_u=2.7,
    shgc=0.35,
    glazing_ratio=0.30,
)

# Base HVACConfig — all APPLY_* flags OFF, lightweight optimizer settings.
BASE_CFG = HVACConfig(
    years=5,                     # change to 20 for full PhD run
    hvac_system_type="Chiller_AHU",
    COP_COOL_NOM=4.5,
    COP_HEAT_NOM=3.2,
    CO2_FACTOR=0.536,            # Egypt grid (kgCO2/kWh)
    RF_STAR=2e-4,                # Kern-Seaton asymptote
    B_FOUL=0.015,                # fouling growth constant
    DUST_RATE=1.2,               # kg dust/day
    K_CLOG=6.0,
    DEG_TRIGGER=0.55,
    TIME_STEP_HOURS=24.0,        # daily step — fastest stable mode
    APO_POP=8,                   # S3 optimizer population (use 18 for publication)
    APO_ITERS=4,                 # S3 optimizer iterations (use 10 for publication)
    W_ENERGY=0.35,
    W_DEGRAD=0.25,
    W_COMFORT=0.25,
    W_CARBON=0.15,
    # ── HX hydraulic/thermal defaults (needed when APPLY_HX_* are ON) ──────
    HX_WATER_DP_CLEAN_KPA=35.0,
    HX_WATER_FLOW_NOM_M3H=0.0,   # 0 → derived from load inside engine
    HX_UA_LOSS_FACTOR=0.30,       # 30 % capacity loss at full degradation
    HX_AIR_FOULING_FACTOR=0.75,
    HX_PUMP_EFF=0.65,
    HX_CHW_DT_K=5.0,
    # ── Part-load COP curve (quadratic, typical centrifugal chiller) ────────
    PLR_CURVE_TYPE="Quadratic",
    PLR_A=0.85,
    PLR_B=0.25,
    PLR_C=-0.10,
    PLR_MIN_MODIFIER=0.55,
    PLR_MAX_MODIFIER=1.15,
    # ── Latent load ─────────────────────────────────────────────────────────
    INDOOR_RH_TARGET_PCT=50.0,
    LATENT_VENTILATION_FRACTION=0.35,
    # ── All publication coupled effects OFF (base) ───────────────────────────
    APPLY_PART_LOAD_COP_TO_CORE=False,
    APPLY_LATENT_LOAD_TO_CORE=False,
    APPLY_HX_AIR_PRESSURE_TO_FAN=False,
    APPLY_HX_WATER_PRESSURE_TO_PUMP=False,
    APPLY_HX_UA_TO_CAPACITY=False,
    APPLY_NATIVE_ZONE_LOADS=False,
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Effect variants — each entry enables exactly one (or all) coupled module(s)
# ──────────────────────────────────────────────────────────────────────────────

EFFECT_VARIANTS: dict[str, dict] = {
    "BASE_AllOff": {
        "APPLY_PART_LOAD_COP_TO_CORE": False,
        "APPLY_LATENT_LOAD_TO_CORE": False,
        "APPLY_HX_AIR_PRESSURE_TO_FAN": False,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": False,
        "APPLY_HX_UA_TO_CAPACITY": False,
    },
    "PLR_COP": {
        "APPLY_PART_LOAD_COP_TO_CORE": True,
        "APPLY_LATENT_LOAD_TO_CORE": False,
        "APPLY_HX_AIR_PRESSURE_TO_FAN": False,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": False,
        "APPLY_HX_UA_TO_CAPACITY": False,
    },
    "Latent_Load": {
        "APPLY_PART_LOAD_COP_TO_CORE": False,
        "APPLY_LATENT_LOAD_TO_CORE": True,
        "APPLY_HX_AIR_PRESSURE_TO_FAN": False,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": False,
        "APPLY_HX_UA_TO_CAPACITY": False,
    },
    "HX_Air_Fan": {
        "APPLY_PART_LOAD_COP_TO_CORE": False,
        "APPLY_LATENT_LOAD_TO_CORE": False,
        "APPLY_HX_AIR_PRESSURE_TO_FAN": True,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": False,
        "APPLY_HX_UA_TO_CAPACITY": False,
    },
    "HX_Water_Pump": {
        "APPLY_PART_LOAD_COP_TO_CORE": False,
        "APPLY_LATENT_LOAD_TO_CORE": False,
        "APPLY_HX_AIR_PRESSURE_TO_FAN": False,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": True,
        "APPLY_HX_UA_TO_CAPACITY": False,
    },
    "HX_UA_Capacity": {
        "APPLY_PART_LOAD_COP_TO_CORE": False,
        "APPLY_LATENT_LOAD_TO_CORE": False,
        "APPLY_HX_AIR_PRESSURE_TO_FAN": False,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": False,
        "APPLY_HX_UA_TO_CAPACITY": True,
    },
    "ALL_Effects": {
        "APPLY_PART_LOAD_COP_TO_CORE": True,
        "APPLY_LATENT_LOAD_TO_CORE": True,
        "APPLY_HX_AIR_PRESSURE_TO_FAN": True,
        "APPLY_HX_WATER_PRESSURE_TO_PUMP": True,
        "APPLY_HX_UA_TO_CAPACITY": True,
    },
}

# The four target KPIs
KPIS = [
    "Total Energy MWh",
    "Mean Comfort Deviation C",
    "Total CO2 tonne",
    "Mean Degradation Index",
]

KPI_LABELS = {
    "Total Energy MWh": "Energy Consumption (MWh)",
    "Mean Comfort Deviation C": "Comfort Deviation (°C)",
    "Total CO2 tonne": "Carbon Emissions (tonne CO₂)",
    "Mean Degradation Index": "Degradation Index (−)",
}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Runner
# ──────────────────────────────────────────────────────────────────────────────

def make_cfg(flags: dict) -> HVACConfig:
    """Clone base config and apply the effect flags for this variant."""
    d = asdict(BASE_CFG)
    d.update(flags)
    return HVACConfig(**d)


def run_all_effects(output_root: str = "effects_output") -> pd.DataFrame:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    figs_dir = root / "figures"
    figs_dir.mkdir(exist_ok=True)

    summary_rows = []

    total_runs = len(EFFECT_VARIANTS)
    for run_idx, (variant_name, flags) in enumerate(EFFECT_VARIANTS.items(), 1):
        print(f"\n[{run_idx}/{total_runs}] Effect variant: {variant_name}")
        cfg = make_cfg(flags)

        out_dir = root / variant_name
        try:
            paths = run_scenario_model(
                output_dir=out_dir,
                axis_mode="one_strategy",       # all 4 strategies, one severity/climate
                bldg=BLDG,
                cfg=cfg,
                weather_mode="synthetic",
                fixed_severity="Moderate",
                fixed_climate="C0_Baseline",
                degradation_model="physics",
                include_baseline_layer=False,
                random_state=42,
            )
            summary_df = pd.read_csv(paths["summary_csv"])
            for _, row in summary_df.iterrows():
                entry = {
                    "variant": variant_name,
                    "strategy": row.get("strategy", "?"),
                }
                for kpi in KPIS:
                    entry[kpi] = float(row.get(kpi, np.nan))
                # Enabled flags summary
                active = [k.replace("APPLY_", "").replace("_TO_CORE", "").replace("_TO_FAN", "").replace("_TO_PUMP", "")
                          for k, v in flags.items() if v]
                entry["active_effects"] = ", ".join(active) if active else "none"
                summary_rows.append(entry)
            print(f"  ✓ Done. Strategies: {summary_df['strategy'].tolist()}")
        except Exception as exc:
            print(f"  ✗ Error in variant {variant_name}: {exc}")
            import traceback; traceback.print_exc()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "effects_comparison_summary.csv", index=False)

    # Compute % delta vs BASE_AllOff for each strategy × KPI
    delta_rows = []
    base_ref = summary[summary["variant"] == "BASE_AllOff"].set_index("strategy")
    for _, row in summary[summary["variant"] != "BASE_AllOff"].iterrows():
        entry = {"variant": row["variant"], "strategy": row["strategy"], "active_effects": row["active_effects"]}
        for kpi in KPIS:
            base_val = float(base_ref.at[row["strategy"], kpi]) if row["strategy"] in base_ref.index else np.nan
            val = float(row[kpi])
            entry[f"delta_{kpi}_pct"] = 100.0 * (val - base_val) / max(abs(base_val), 1e-9) if np.isfinite(base_val) else np.nan
            entry[f"abs_{kpi}"] = val
        delta_rows.append(entry)
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(root / "effects_delta_vs_base.csv", index=False)

    # ── Figures: grouped bar chart for each KPI ──────────────────────────────
    strategies = sorted(summary["strategy"].unique().tolist())
    variants_order = list(EFFECT_VARIANTS.keys())
    n_variants = len(variants_order)
    x = np.arange(len(strategies))
    width = 0.12

    for kpi in KPIS:
        fig, ax = plt.subplots(figsize=(13, 5))
        for vi, variant in enumerate(variants_order):
            vals = []
            for stg in strategies:
                row = summary[(summary["variant"] == variant) & (summary["strategy"] == stg)]
                vals.append(float(row[kpi].values[0]) if len(row) else np.nan)
            offset = (vi - n_variants / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=variant)
        ax.set_xlabel("Strategy", fontsize=11)
        ax.set_ylabel(KPI_LABELS[kpi], fontsize=11)
        ax.set_title(f"Effect of Coupled Modules on {KPI_LABELS[kpi]}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        safe_name = kpi.replace(" ", "_").replace("/", "_").replace("°", "")
        plt.savefig(figs_dir / f"effect_{safe_name}.png", dpi=200)
        plt.close()
        print(f"  Figure saved: effect_{safe_name}.png")

    # ── Console summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EFFECTS COMPARISON — ALL STRATEGIES × ALL KPIs")
    print("=" * 80)
    pivot_cols = ["variant", "strategy"] + KPIS
    available = [c for c in pivot_cols if c in summary.columns]
    print(summary[available].to_string(index=False))
    print("\n% Delta vs BASE_AllOff:")
    delta_cols = ["variant", "strategy"] + [f"delta_{k}_pct" for k in KPIS]
    available_d = [c for c in delta_cols if c in delta_df.columns]
    print(delta_df[available_d].to_string(index=False))
    print(f"\nOutputs saved to: {root.resolve()}")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 4. Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all_effects("effects_output")
