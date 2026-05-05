"""
HVAC ROM-Degradation Suite — Full 20-Year PhD Run
===================================================
Runs the full 20-year simulation with ALL coupled effects enabled,
all four strategies (S0–S3), Moderate severity, C0_Baseline climate.

ALL six publication-level coupled modules are active simultaneously:
  • Part-load COP curve            → COP degrades at part-load (centrifugal chiller PLR)
  • Latent moisture load           → Outdoor humidity adds to cooling duty
  • HX air-side ΔP → fan power    → Fouling raises duct pressure ↑ fan W
  • HX water-side ΔP → pump power → Fouling raises hydraulic resistance ↑ pump W
  • HX UA → capacity reduction    → Fouling reduces heat exchange capacity
  • Native zone loads (optional)  → Zone-by-zone load aggregation (OFF by default)

These propagate to all four KPIs automatically:
  Energy consumption (MWh)  → E_period = (P_hvac + P_fan + P_pump + P_aux) × dt
  Comfort deviation (°C)    → |T_zone − T_set|, T_zone boosted by degradation × occ
  Carbon emissions (tonne)  → E_period × CO2_FACTOR (Egypt: 0.536 kgCO2/kWh)
  Degradation index (−)     → deg = 0.5×(Rf/Rf*) + 0.5×(dP/dP_max)

Runtime: ~2–5 min for 20-year × 4-strategy on a modern laptop (daily time step).

How to run:
    python run_full_20yr.py

Output: ./full_20yr_output/
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model


def main():
    out = Path("full_20yr_output")

    bldg = BuildingSpec(
        building_type="Educational / University building",
        location="New Mansoura, Egypt (31.4°N)",
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

    cfg = HVACConfig(
        # ── Simulation horizon ───────────────────────────────────────────────
        years=20,
        TIME_STEP_HOURS=24.0,          # daily step (fastest, publication-valid)

        # ── HVAC system ──────────────────────────────────────────────────────
        hvac_system_type="Chiller_AHU",
        COP_COOL_NOM=4.5,
        COP_HEAT_NOM=3.2,
        COP_AGING_RATE=0.005,
        FAN_EFF=0.70,
        PUMP_SPECIFIC_W_M2=1.30,
        AUXILIARY_W_M2=0.55,

        # ── Degradation parameters ───────────────────────────────────────────
        RF_STAR=2e-4,                  # Kern-Seaton asymptote (m²K/W)
        B_FOUL=0.015,                  # fouling growth constant (day⁻¹)
        DUST_RATE=1.2,                 # dust accumulation (kg/day)
        K_CLOG=6.0,
        DEG_TRIGGER=0.55,

        # ── Setpoint range ───────────────────────────────────────────────────
        T_SET=23.0,
        T_SP_MIN=21.0,
        T_SP_MAX=26.0,
        AF_MIN=0.55,
        AF_MAX=1.00,

        # ── Objective weights ────────────────────────────────────────────────
        W_ENERGY=0.35,
        W_DEGRAD=0.25,
        W_COMFORT=0.25,
        W_CARBON=0.15,

        # ── Egypt grid ───────────────────────────────────────────────────────
        CO2_FACTOR=0.536,
        E_PRICE=0.12,

        # ── Maintenance cost ─────────────────────────────────────────────────
        COST_FILTER=50.0,
        COST_HX=300.0,
        FILTER_INTERVAL=90,
        HX_INTERVAL=180,

        # ── S3 optimizer (APO) ───────────────────────────────────────────────
        # For publication: APO_POP=18, APO_ITERS=10 (slower but more accurate)
        # For fast runs:   APO_POP=10, APO_ITERS=6
        APO_POP=12,
        APO_ITERS=6,

        # ── Publication coupled modules (ALL ON) ──────────────────────────────
        APPLY_PART_LOAD_COP_TO_CORE=True,
        APPLY_LATENT_LOAD_TO_CORE=True,
        APPLY_HX_AIR_PRESSURE_TO_FAN=True,
        APPLY_HX_WATER_PRESSURE_TO_PUMP=True,
        APPLY_HX_UA_TO_CAPACITY=True,
        APPLY_NATIVE_ZONE_LOADS=False,  # set True + supply zone_df for zone-level

        # ── Part-load COP curve coefficients (centrifugal chiller) ───────────
        PLR_CURVE_TYPE="Quadratic",
        PLR_A=0.85,
        PLR_B=0.25,
        PLR_C=-0.10,
        PLR_D=0.00,
        PLR_MIN_MODIFIER=0.55,
        PLR_MAX_MODIFIER=1.15,

        # ── Latent load parameters ────────────────────────────────────────────
        INDOOR_RH_TARGET_PCT=50.0,
        ATM_PRESSURE_PA=101325.0,
        LATENT_VENTILATION_FRACTION=0.35,
        FLOOR_TO_FLOOR_M=3.2,
        LATENT_HEAT_VAPORIZATION_KJ_KG=2501.0,

        # ── HX hydraulic / thermal coupling ───────────────────────────────────
        HX_AIR_FOULING_FACTOR=0.75,
        HX_WATER_DP_CLEAN_KPA=35.0,
        HX_WATER_FLOW_M3H=0.0,         # 0 → auto-derived from load
        HX_WATER_FLOW_NOM_M3H=0.0,     # 0 → auto-derived
        HX_WATER_FOULING_FACTOR=0.35,
        HX_PUMP_EFF=0.65,
        HX_CHW_DT_K=5.0,
        HX_HW_DT_K=10.0,
        HX_UA_CLEAN_KW_K=0.0,          # placeholder; capacity formula uses UA_LOSS_FACTOR
        HX_UA_LOSS_FACTOR=0.30,
        HX_LMTD_CORRECTION=0.90,
        HX_AIR_DENSITY_KG_M3=1.20,
        HX_WATER_DENSITY_KG_M3=997.0,
        HX_CP_AIR_KJ_KG_K=1.006,
        HX_CP_WATER_KJ_KG_K=4.186,
    )

    print("Starting 20-year coupled HVAC simulation …")
    print(f"  Strategies  : S0, S1, S2, S3")
    print(f"  Time step   : {cfg.TIME_STEP_HOURS} h (daily)")
    print(f"  Years       : {cfg.years}")
    print(f"  APO pop/iter: {cfg.APO_POP} / {cfg.APO_ITERS}")
    print(f"  All APPLY_* : True")
    print()

    result = run_scenario_model(
        output_dir=out,
        axis_mode="one_strategy",           # all 4 strategies, single severity/climate
        bldg=bldg,
        cfg=cfg,
        weather_mode="synthetic",
        fixed_severity="Moderate",
        fixed_climate="C0_Baseline",
        degradation_model="physics",        # Kern-Seaton + linear dust
        include_baseline_layer=True,        # also runs a no-degradation baseline
        include_baseline_as_scenario=True,
        random_state=42,
    )

    # ── Print headline KPIs ──────────────────────────────────────────────────
    summary = pd.read_csv(result["summary_csv"])
    kpis = ["strategy", "Total Energy MWh", "Mean Comfort Deviation C",
            "Total CO2 tonne", "Mean Degradation Index",
            "Occupied Discomfort Days", "Total Cost USD"]
    available = [c for c in kpis if c in summary.columns]
    print("\n" + "=" * 70)
    print("20-YEAR SIMULATION RESULTS — HEADLINE KPIs")
    print("=" * 70)
    print(summary[available].to_string(index=False))
    print()
    print("Annual breakdown:")
    annual = pd.read_csv(result["annual_csv"])
    annual_cols = ["strategy", "year", "annual_energy_MWh", "mean_delta",
                   "mean_comfort_dev", "annual_co2_tonne"]
    available_a = [c for c in annual_cols if c in annual.columns]
    print(annual[available_a].to_string(index=False))

    print(f"\n✓ Full outputs: {out.resolve()}")
    print(f"  Excel report : {result['excel_report']}")
    print(f"  PDF report   : {result['pdf_report']}")
    print(f"  Figures      : {result['figures_dir']}")


if __name__ == "__main__":
    main()
