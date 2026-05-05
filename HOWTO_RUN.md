# HVAC ROM-Degradation Suite — How to Run Guide

## Quick-Start (3 commands)

```bash
pip install -r requirements.txt
python run_effects_comparison.py   # recommended first run (~3 min)
python run_full_20yr.py            # full 20-year PhD run (~5 min)
```

---

## 1 · Fixing Slow Response

The main cause of slowness is the **S3 APO optimizer**. On every time step it calls `evaluate_controls()` a total of `APO_POP × APO_ITERS` times.

| Setting | Runs per step | 20-yr wall time* |
|---|---|---|
| Default `APO_POP=18, APO_ITERS=10` | 180 | ~15 min |
| **Recommended `APO_POP=12, APO_ITERS=6`** | 72 | ~5 min |
| Fast draft `APO_POP=8, APO_ITERS=4` | 32 | ~2 min |

*Approximate, 20-year, 4 strategies, daily time step.*

```python
# In your HVACConfig:
cfg = HVACConfig(
    TIME_STEP_HOURS=24.0,   # ← daily step is fastest (keep this!)
    APO_POP=12,             # reduce from 18 → 12
    APO_ITERS=6,            # reduce from 10 → 6
    years=20,
)
```

**Never use `TIME_STEP_HOURS=1` (hourly)** unless you have a specific sub-daily analysis need — it multiplies total steps by 24×.

---

## 2 · Enabling Coupled Effects So They Propagate to All 4 KPIs

The engine has **six publication-level coupled modules**, all disabled by default. You must explicitly enable each one.

### The six APPLY_* flags

| Flag | What it does | KPI affected |
|---|---|---|
| `APPLY_PART_LOAD_COP_TO_CORE` | COP degrades at part-load via PLR curve | ↑ Energy, ↑ CO₂ |
| `APPLY_LATENT_LOAD_TO_CORE` | Outdoor humidity adds latent cooling load | ↑ Energy, ↑ CO₂ |
| `APPLY_HX_AIR_PRESSURE_TO_FAN` | HX fouling raises air-side ΔP → fan W | ↑ Energy, ↑ Degradation, ↑ CO₂ |
| `APPLY_HX_WATER_PRESSURE_TO_PUMP` | HX fouling raises water-side ΔP → pump W | ↑ Energy, ↑ CO₂ |
| `APPLY_HX_UA_TO_CAPACITY` | Fouling reduces HX heat exchange capacity | ↑ Comfort deviation, ↑ Degradation |
| `APPLY_NATIVE_ZONE_LOADS` | Zone-by-zone load aggregation (supply zone_df) | All KPIs |

### How effects reach the KPIs

After the main solver (`evaluate_controls`) runs, `apply_core_coupled_corrections()` is called. It re-computes:

```
P_hvac  ← Q_HVAC / max(COP × PLR_modifier, 0.8)     ← APPLY_PART_LOAD_COP
P_fan   ← Q_air × af / FAN_EFF × dP_fan              ← APPLY_HX_AIR_PRESSURE
P_pump  ← HX water ΔP × flow / pump_eff              ← APPLY_HX_WATER_PRESSURE
E_period← (P_hvac + P_fan + P_pump + P_aux) × dt     → Energy KPI
co2     ← E_period × CO2_FACTOR                       → Carbon KPI
comfort ← |T_zone − T_set|, T_zone includes deg×occ   → Comfort KPI
deg     ← 0.5×(Rf/Rf*) + 0.5×(dP/dP_max)            → Degradation KPI
```

### Minimal working configuration with all effects ON

```python
from hvac_v3_engine import BuildingSpec, HVACConfig, run_scenario_model

cfg = HVACConfig(
    years=20,
    TIME_STEP_HOURS=24.0,
    APO_POP=12,
    APO_ITERS=6,

    # ── Physical degradation parameters ──
    RF_STAR=2e-4,
    B_FOUL=0.015,
    DUST_RATE=1.2,
    K_CLOG=6.0,
    CO2_FACTOR=0.536,          # Egypt grid

    # ── HX parameters (required when APPLY_HX_* are ON) ──
    HX_WATER_DP_CLEAN_KPA=35.0,
    HX_WATER_FLOW_NOM_M3H=0.0, # 0 → auto-derived from load
    HX_UA_LOSS_FACTOR=0.30,
    HX_PUMP_EFF=0.65,
    HX_CHW_DT_K=5.0,

    # ── PLR curve (required when APPLY_PART_LOAD_COP is ON) ──
    PLR_CURVE_TYPE="Quadratic",
    PLR_A=0.85,
    PLR_B=0.25,
    PLR_C=-0.10,

    # ── Latent load (required when APPLY_LATENT_LOAD is ON) ──
    INDOOR_RH_TARGET_PCT=50.0,
    LATENT_VENTILATION_FRACTION=0.35,

    # ── Enable all coupled effects ──
    APPLY_PART_LOAD_COP_TO_CORE=True,
    APPLY_LATENT_LOAD_TO_CORE=True,
    APPLY_HX_AIR_PRESSURE_TO_FAN=True,
    APPLY_HX_WATER_PRESSURE_TO_PUMP=True,
    APPLY_HX_UA_TO_CAPACITY=True,
)

result = run_scenario_model(
    output_dir="my_output",
    axis_mode="one_strategy",   # all 4 strategies, one severity + climate
    bldg=BuildingSpec(conditioned_area_m2=2000.0, floors=3, n_spaces=20),
    cfg=cfg,
    fixed_severity="Moderate",
    fixed_climate="C0_Baseline",
)
```

---

## 3 · Choosing `axis_mode`

| `axis_mode` | What runs | Output files |
|---|---|---|
| `"baseline_scenario"` | No-degradation reference only | `baseline_no_degradation_*.csv` |
| `"one_strategy"` | All 4 strategies (S0–S3), 1 severity, 1 climate | `one_axis_strategy_*.csv` |
| `"one_severity"` | 4 severities, 1 strategy, 1 climate | `one_axis_severity_*.csv` |
| `"two_axis"` | 4 strategies × 4 severities, 1 climate | `matrix_*.csv` |
| `"three_axis"` | 4 × 4 × 4 = 64 combos | `three_axis_*.csv` |

**Start with `"one_strategy"`** to verify results fast, then scale up.

---

## 4 · The 4 PhD KPI Columns (in summary CSV)

| KPI | Summary column | Daily column |
|---|---|---|
| Energy consumption | `Total Energy MWh` | `energy_kwh_period` |
| Comfort deviation | `Mean Comfort Deviation C` | `comfort_dev_C` |
| Carbon emissions | `Total CO2 tonne` | `co2_kg_period` |
| Degradation index | `Mean Degradation Index` | `delta` |

---

## 5 · Using the Streamlit App

```bash
streamlit run streamlit_app.py
```

**In the UI, to prevent slowness:**
- Tab "Building Identity & Setup" → **Simulation years**: set 5–10 for exploration, 20 for final
- Tab "Building Identity & Setup" → **Time step**: keep "Daily"
- Tab "Building Identity & Setup" → **APO population**: set 10–12
- Tab "Building Identity & Setup" → **APO iterations**: set 5–6
- Tab "Parameter Switches" → toggle the 6 `APPLY_*` switches ON
- Tab "Scenario Modeling" → run "Strategy comparison (one_strategy)" first

---

## 6 · Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `ValueError: Unsupported time-step` | TIME_STEP_HOURS not in {1, 3, 6, 12, 24} | Use only those values |
| `ZeroDivisionError` in pump/fan | HX_PUMP_EFF or FAN_EFF = 0 | Set > 0 (default 0.65 / 0.70) |
| Very large comfort deviation | HX_UA_TO_CAPACITY on, building undersized | Increase `cooling_intensity_w_m2` or reduce `HX_UA_LOSS_FACTOR` |
| S3 never finishes | APO_POP × APO_ITERS too large | Reduce as described in §1 |
| `FileNotFoundError` for EPW | Bad path in `epw_path=` | Use `weather_mode="synthetic"` first |

---

## 7 · Provided Scripts

| Script | Purpose | Run time |
|---|---|---|
| `run_example.py` | Minimal 1-year demo, baseline_scenario mode | ~5 s |
| `run_effects_comparison.py` | Isolates each effect, compares on 4 KPIs | ~3 min |
| `run_full_20yr.py` | Full 20-year, all effects ON, all strategies | ~5 min |
| `streamlit_app.py` | Interactive web dashboard | interactive |
