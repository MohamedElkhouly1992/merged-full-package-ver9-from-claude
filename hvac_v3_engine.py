
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

# Optional ML libs
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler


SCENARIOS = {
    "S0": "Unaware baseline controller + fixed maintenance",
    "S1": "Reactive maintenance + fixed control",
    "S2": "Preventive scheduled maintenance + fixed control",
    "S3": "Predictive maintenance + weight-sensitive optimization",
}

SEVERITY_LEVELS: Dict[str, Dict[str, float]] = {
    "Mild": {"B_FOUL_mult": 0.70, "DUST_RATE_mult": 0.70, "COP_AGING_RATE_mult": 0.70, "RF_STAR_mult": 0.90, "K_CLOG_mult": 0.90, "DEG_TRIGGER_shift": +0.03},
    "Moderate": {"B_FOUL_mult": 1.00, "DUST_RATE_mult": 1.00, "COP_AGING_RATE_mult": 1.00, "RF_STAR_mult": 1.00, "K_CLOG_mult": 1.00, "DEG_TRIGGER_shift": 0.00},
    "Severe": {"B_FOUL_mult": 1.35, "DUST_RATE_mult": 1.40, "COP_AGING_RATE_mult": 1.30, "RF_STAR_mult": 1.15, "K_CLOG_mult": 1.20, "DEG_TRIGGER_shift": -0.03},
    "High": {"B_FOUL_mult": 1.60, "DUST_RATE_mult": 1.65, "COP_AGING_RATE_mult": 1.50, "RF_STAR_mult": 1.25, "K_CLOG_mult": 1.30, "DEG_TRIGGER_shift": -0.05},
}

CLIMATE_LEVELS: Dict[str, Dict[str, float]] = {
    "C0_Baseline": {"temp_shift": 0.0, "summer_pulse": 0.0, "future_drift_per_year": 0.03, "rh_shift": 0.0, "solar_mult": 1.00},
    "C1_Warm": {"temp_shift": 1.5, "summer_pulse": 0.8, "future_drift_per_year": 0.04, "rh_shift": -1.0, "solar_mult": 1.03},
    "C2_Heatwave": {"temp_shift": 1.0, "summer_pulse": 3.0, "future_drift_per_year": 0.05, "rh_shift": -4.0, "solar_mult": 1.05},
    "C3_FutureHot": {"temp_shift": 4.0, "summer_pulse": 1.5, "future_drift_per_year": 0.08, "rh_shift": -2.0, "solar_mult": 1.04},
}

HVAC_PRESETS = {
    "Chiller_AHU": {"COP_COOL_NOM": 4.5, "COP_HEAT_NOM": 3.2, "FAN_EFF": 0.70, "PUMP_SPECIFIC_W_M2": 1.30, "AUXILIARY_W_M2": 0.55},
    "VRF": {"COP_COOL_NOM": 3.8, "COP_HEAT_NOM": 3.6, "FAN_EFF": 0.62, "PUMP_SPECIFIC_W_M2": 0.20, "AUXILIARY_W_M2": 0.35},
    "Packaged_DX": {"COP_COOL_NOM": 3.2, "COP_HEAT_NOM": 3.0, "FAN_EFF": 0.60, "PUMP_SPECIFIC_W_M2": 0.10, "AUXILIARY_W_M2": 0.35},
    "Heat_Pump": {"COP_COOL_NOM": 3.4, "COP_HEAT_NOM": 3.8, "FAN_EFF": 0.65, "PUMP_SPECIFIC_W_M2": 0.40, "AUXILIARY_W_M2": 0.35},
    "Custom": {"COP_COOL_NOM": 4.0, "COP_HEAT_NOM": 3.2, "FAN_EFF": 0.65, "PUMP_SPECIFIC_W_M2": 1.00, "AUXILIARY_W_M2": 0.50},
}

TIME_STEP_OPTIONS = {"Hourly": 1.0, "3-hour": 3.0, "6-hour": 6.0, "12-hour": 12.0, "Daily": 24.0}





ZONE_TYPE_DEFAULT_FACTORS = {
    "Lecture": {"term_factor": 0.95, "break_factor": 0.20, "summer_factor": 0.10},
    "Office": {"term_factor": 0.85, "break_factor": 0.55, "summer_factor": 0.35},
    "Lab": {"term_factor": 0.90, "break_factor": 0.45, "summer_factor": 0.30},
    "Corridor": {"term_factor": 0.60, "break_factor": 0.45, "summer_factor": 0.35},
    "Service": {"term_factor": 0.70, "break_factor": 0.65, "summer_factor": 0.60},
    "Custom": {"term_factor": 1.00, "break_factor": 1.00, "summer_factor": 1.00},
}


@dataclass
class BuildingSpec:
    building_type: str = "Educational / University building"
    location: str = "User-defined"
    conditioned_area_m2: float = 5000.0
    floors: int = 4
    n_spaces: int = 40
    occupancy_density_p_m2: float = 0.08
    lighting_w_m2: float = 10.0
    equipment_w_m2: float = 8.0
    airflow_m3h_m2: float = 4.0
    infiltration_ach: float = 0.5
    sensible_w_per_person: float = 75.0
    cooling_intensity_w_m2: float = 100.0
    heating_intensity_w_m2: float = 55.0
    wall_u: float = 0.6
    roof_u: float = 0.35
    window_u: float = 2.7
    shgc: float = 0.35
    glazing_ratio: float = 0.30


@dataclass
class HVACConfig:
    years: int = 20
    days_per_year: int = 365
    hvac_system_type: str = "Chiller_AHU"
    COP_COOL_NOM: float = 4.5
    COP_HEAT_NOM: float = 3.2
    COP_AGING_RATE: float = 0.005
    FAN_EFF: float = 0.70
    # Pump and auxiliary terms are included in total electrical energy when enabled.
    # PUMP_SPECIFIC_W_M2 is area-normalized pump demand; AUXILIARY_W_M2 represents controls, standby, valves, small motors, etc.
    PUMP_SPECIFIC_W_M2: float = 1.30
    AUXILIARY_W_M2: float = 0.55
    T_SET: float = 23.0
    T_SP_MIN: float = 21.0
    T_SP_MAX: float = 26.0
    AF_MIN: float = 0.55
    AF_MAX: float = 1.00
    RF_STAR: float = 2e-4
    B_FOUL: float = 0.015
    RF_THRESH: float = 1.6e-4
    RF_WARN: float = 1.2e-4
    DP_CLEAN: float = 150.0
    DP_THRESH: float = 420.0
    DP_WARN: float = 320.0
    DP_MAX: float = 450.0
    DUST_RATE: float = 1.2
    K_CLOG: float = 6.0
    DEG_TRIGGER: float = 0.55
    E_PRICE: float = 0.12
    CO2_FACTOR: float = 0.536
    COST_FILTER: float = 50.0
    COST_HX: float = 300.0
    FILTER_INTERVAL: int = 90
    HX_INTERVAL: int = 180
    W_ENERGY: float = 0.35
    W_DEGRAD: float = 0.25
    W_COMFORT: float = 0.25
    W_CARBON: float = 0.15
    DT_REF_COOL: float = 15.0
    DT_REF_HEAT: float = 18.0
    A_COOL_ENV: float = 0.45
    A_HEAT_ENV: float = 0.55
    INTERNAL_USE_FACTOR: float = 0.65
    HEAT_INTERNAL_CREDIT: float = 0.60
    SOLAR_COOL_FACTOR: float = 0.12
    INFIL_COOL_FACTOR: float = 0.08
    INFIL_HEAT_FACTOR: float = 0.10
    HUMIDITY_COOL_FACTOR: float = 0.004
    HUMIDITY_COMFORT_FACTOR: float = 0.02
    APO_POP: int = 50
    APO_ITERS: int = 100
    degradation_model: str = "physics"
    LINEAR_DEG_PER_DAY: float = 0.00012
    EXP_DEG_RATE_PER_DAY: float = 0.00018
    # Numerical time basis. 24 h preserves the original daily model.
    TIME_STEP_HOURS: float = 24.0
    # Parameter switches used by the Streamlit UI. They zero or disable terms without
    # duplicating the model outside this engine.
    USE_HVAC_PRESET: bool = True
    USE_ENVELOPE: bool = True
    USE_WALLS: bool = True
    USE_ROOF: bool = True
    USE_WINDOWS: bool = True
    USE_SOLAR: bool = True
    USE_INFILTRATION: bool = True
    USE_INTERNAL_GAINS: bool = True
    USE_PEOPLE_GAINS: bool = True
    USE_LIGHTING_GAINS: bool = True
    USE_EQUIPMENT_GAINS: bool = True
    USE_HVAC_FANS: bool = True
    USE_HVAC_PUMPS: bool = True
    USE_HVAC_AUXILIARY: bool = True
    USE_COOLING: bool = True
    USE_HEATING: bool = True
    USE_DEGRADATION: bool = True
    USE_CARBON: bool = True
    USE_MAINTENANCE_COST: bool = True
    # EMS / advanced control options. Defaults reproduce the original model.
    EMS_MODE: str = "Disabled"
    EMS_OCC_CONTROL: bool = False
    EMS_NIGHT_SETBACK: bool = False
    EMS_DEMAND_RESPONSE: bool = False
    EMS_ECONOMIZER: bool = False
    EMS_OPTIMUM_START: bool = False
    EMS_CUSTOM_SCHEDULE_ENABLED: bool = False
    EMS_LOW_OCC_THRESHOLD: float = 0.25
    EMS_LOW_OCC_AIRFLOW_FACTOR: float = 0.65
    EMS_LOW_OCC_SETPOINT_SHIFT_C: float = 1.0
    EMS_NIGHT_START_HOUR: float = 19.0
    EMS_NIGHT_END_HOUR: float = 6.0
    EMS_NIGHT_SETPOINT_SHIFT_C: float = 2.0
    EMS_NIGHT_AIRFLOW_FACTOR: float = 0.55
    EMS_DR_START_HOUR: float = 13.0
    EMS_DR_END_HOUR: float = 17.0
    EMS_DR_SETPOINT_SHIFT_C: float = 1.5
    EMS_DR_AIRFLOW_REDUCTION: float = 0.15
    EMS_ECONOMIZER_TEMP_LOW_C: float = 16.0
    EMS_ECONOMIZER_TEMP_HIGH_C: float = 22.0
    EMS_ECONOMIZER_COOLING_REDUCTION: float = 0.20
    EMS_OPTIMUM_START_HOUR: float = 7.0
    EMS_PRECOOL_SHIFT_C: float = -0.8

    # Strong-coupled publication modules. Defaults are OFF to preserve backward-compatible results.
    APPLY_PART_LOAD_COP_TO_CORE: bool = False
    APPLY_LATENT_LOAD_TO_CORE: bool = False
    APPLY_HX_AIR_PRESSURE_TO_FAN: bool = False
    APPLY_HX_WATER_PRESSURE_TO_PUMP: bool = False
    APPLY_HX_UA_TO_CAPACITY: bool = False
    APPLY_NATIVE_ZONE_LOADS: bool = False

    # Part-load COP curve coefficients: f_PLR = a + b*PLR + c*PLR^2 + d*PLR^3
    PLR_CURVE_TYPE: str = "Quadratic"
    PLR_A: float = 0.85
    PLR_B: float = 0.25
    PLR_C: float = -0.10
    PLR_D: float = 0.00
    PLR_MIN_MODIFIER: float = 0.55
    PLR_MAX_MODIFIER: float = 1.15

    # Latent-load coupling inputs.
    INDOOR_RH_TARGET_PCT: float = 50.0
    ATM_PRESSURE_PA: float = 101325.0
    LATENT_VENTILATION_FRACTION: float = 0.35
    FLOOR_TO_FLOOR_M: float = 3.2
    LATENT_HEAT_VAPORIZATION_KJ_KG: float = 2501.0

    # Heat-exchanger hydraulic/thermal coupling inputs.
    HX_AIR_FOULING_FACTOR: float = 0.75
    HX_WATER_DP_CLEAN_KPA: float = 35.0
    HX_WATER_FLOW_M3H: float = 0.0
    HX_WATER_FLOW_NOM_M3H: float = 0.0
    HX_WATER_FOULING_FACTOR: float = 0.35
    HX_PUMP_EFF: float = 0.65
    HX_CHW_DT_K: float = 5.0
    HX_HW_DT_K: float = 10.0
    HX_UA_CLEAN_KW_K: float = 0.0
    HX_UA_LOSS_FACTOR: float = 0.30
    HX_LMTD_CORRECTION: float = 0.90
    HX_AIR_DENSITY_KG_M3: float = 1.20
    HX_WATER_DENSITY_KG_M3: float = 997.0
    HX_CP_AIR_KJ_KG_K: float = 1.006
    HX_CP_WATER_KJ_KG_K: float = 4.186


def apply_hvac_preset(cfg: HVACConfig) -> HVACConfig:
    """Apply an HVAC preset unless the UI/user selected Custom or disabled presets."""
    out = HVACConfig(**asdict(cfg))
    if not getattr(out, "USE_HVAC_PRESET", True) or out.hvac_system_type == "Custom":
        return out
    preset = HVAC_PRESETS.get(out.hvac_system_type, HVAC_PRESETS["Chiller_AHU"])
    out.COP_COOL_NOM = preset["COP_COOL_NOM"]
    out.COP_HEAT_NOM = preset["COP_HEAT_NOM"]
    out.FAN_EFF = preset["FAN_EFF"]
    out.PUMP_SPECIFIC_W_M2 = preset.get("PUMP_SPECIFIC_W_M2", out.PUMP_SPECIFIC_W_M2)
    out.AUXILIARY_W_M2 = preset.get("AUXILIARY_W_M2", out.AUXILIARY_W_M2)
    return out



def aggregate_zone_occupancy(bldg: BuildingSpec, zone_df: Optional[pd.DataFrame]) -> Tuple[BuildingSpec, Dict[str, float]]:
    if zone_df is None or len(zone_df) == 0:
        return bldg, {
            "mode": "general",
            "weighted_occ_density": bldg.occupancy_density_p_m2,
            "schedule_profile": {"term_factor": 0.80, "break_factor": 0.25, "summer_factor": 0.35},
        }

    df = zone_df.copy()
    required = ["zone_name", "zone_type", "area_m2", "occ_density"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Zone occupancy table missing column: {col}")

    # backward compatibility with old single schedule_factor
    if "term_factor" not in df.columns:
        if "schedule_factor" in df.columns:
            df["term_factor"] = df["schedule_factor"]
            df["break_factor"] = df["schedule_factor"]
            df["summer_factor"] = df["schedule_factor"]
        else:
            df["term_factor"] = np.nan
            df["break_factor"] = np.nan
            df["summer_factor"] = np.nan

    for factor_col in ["term_factor", "break_factor", "summer_factor"]:
        for i, row in df.iterrows():
            if pd.isna(row[factor_col]):
                prof = ZONE_TYPE_DEFAULT_FACTORS.get(str(row["zone_type"]), ZONE_TYPE_DEFAULT_FACTORS["Custom"])
                df.at[i, factor_col] = prof[factor_col]

    area_sum = float(df["area_m2"].sum())
    if area_sum <= 0:
        raise ValueError("Zone area total must be > 0")

    weighted_occ_density = float((df["area_m2"] * df["occ_density"]).sum() / area_sum)
    # schedule factors should be weighted by peak occupancy contribution
    occ_weight = df["area_m2"] * df["occ_density"]
    if float(occ_weight.sum()) <= 0:
        occ_weight = df["area_m2"]

    schedule_profile = {
        "term_factor": float((occ_weight * df["term_factor"]).sum() / occ_weight.sum()),
        "break_factor": float((occ_weight * df["break_factor"]).sum() / occ_weight.sum()),
        "summer_factor": float((occ_weight * df["summer_factor"]).sum() / occ_weight.sum()),
    }

    out = BuildingSpec(**asdict(bldg))
    out.conditioned_area_m2 = area_sum
    out.n_spaces = int(len(df))
    out.occupancy_density_p_m2 = weighted_occ_density

    zone_table = df[["zone_name", "zone_type", "area_m2", "occ_density", "term_factor", "break_factor", "summer_factor"]].copy()

    return out, {
        "mode": "zone_specific",
        "weighted_occ_density": weighted_occ_density,
        "schedule_profile": schedule_profile,
        "n_zones": int(len(df)),
        "zone_table": zone_table.to_dict(orient="records"),
    }



def derive_building_numbers(bldg: BuildingSpec) -> Dict[str, float]:
    return {
        "Q_cool_des_kw": bldg.conditioned_area_m2 * bldg.cooling_intensity_w_m2 / 1000.0,
        "Q_heat_des_kw": bldg.conditioned_area_m2 * bldg.heating_intensity_w_m2 / 1000.0,
        "Q_air_nom_m3h": bldg.conditioned_area_m2 * bldg.airflow_m3h_m2,
        "N_people_max": bldg.conditioned_area_m2 * bldg.occupancy_density_p_m2,
        "Internal_kw_max": bldg.conditioned_area_m2 * (bldg.lighting_w_m2 + bldg.equipment_w_m2) / 1000.0,
    }


def synthetic_daily_weather(random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for doy in range(1, 366):
        t_mean = 21.5 + 9.0 * math.sin(2 * math.pi * (doy - 80) / 365.0) + rng.normal(0, 0.5)
        t_max = t_mean + 5.0 + 1.5 * max(math.sin(2 * math.pi * (doy - 80) / 365.0), 0.0) + rng.normal(0, 0.4)
        rh_mean = float(np.clip(65 + 15 * math.sin(2 * math.pi * (doy - 150) / 365.0) + rng.normal(0, 2.0), 25, 95))
        ghi_mean = float(max(0.0, 350 + 250 * max(math.sin(math.pi * doy / 365.0), 0.0) + rng.normal(0, 20.0)))
        rows.append({"day_of_year": doy, "T_mean_C": t_mean, "T_max_C": t_max, "RH_mean_pct": rh_mean, "GHI_mean_Wm2": ghi_mean})
    return pd.DataFrame(rows)


def read_epw_daily(epw_path: str | Path) -> pd.DataFrame:
    epw_path = Path(epw_path)
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")
    names = [
        "Year", "Month", "Day", "Hour", "Minute", "DataSource", "DryBulb", "DewPoint", "RH", "Pressure",
        "ExtHorzRad", "ExtDirNormRad", "HorzIRSky", "GHI", "DNI", "DHI", "GHIllum", "DNIllum", "DHIllum",
        "ZenLum", "WindDir", "WindSpd", "TotSkyCvr", "OpaqSkyCvr", "Visibility", "CeilingHgt", "PresWeathObs",
        "PresWeathCodes", "PrecipWater", "AerosolOptDepth", "SnowDepth", "DaysSinceSnow", "Albedo",
        "LiquidPrecipDepth", "LiquidPrecipQty",
    ]
    df = pd.read_csv(epw_path, skiprows=8, header=None, names=names)
    use = df[["Month", "Day", "DryBulb", "RH", "GHI"]].copy()
    use = use[~((use["Month"] == 2) & (use["Day"] == 29))].copy()
    daily = use.groupby(["Month", "Day"], as_index=False).agg(
        T_mean_C=("DryBulb", "mean"),
        T_max_C=("DryBulb", "max"),
        RH_mean_pct=("RH", "mean"),
        GHI_mean_Wm2=("GHI", "mean"),
    )
    daily["date"] = pd.to_datetime({"year": 2001, "month": daily["Month"], "day": daily["Day"]})
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily = daily.sort_values("day_of_year")[["day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]].reset_index(drop=True)
    if len(daily) != 365:
        raise ValueError(f"Expected 365 daily rows after EPW aggregation, got {len(daily)}")
    return daily




def ensure_365_daily_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded daily weather to the engine format without changing model equations."""
    if df is None or len(df) == 0:
        raise ValueError("Weather dataframe is empty.")
    out = df.copy()
    aliases = {
        "T_amb_C": "T_mean_C",
        "Outdoor Dry-Bulb Temperature": "T_mean_C",
        "Relative Humidity": "RH_mean_pct",
        "RH_pct": "RH_mean_pct",
        "Global Solar Radiation": "GHI_mean_Wm2",
        "GHI_Wm2": "GHI_mean_Wm2",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    if "T_max_C" not in out.columns and "T_mean_C" in out.columns:
        out["T_max_C"] = pd.to_numeric(out["T_mean_C"], errors="coerce") + 5.0
    if "RH_mean_pct" not in out.columns:
        out["RH_mean_pct"] = 60.0
    if "GHI_mean_Wm2" not in out.columns:
        out["GHI_mean_Wm2"] = 0.0
    if "day_of_year" not in out.columns:
        if "Date/Time" in out.columns:
            out["day_of_year"] = pd.to_datetime(out["Date/Time"], errors="coerce").dt.dayofyear
        elif "date" in out.columns:
            out["day_of_year"] = pd.to_datetime(out["date"], errors="coerce").dt.dayofyear
        else:
            out["day_of_year"] = np.arange(1, len(out) + 1)
    cols = ["day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=cols).copy()
    out["day_of_year"] = out["day_of_year"].astype(int)
    out = out[(out["day_of_year"] >= 1) & (out["day_of_year"] <= 366)].copy()
    out.loc[out["day_of_year"] > 365, "day_of_year"] = 365
    out = out.groupby("day_of_year", as_index=False)[cols[1:]].mean().sort_values("day_of_year")
    if len(out) != 365 or out["day_of_year"].tolist() != list(range(1, 366)):
        out = out.set_index("day_of_year").reindex(range(1, 366))
        out[cols[1:]] = out[cols[1:]].interpolate(limit_direction="both").ffill().bfill()
        out = out.reset_index().rename(columns={"index":"day_of_year"})
    if out[cols[1:]].isna().any().any():
        raise ValueError("Could not normalize weather data to 365 daily records.")
    return out[cols].reset_index(drop=True)


def read_weather_csv_daily(csv_path: str | Path) -> pd.DataFrame:
    """Read daily weather CSV in engine format or common date/temp/RH/GHI format."""
    df = pd.read_csv(csv_path)
    if {"T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"}.issubset(df.columns):
        return ensure_365_daily_weather(df)
    colmap = {str(c).strip().lower(): c for c in df.columns}
    def find(names):
        for n in names:
            if n.lower() in colmap:
                return colmap[n.lower()]
        for c in df.columns:
            low = str(c).lower()
            if any(n.lower() in low for n in names):
                return c
        return None
    date_col = find(["Date/Time", "date", "datetime", "timestamp"])
    temp_col = find(["T_amb_C", "temperature", "temp", "dry-bulb", "DryBulb", "Outdoor Dry-Bulb Temperature"])
    rh_col = find(["RH", "humidity", "Relative Humidity", "RH_pct"])
    ghi_col = find(["GHI", "solar", "Global Solar Radiation", "GHI_Wm2"])
    if date_col is None or temp_col is None:
        raise ValueError("Weather CSV must contain date/time and outdoor temperature columns.")
    work = pd.DataFrame({
        "Date/Time": pd.to_datetime(df[date_col], errors="coerce"),
        "temp": pd.to_numeric(df[temp_col], errors="coerce"),
        "rh": pd.to_numeric(df[rh_col], errors="coerce") if rh_col else 60.0,
        "ghi": pd.to_numeric(df[ghi_col], errors="coerce") if ghi_col else 0.0,
    }).dropna(subset=["Date/Time", "temp"])
    work["date_only"] = work["Date/Time"].dt.floor("D")
    daily = work.groupby("date_only", as_index=False).agg(
        T_mean_C=("temp", "mean"),
        T_max_C=("temp", "max"),
        RH_mean_pct=("rh", "mean"),
        GHI_mean_Wm2=("ghi", "mean"),
    )
    daily["day_of_year"] = daily["date_only"].dt.dayofyear
    daily = daily[~((daily["date_only"].dt.month == 2) & (daily["date_only"].dt.day == 29))].copy()
    return ensure_365_daily_weather(daily[["day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]])


def read_weather_auto_daily(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".epw":
        return read_epw_daily(path)
    if path.suffix.lower() in [".csv", ".txt"]:
        return read_weather_csv_daily(path)
    raise ValueError("Unsupported weather upload. Use EPW or CSV.")


def _target_freq_from_hours(time_step_hours: float) -> str:
    hours = resolve_time_step_hours(time_step_hours)
    if hours == 24.0:
        return "1D"
    return f"{int(hours)}h"


def _prepare_timeseries_from_timestamped_df(work: pd.DataFrame, time_step_hours: float) -> pd.DataFrame:
    """Normalize timestamped weather to the selected native simulation time-step.

    Output columns are intentionally the same load-driver names used by the original
    engine, plus hour/step metadata. This lets the model use hourly/sub-daily EPW or
    CSV weather directly instead of repeating one daily value within each day.
    """
    if "Date/Time" not in work.columns:
        raise ValueError("Timestamped weather dataframe must contain Date/Time.")
    hours = resolve_time_step_hours(time_step_hours)
    df = work.copy()
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
    df = df.dropna(subset=["Date/Time"]).sort_values("Date/Time")
    if df.empty:
        raise ValueError("No valid timestamps found in weather data.")
    # Use a non-leap representative year to keep all scenario years comparable.
    df = df[~((df["Date/Time"].dt.month == 2) & (df["Date/Time"].dt.day == 29))].copy()
    for col, default in [("T_mean_C", np.nan), ("T_max_C", np.nan), ("RH_mean_pct", 60.0), ("GHI_mean_Wm2", 0.0)]:
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["T_mean_C"].isna().all():
        raise ValueError("Weather data must contain outdoor dry-bulb temperature.")
    df["T_mean_C"] = df["T_mean_C"].interpolate(limit_direction="both").ffill().bfill()
    df["T_max_C"] = df["T_max_C"].fillna(df["T_mean_C"])
    df["RH_mean_pct"] = df["RH_mean_pct"].interpolate(limit_direction="both").ffill().bfill().fillna(60.0)
    df["GHI_mean_Wm2"] = df["GHI_mean_Wm2"].interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    df = df.set_index("Date/Time")
    freq = _target_freq_from_hours(hours)
    if hours == 24.0:
        agg = df.resample(freq).agg(
            T_mean_C=("T_mean_C", "mean"),
            T_max_C=("T_mean_C", "max"),
            RH_mean_pct=("RH_mean_pct", "mean"),
            GHI_mean_Wm2=("GHI_mean_Wm2", "mean"),
        )
    else:
        agg = df.resample(freq).agg(
            T_mean_C=("T_mean_C", "mean"),
            T_max_C=("T_mean_C", "max"),
            RH_mean_pct=("RH_mean_pct", "mean"),
            GHI_mean_Wm2=("GHI_mean_Wm2", "mean"),
        )
    agg = agg.dropna(subset=["T_mean_C"]).reset_index()
    if agg.empty:
        raise ValueError("Weather resampling produced no usable rows.")
    agg = agg[~((agg["Date/Time"].dt.month == 2) & (agg["Date/Time"].dt.day == 29))].copy()
    agg["day_of_year"] = agg["Date/Time"].dt.dayofyear.astype(int)
    agg.loc[agg["day_of_year"] > 365, "day_of_year"] = 365
    agg["hour_of_day"] = agg["Date/Time"].dt.hour + agg["Date/Time"].dt.minute / 60.0
    agg["step_of_year"] = np.arange(1, len(agg) + 1)
    agg["time_step_hours"] = hours
    expected = steps_per_year_from_hours(hours)
    # Reindex against a representative 2001 year. This fills occasional missing CSV/EPW records.
    base_index = pd.date_range("2001-01-01 00:00:00", periods=expected, freq=freq)
    tmp = agg.copy()
    tmp["Date/Time"] = tmp["Date/Time"].apply(lambda x: x.replace(year=2001))
    tmp = tmp.set_index("Date/Time").reindex(base_index)
    tmp[["T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]] = tmp[["T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]].interpolate(limit_direction="both").ffill().bfill()
    tmp = tmp.reset_index().rename(columns={"index": "Date/Time"})
    tmp["day_of_year"] = tmp["Date/Time"].dt.dayofyear.astype(int)
    tmp["hour_of_day"] = tmp["Date/Time"].dt.hour + tmp["Date/Time"].dt.minute / 60.0
    tmp["step_of_year"] = np.arange(1, len(tmp) + 1)
    tmp["time_step_hours"] = hours
    tmp["weather_native_resolution"] = "timestamped_resampled"
    return tmp[["step_of_year", "day_of_year", "hour_of_day", "time_step_hours", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2", "weather_native_resolution"]]


def expand_daily_weather_to_timeseries(daily: pd.DataFrame, time_step_hours: float) -> pd.DataFrame:
    """Expand a 365-row daily file to the selected time-step with synthetic diurnal shape."""
    hours = resolve_time_step_hours(time_step_hours)
    daily = ensure_365_daily_weather(daily)
    steps_per_day = max(1, int(round(24.0 / hours)))
    rows = []
    for _, r in daily.iterrows():
        doy = int(r["day_of_year"])
        t_mean = float(r["T_mean_C"])
        t_max = float(r.get("T_max_C", t_mean + 5.0))
        amp = max(t_max - t_mean, 2.0)
        for j in range(steps_per_day):
            hour = j * hours
            if hours >= 24:
                temp = t_mean
                ghi = float(r["GHI_mean_Wm2"])
            else:
                # Warmest in mid-afternoon, coolest near early morning.
                temp = t_mean + amp * math.sin(2 * math.pi * (hour - 8.0) / 24.0)
                solar_shape = max(math.sin(math.pi * (hour + hours / 2.0 - 6.0) / 12.0), 0.0)
                # Keep the daily mean roughly comparable by distributing daylight intensity.
                ghi = max(float(r["GHI_mean_Wm2"]) * solar_shape * 1.8, 0.0)
            rows.append({
                "day_of_year": doy,
                "hour_of_day": float(hour),
                "T_mean_C": float(temp),
                "T_max_C": float(max(t_max, temp)),
                "RH_mean_pct": float(r["RH_mean_pct"]),
                "GHI_mean_Wm2": float(ghi),
            })
    out = pd.DataFrame(rows)
    out["step_of_year"] = np.arange(1, len(out) + 1)
    out["time_step_hours"] = hours
    out["weather_native_resolution"] = "daily_expanded_diurnal"
    return out[["step_of_year", "day_of_year", "hour_of_day", "time_step_hours", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2", "weather_native_resolution"]]


def synthetic_weather_timeseries(time_step_hours: float = 24.0, random_state: int = 42) -> pd.DataFrame:
    return expand_daily_weather_to_timeseries(synthetic_daily_weather(random_state), time_step_hours)


def read_epw_timeseries(epw_path: str | Path, time_step_hours: float = 24.0) -> pd.DataFrame:
    epw_path = Path(epw_path)
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")
    names = [
        "Year", "Month", "Day", "Hour", "Minute", "DataSource", "DryBulb", "DewPoint", "RH", "Pressure",
        "ExtHorzRad", "ExtDirNormRad", "HorzIRSky", "GHI", "DNI", "DHI", "GHIllum", "DNIllum", "DHIllum",
        "ZenLum", "WindDir", "WindSpd", "TotSkyCvr", "OpaqSkyCvr", "Visibility", "CeilingHgt", "PresWeathObs",
        "PresWeathCodes", "PrecipWater", "AerosolOptDepth", "SnowDepth", "DaysSinceSnow", "Albedo",
        "LiquidPrecipDepth", "LiquidPrecipQty",
    ]
    df = pd.read_csv(epw_path, skiprows=8, header=None, names=names)
    rows = []
    for _, r in df.iterrows():
        try:
            month = int(float(r["Month"])); day = int(float(r["Day"])); hour = int(float(r["Hour"]))
            if month == 2 and day == 29:
                continue
            ts = pd.Timestamp(year=2001, month=month, day=day, hour=max(0, min(hour - 1, 23)))
            rows.append({"Date/Time": ts, "T_mean_C": float(r["DryBulb"]), "T_max_C": float(r["DryBulb"]), "RH_mean_pct": float(r["RH"]), "GHI_mean_Wm2": max(float(r["GHI"]), 0.0)})
        except Exception:
            continue
    if not rows:
        raise ValueError("No valid hourly rows parsed from EPW file.")
    return _prepare_timeseries_from_timestamped_df(pd.DataFrame(rows), time_step_hours)


def read_weather_csv_timeseries(csv_path: str | Path, time_step_hours: float = 24.0) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return ensure_weather_timeseries(df, time_step_hours)


def read_weather_auto_timeseries(path: str | Path, time_step_hours: float = 24.0) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".epw":
        return read_epw_timeseries(path, time_step_hours)
    if path.suffix.lower() in [".csv", ".txt"]:
        return read_weather_csv_timeseries(path, time_step_hours)
    raise ValueError("Unsupported weather upload. Use EPW or CSV.")


def ensure_weather_timeseries(df: pd.DataFrame, time_step_hours: float = 24.0) -> pd.DataFrame:
    """Accept daily or timestamped weather and return native simulation time-step rows."""
    if df is None or len(df) == 0:
        raise ValueError("Weather dataframe is empty.")
    work = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].copy()
    aliases = {
        "T_amb_C": "T_mean_C",
        "Outdoor Dry-Bulb Temperature": "T_mean_C",
        "DryBulb": "T_mean_C",
        "temperature": "T_mean_C",
        "temp": "T_mean_C",
        "Relative Humidity": "RH_mean_pct",
        "RH_pct": "RH_mean_pct",
        "RH": "RH_mean_pct",
        "humidity": "RH_mean_pct",
        "Global Solar Radiation": "GHI_mean_Wm2",
        "Global Horizontal Solar": "GHI_mean_Wm2",
        "GHI_Wm2": "GHI_mean_Wm2",
        "GHI": "GHI_mean_Wm2",
        "solar": "GHI_mean_Wm2",
    }
    lower_map = {str(c).strip().lower(): c for c in work.columns}
    for old, new in aliases.items():
        key = old.lower()
        if key in lower_map and new not in work.columns:
            work[new] = work[lower_map[key]]
    # Fuzzy column search for practical CSVs.
    def _find(names):
        for c in work.columns:
            low = str(c).strip().lower()
            if any(n.lower() == low or n.lower() in low for n in names):
                return c
        return None
    date_col = _find(["Date/Time", "date", "datetime", "timestamp", "time"])
    if date_col is not None and date_col != "Date/Time":
        work["Date/Time"] = work[date_col]
    if "T_mean_C" not in work.columns:
        c = _find(["dry-bulb", "temperature", "temp", "outdoor dry"])
        if c is not None:
            work["T_mean_C"] = work[c]
    if "RH_mean_pct" not in work.columns:
        c = _find(["relative humidity", "humidity", "rh"])
        work["RH_mean_pct"] = work[c] if c is not None else 60.0
    if "GHI_mean_Wm2" not in work.columns:
        c = _find(["global horizontal", "global solar", "ghi", "solar"])
        work["GHI_mean_Wm2"] = work[c] if c is not None else 0.0
    if "T_max_C" not in work.columns and "T_mean_C" in work.columns:
        work["T_max_C"] = work["T_mean_C"]
    if "Date/Time" in work.columns:
        parsed = pd.to_datetime(work["Date/Time"], errors="coerce")
        if parsed.notna().sum() >= max(3, int(0.5 * len(work))):
            work["Date/Time"] = parsed
            return _prepare_timeseries_from_timestamped_df(work, time_step_hours)
    # Fall back to daily normalization then expand.
    return expand_daily_weather_to_timeseries(ensure_365_daily_weather(work), time_step_hours)


def weather_steps_per_year(base_weather: pd.DataFrame, time_step_hours: float) -> int:
    return int(len(base_weather)) if base_weather is not None and len(base_weather) > 0 else steps_per_year_from_hours(time_step_hours)


def step_time_fields_from_weather(step: int, time_step_hours: float, base_weather: pd.DataFrame) -> Dict[str, float | int]:
    hours = resolve_time_step_hours(time_step_hours)
    n = weather_steps_per_year(base_weather, hours)
    row = base_weather.iloc[int(step % n)]
    doy = int(row.get("day_of_year", int(math.floor((step % n) * hours / 24.0)) + 1))
    hour = float(row.get("hour_of_day", ((step % n) * hours) % 24.0))
    year = int(step // n) + 1
    elapsed_days = (year - 1) * 365.0 + (doy - 1) + hour / 24.0
    return {
        "step": int(step + 1),
        "elapsed_days": float(elapsed_days),
        "day": int((year - 1) * 365 + doy),
        "year": int(year),
        "day_of_year": int(doy),
        "hour_of_day": float(hour),
        "time_step_hours": float(hours),
        "time_scale_days": float(hours / 24.0),
    }


def weather_summary_dict(df: pd.DataFrame, source: str, epw_path: str | None) -> Dict[str, object]:
    return {
        "source_mode": source,
        "epw_path": epw_path,
        "n_records": int(len(df)),
        "n_days": int(df["day_of_year"].nunique()) if "day_of_year" in df.columns else int(len(df)),
        "time_step_hours": float(df["time_step_hours"].iloc[0]) if "time_step_hours" in df.columns and len(df) else 24.0,
        "native_resolution": str(df["weather_native_resolution"].iloc[0]) if "weather_native_resolution" in df.columns and len(df) else "daily",
        "T_mean_annual_avg_C": float(df["T_mean_C"].mean()),
        "T_max_annual_avg_C": float(df["T_max_C"].mean()),
        "RH_annual_avg_pct": float(df["RH_mean_pct"].mean()),
        "GHI_annual_avg_Wm2": float(df["GHI_mean_Wm2"].mean()),
    }


def _calendar_occupancy_factor(doy: int, schedule_profile: Optional[Dict[str, float]] = None) -> float:
    week = ((doy - 1) // 7) % 52
    in_sem = (1 <= week <= 16) or (20 <= week <= 34)
    is_summer = 180 <= doy <= 242
    if schedule_profile is None:
        return float(0.80 if in_sem else (0.35 if is_summer else 0.25))
    occ = schedule_profile["term_factor"] if in_sem else (schedule_profile["summer_factor"] if is_summer else schedule_profile["break_factor"])
    return float(np.clip(occ, 0.0, 1.0))


def _hourly_occupancy_multiplier(doy: int, hour: float, time_step_hours: float) -> float:
    if time_step_hours >= 24.0:
        return 1.0
    weekday = ((doy - 1) % 7) < 5
    if not weekday:
        return 0.20
    h = (hour + time_step_hours / 2.0) % 24.0
    if 7.0 <= h <= 19.0:
        return float(0.35 + 0.65 * max(math.sin(math.pi * (h - 7.0) / 12.0), 0.0))
    return 0.10


def climate_and_operation_for_step(step: int, time_step_hours: float, base_weather: pd.DataFrame, climate_name: str, schedule_profile: Optional[Dict[str, float]] = None) -> Tuple[float, float, float, float, float]:
    hours = resolve_time_step_hours(time_step_hours)
    n = weather_steps_per_year(base_weather, hours)
    year_idx = step // n
    row = base_weather.iloc[int(step % n)]
    doy = int(row.get("day_of_year", int(math.floor((step % n) * hours / 24.0)) + 1))
    hour = float(row.get("hour_of_day", ((step % n) * hours) % 24.0))
    rules = CLIMATE_LEVELS[climate_name]
    pulse = 0.0
    if 150 <= doy <= 260:
        phase = (doy - 150) / (260 - 150) * math.pi
        pulse = rules["summer_pulse"] * math.sin(phase)
    T_mean = float(row["T_mean_C"] + rules["temp_shift"] + pulse + rules["future_drift_per_year"] * year_idx)
    T_max = float(row.get("T_max_C", row["T_mean_C"]) + rules["temp_shift"] + 1.15 * pulse + 1.2 * rules["future_drift_per_year"] * year_idx)
    RH_mean = float(np.clip(row["RH_mean_pct"] + rules["rh_shift"], 15, 95))
    GHI_mean = float(max(0.0, row["GHI_mean_Wm2"] * rules["solar_mult"]))
    occ = _calendar_occupancy_factor(doy, schedule_profile) * _hourly_occupancy_multiplier(doy, hour, hours)
    occ = float(np.clip(occ, 0.0, 1.0))
    return T_mean, T_max, RH_mean, GHI_mean, occ


def climate_and_operation_for_day(d: int, base_weather: pd.DataFrame, climate_name: str, schedule_profile: Optional[Dict[str, float]] = None) -> Tuple[float, float, float, float, float]:
    """Backward-compatible daily wrapper."""
    daily = base_weather
    if "time_step_hours" in daily.columns and len(daily) != 365:
        # choose the first record belonging to the requested day
        idx = int(d % 365) + 1
        subset = daily[daily["day_of_year"] == idx]
        if not subset.empty:
            row = subset.iloc[0]
            return climate_and_operation_for_step(int(row.get("step_of_year", 1)) - 1, float(row.get("time_step_hours", 24.0)), daily, climate_name, schedule_profile)
    return climate_and_operation_for_step(int(d), 24.0, expand_daily_weather_to_timeseries(ensure_365_daily_weather(daily), 24.0), climate_name, schedule_profile)


def apply_severity(cfg: HVACConfig, severity: str) -> HVACConfig:
    rules = SEVERITY_LEVELS[severity]
    out = HVACConfig(**asdict(cfg))
    out.B_FOUL *= rules["B_FOUL_mult"]
    out.DUST_RATE *= rules["DUST_RATE_mult"]
    out.COP_AGING_RATE *= rules["COP_AGING_RATE_mult"]
    out.RF_STAR *= rules["RF_STAR_mult"]
    out.K_CLOG *= rules["K_CLOG_mult"]
    out.DEG_TRIGGER = float(np.clip(out.DEG_TRIGGER + rules["DEG_TRIGGER_shift"], 0.35, 0.75))
    return out


def degradation_index(cfg: HVACConfig, rf: float, dust: float) -> Tuple[float, float]:
    dp = min(cfg.DP_CLEAN + cfg.K_CLOG * dust, cfg.DP_MAX)
    deg = 0.5 * (rf / max(cfg.RF_STAR, 1e-12)) + 0.5 * (dp / cfg.DP_MAX)
    return dp, deg



def _safe_clip(value: float, lo: float, hi: float) -> float:
    try:
        return float(np.clip(float(value), lo, hi))
    except Exception:
        return float(lo)


def saturation_vapor_pressure_pa(T_C: float) -> float:
    """Tetens equation for saturation vapor pressure over water, Pa."""
    T_C = float(T_C)
    return float(610.94 * math.exp((17.625 * T_C) / (T_C + 243.04)))


def humidity_ratio_kgkg(T_C: float, RH_pct: float, pressure_pa: float = 101325.0) -> float:
    RH = _safe_clip(float(RH_pct) / 100.0, 0.0, 1.0)
    p_ws = saturation_vapor_pressure_pa(T_C)
    p_v = min(RH * p_ws, 0.98 * pressure_pa)
    return float(0.62198 * p_v / max(pressure_pa - p_v, 1e-9))


def estimate_latent_cooling_kw(bldg: BuildingSpec, cfg: HVACConfig, derived: Dict[str, float], T_mean: float, RH_mean: float, occ: float, af: float = 1.0) -> float:
    """Return latent cooling load in kW for outdoor air moisture removal.

    This is only applied to the core load when APPLY_LATENT_LOAD_TO_CORE is True.
    """
    if not getattr(cfg, "APPLY_LATENT_LOAD_TO_CORE", False):
        return 0.0
    w_out = humidity_ratio_kgkg(T_mean, RH_mean, getattr(cfg, "ATM_PRESSURE_PA", 101325.0))
    w_in = humidity_ratio_kgkg(float(getattr(cfg, "T_SET", 23.0)), getattr(cfg, "INDOOR_RH_TARGET_PCT", 50.0), getattr(cfg, "ATM_PRESSURE_PA", 101325.0))
    dw = max(w_out - w_in, 0.0)
    if dw <= 0:
        return 0.0
    rho_air = getattr(cfg, "HX_AIR_DENSITY_KG_M3", 1.2)
    # Ventilation portion of mechanical airflow plus a simple infiltration equivalent.
    vent_frac = _safe_clip(getattr(cfg, "LATENT_VENTILATION_FRACTION", 0.35), 0.0, 1.0)
    mech_air_m3h = derived.get("Q_air_nom_m3h", bldg.conditioned_area_m2 * bldg.airflow_m3h_m2) * max(float(af), 0.0) * max(float(occ), 0.0) * vent_frac
    volume_m3 = bldg.conditioned_area_m2 * getattr(cfg, "FLOOR_TO_FLOOR_M", 3.2)
    infil_air_m3h = bldg.infiltration_ach * volume_m3 if getattr(cfg, "USE_INFILTRATION", True) else 0.0
    m_air = (mech_air_m3h + infil_air_m3h) / 3600.0 * rho_air
    h_fg = getattr(cfg, "LATENT_HEAT_VAPORIZATION_KJ_KG", 2501.0)
    return float(max(m_air * h_fg * dw, 0.0))


def part_load_modifier(cfg: HVACConfig, plr: float) -> float:
    if not getattr(cfg, "APPLY_PART_LOAD_COP_TO_CORE", False):
        return 1.0
    x = _safe_clip(plr, 0.0, 1.5)
    curve_type = str(getattr(cfg, "PLR_CURVE_TYPE", "Quadratic"))
    a, b, c, d = float(getattr(cfg, "PLR_A", 0.85)), float(getattr(cfg, "PLR_B", 0.25)), float(getattr(cfg, "PLR_C", -0.10)), float(getattr(cfg, "PLR_D", 0.0))
    if curve_type.lower().startswith("linear"):
        mod = a + b * x
    elif curve_type.lower().startswith("cubic"):
        mod = a + b * x + c * x**2 + d * x**3
    else:
        mod = a + b * x + c * x**2
    return _safe_clip(mod, getattr(cfg, "PLR_MIN_MODIFIER", 0.55), getattr(cfg, "PLR_MAX_MODIFIER", 1.15))


def apply_part_load_cop(cfg: HVACConfig, cop: float, q_hvac_kw: float, mode: str, derived: Dict[str, float]) -> Tuple[float, float, float]:
    design = derived.get("Q_cool_des_kw", 0.0) if mode == "cooling" else derived.get("Q_heat_des_kw", 0.0)
    plr = float(q_hvac_kw) / max(float(design), 1e-9)
    mod = part_load_modifier(cfg, plr)
    return max(0.8, float(cop) * mod), plr, mod


def hx_air_pressure_pa(cfg: HVACConfig, base_dp_pa: float, af: float, deg: float) -> float:
    if not getattr(cfg, "APPLY_HX_AIR_PRESSURE_TO_FAN", False):
        return float(base_dp_pa)
    dp = float(getattr(cfg, "DP_CLEAN", base_dp_pa)) * max(float(af), 0.05) ** 2
    dp *= 1.0 + float(getattr(cfg, "HX_AIR_FOULING_FACTOR", 0.75)) * max(float(deg), 0.0)
    return float(min(max(dp, getattr(cfg, "DP_CLEAN", 150.0)), max(getattr(cfg, "DP_MAX", 450.0) * 2.0, dp)))


def hx_water_pump_terms(bldg: BuildingSpec, cfg: HVACConfig, q_hvac_kw: float, mode: str, deg: float) -> Dict[str, float]:
    """Detailed water-side pump estimate. Returns P_pump, dP_water_kPa, water_flow_m3h."""
    if not getattr(cfg, "APPLY_HX_WATER_PRESSURE_TO_PUMP", False):
        return {}
    cp = float(getattr(cfg, "HX_CP_WATER_KJ_KG_K", 4.186))
    rho = float(getattr(cfg, "HX_WATER_DENSITY_KG_M3", 997.0))
    dt = float(getattr(cfg, "HX_CHW_DT_K", 5.0) if mode == "cooling" else getattr(cfg, "HX_HW_DT_K", 10.0))
    user_flow = float(getattr(cfg, "HX_WATER_FLOW_M3H", 0.0))
    if user_flow > 0:
        flow_m3h = user_flow
    else:
        m_kg_s = max(float(q_hvac_kw), 0.0) / max(cp * dt, 1e-9)
        flow_m3h = m_kg_s / max(rho, 1e-9) * 3600.0
    nominal_flow = float(getattr(cfg, "HX_WATER_FLOW_NOM_M3H", 0.0)) or max(flow_m3h, 1e-9)
    flow_ratio = flow_m3h / max(nominal_flow, 1e-9)
    dp_kpa = float(getattr(cfg, "HX_WATER_DP_CLEAN_KPA", 35.0)) * flow_ratio ** 2 * (1.0 + float(getattr(cfg, "HX_WATER_FOULING_FACTOR", 0.35)) * max(float(deg), 0.0))
    flow_m3s = flow_m3h / 3600.0
    pump_eff = max(float(getattr(cfg, "HX_PUMP_EFF", 0.65)), 1e-6)
    p_pump_kw = flow_m3s * dp_kpa * 1000.0 / pump_eff / 1000.0
    return {"P_pump": float(max(p_pump_kw, 0.0)), "dP_water_kPa": float(dp_kpa), "water_flow_m3h": float(flow_m3h)}


def coupled_module_notes(cfg: HVACConfig) -> str:
    active = []
    for attr, label in [
        ("APPLY_PART_LOAD_COP_TO_CORE", "part-load COP"),
        ("APPLY_LATENT_LOAD_TO_CORE", "latent cooling"),
        ("APPLY_HX_AIR_PRESSURE_TO_FAN", "HX air pressure → fan"),
        ("APPLY_HX_WATER_PRESSURE_TO_PUMP", "HX water pressure → pump"),
        ("APPLY_HX_UA_TO_CAPACITY", "HX UA capacity"),
        ("APPLY_NATIVE_ZONE_LOADS", "native zone loads"),
    ]:
        if getattr(cfg, attr, False):
            active.append(label)
    return ", ".join(active) if active else "none"


def apply_core_coupled_corrections(
    bldg: BuildingSpec,
    cfg: HVACConfig,
    derived: Dict[str, float],
    res: Dict[str, float],
    loads: Dict[str, float],
    T_mean: float,
    RH_mean: float,
    occ: float,
    T_sp: float,
    af: float,
    duration_hours: float,
) -> Dict[str, float]:
    """Apply optional publication-level coupled corrections to an already formed timestep result."""
    q_hvac = float(res.get("Q_HVAC_kw", loads.get("Q_HVAC_kw", 0.0)))
    mode = str(res.get("mode", loads.get("mode", "cooling")))
    deg = float(res.get("deg_next", res.get("delta", 0.0)))
    cop_base = float(res.get("cop", 1.0))
    cop_corr, plr, plr_mod = apply_part_load_cop(cfg, cop_base, q_hvac, mode, derived)
    res["COP_base_before_PLR"] = cop_base
    res["PLR"] = plr
    res["PLR_modifier"] = plr_mod
    res["cop"] = cop_corr
    res["P_hvac"] = q_hvac / max(cop_corr, 0.8)

    dp_base = float(res.get("dp_next", getattr(cfg, "DP_CLEAN", 150.0)))
    dp_fan = hx_air_pressure_pa(cfg, dp_base, af, deg)
    res["dP_fan_Pa"] = dp_fan
    if getattr(cfg, "USE_HVAC_FANS", True):
        res["P_fan"] = (derived["Q_air_nom_m3h"] * af / 3600.0 * dp_fan / max(cfg.FAN_EFF, 1e-6)) / 1000.0
    else:
        res["P_fan"] = 0.0

    power_terms = auxiliary_power_terms(bldg, cfg, occ, deg, af=af, q_hvac_kw=q_hvac, mode=mode)
    res["P_pump"] = power_terms.get("P_pump", res.get("P_pump", 0.0))
    res["P_aux"] = power_terms.get("P_aux", res.get("P_aux", 0.0))
    res["dP_water_kPa"] = power_terms.get("dP_water_kPa", 0.0)
    res["water_flow_m3h"] = power_terms.get("water_flow_m3h", 0.0)

    p_tot = float(res.get("P_hvac", 0.0)) + float(res.get("P_fan", 0.0)) + float(res.get("P_pump", 0.0)) + float(res.get("P_aux", 0.0))
    e_period = p_tot * duration_hours
    res["P_tot"] = p_tot
    res["E_hvac"] = float(res.get("P_hvac", 0.0)) * duration_hours
    res["E_fan"] = float(res.get("P_fan", 0.0)) * duration_hours
    res["E_pump"] = float(res.get("P_pump", 0.0)) * duration_hours
    res["E_aux"] = float(res.get("P_aux", 0.0)) * duration_hours
    res["E_period"] = e_period
    res["E_day"] = e_period
    res["co2"] = e_period * cfg.CO2_FACTOR if getattr(cfg, "USE_CARBON", True) else 0.0

    e_n = e_period / max((derived["Q_cool_des_kw"] * duration_hours * 1.5), 1e-9)
    d_n = deg
    c_n = float(res.get("comfort_dev", 0.0)) / 3.0
    co2_n = float(res.get("co2", 0.0)) / max((derived["Q_cool_des_kw"] * cfg.CO2_FACTOR * duration_hours * 1.5), 1e-9)
    res["objective"] = cfg.W_ENERGY * e_n + cfg.W_DEGRAD * d_n + cfg.W_COMFORT * c_n + cfg.W_CARBON * co2_n
    res["coupled_modules_active"] = coupled_module_notes(cfg)
    res["latent_cooling_kw"] = loads.get("latent_cooling_kw", 0.0)
    res["zone_load_mode"] = loads.get("zone_load_mode", "building_aggregate")
    res["hx_capacity_factor"] = loads.get("hx_capacity_factor", 1.0)
    res["capacity_unmet_kw"] = loads.get("capacity_unmet_kw", 0.0)
    return res

def auxiliary_power_terms(bldg: BuildingSpec, cfg: HVACConfig, occ: float, deg: float, af: float = 1.0, q_hvac_kw: float = 0.0, mode: str = "cooling") -> Dict[str, float]:
    """Return pump and auxiliary electrical power terms in kW.

    Simple mode: area-normalized pump W/m².
    Coupled mode: optional HX water-side pressure drop replaces simple pump power.
    """
    operating_factor = max(float(occ), 0.35)
    pump_kw = 0.0
    aux_kw = 0.0
    out: Dict[str, float] = {}
    if getattr(cfg, "USE_HVAC_PUMPS", True):
        if getattr(cfg, "APPLY_HX_WATER_PRESSURE_TO_PUMP", False):
            hx_terms = hx_water_pump_terms(bldg, cfg, q_hvac_kw, mode, deg)
            pump_kw = float(hx_terms.get("P_pump", 0.0))
            out.update(hx_terms)
        else:
            pump_kw = bldg.conditioned_area_m2 * getattr(cfg, "PUMP_SPECIFIC_W_M2", 0.0) / 1000.0
            pump_kw *= operating_factor * (1.0 + 0.30 * max(float(deg), 0.0))
    if getattr(cfg, "USE_HVAC_AUXILIARY", True):
        aux_kw = bldg.conditioned_area_m2 * getattr(cfg, "AUXILIARY_W_M2", 0.0) / 1000.0
        aux_kw *= operating_factor
    out.update({"P_pump": float(pump_kw), "P_aux": float(aux_kw)})
    return out

def severity_scalar(severity: str) -> float:
    return {
        "Mild": 0.70,
        "Moderate": 1.00,
        "Severe": 1.35,
        "High": 1.60,
    }.get(severity, 1.0)


def weather_stress_scalar(T_mean: float, RH_mean: float, GHI_mean: float) -> float:
    temp_stress = max((T_mean - 24.0) / 12.0, 0.0)
    humid_stress = max((RH_mean - 60.0) / 30.0, 0.0)
    solar_stress = min(max(GHI_mean / 700.0, 0.0), 1.5)
    return 1.0 + 0.45 * temp_stress + 0.10 * humid_stress + 0.05 * solar_stress


def ts_degradation_update(
    cfg: HVACConfig,
    severity: str,
    prev_delta: float,
    T_mean: float,
    RH_mean: float,
    GHI_mean: float,
    model_name: str,
    time_scale_days: float = 1.0,
) -> Tuple[float, float, float, float]:
    if not getattr(cfg, "USE_DEGRADATION", True):
        return 0.0, 0.0, cfg.DP_CLEAN, 0.0
    sev_mult = severity_scalar(severity)
    stress = weather_stress_scalar(T_mean, RH_mean, GHI_mean)

    if model_name == "linear_ts":
        delta_next = min(1.0, prev_delta + cfg.LINEAR_DEG_PER_DAY * time_scale_days * sev_mult * stress)
    elif model_name == "exponential_ts":
        rate = cfg.EXP_DEG_RATE_PER_DAY * time_scale_days * sev_mult * stress
        delta_next = 1.0 - (1.0 - prev_delta) * math.exp(-rate)
    else:
        raise ValueError(f"Unsupported time-series degradation model: {model_name}")

    rf_next = min(cfg.RF_STAR, cfg.RF_STAR * min(delta_next * 1.20, 1.0))
    dp_next = min(cfg.DP_CLEAN + delta_next * (cfg.DP_MAX - cfg.DP_CLEAN), cfg.DP_MAX)
    dust_next = max((dp_next - cfg.DP_CLEAN) / max(cfg.K_CLOG, 1e-9), 0.0)
    return rf_next, dust_next, dp_next, delta_next

def resolve_time_step_hours(time_step: str | float | int | None = None, fallback: float = 24.0) -> float:
    """Return calculation time-step hours. Values must divide 24 for stable daily indexing."""
    if time_step is None:
        hours = fallback
    elif isinstance(time_step, str):
        if time_step in TIME_STEP_OPTIONS:
            hours = TIME_STEP_OPTIONS[time_step]
        else:
            hours = float(time_step)
    else:
        hours = float(time_step)
    allowed = [1.0, 3.0, 6.0, 12.0, 24.0]
    if hours not in allowed:
        raise ValueError(f"Unsupported time-step {hours} h. Use one of {allowed}.")
    return hours


def steps_per_year_from_hours(time_step_hours: float) -> int:
    return int(round(365 * 24 / resolve_time_step_hours(time_step_hours)))


def step_time_fields(step: int, time_step_hours: float) -> Dict[str, float | int]:
    elapsed_days = step * time_step_hours / 24.0
    day_index = int(math.floor(elapsed_days))
    doy = (day_index % 365) + 1
    year = (day_index // 365) + 1
    return {
        "step": int(step + 1),
        "elapsed_days": float(elapsed_days),
        "day": int(day_index + 1),
        "year": int(year),
        "day_of_year": int(doy),
        "time_step_hours": float(time_step_hours),
        "time_scale_days": float(time_step_hours / 24.0),
    }


def _clone_bldg(bldg: BuildingSpec) -> BuildingSpec:
    return BuildingSpec(**asdict(bldg))


def _clone_cfg(cfg: HVACConfig) -> HVACConfig:
    return HVACConfig(**asdict(cfg))


def _load_base_weather(
    weather_mode: str = "synthetic",
    epw_path: str | None = None,
    csv_path: str | None = None,
    weather_df: Optional[pd.DataFrame] = None,
    random_state: int = 42,
    time_step_hours: float = 24.0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    hours = resolve_time_step_hours(time_step_hours)
    if weather_df is not None:
        base_weather = ensure_weather_timeseries(weather_df, hours)
        return base_weather, weather_summary_dict(base_weather, "uploaded_dataframe_native_timeseries", None)
    if weather_mode == "epw" and epw_path:
        base_weather = read_epw_timeseries(epw_path, hours)
        return base_weather, weather_summary_dict(base_weather, "epw_native_timeseries", epw_path)
    if weather_mode == "csv" and csv_path:
        base_weather = read_weather_csv_timeseries(csv_path, hours)
        return base_weather, weather_summary_dict(base_weather, "csv_native_timeseries", csv_path)
    if weather_mode == "uploaded" and epw_path:
        base_weather = read_weather_auto_timeseries(epw_path, hours)
        return base_weather, weather_summary_dict(base_weather, "uploaded_native_timeseries", epw_path)
    base_weather = synthetic_weather_timeseries(hours, random_state)
    return base_weather, weather_summary_dict(base_weather, "synthetic_timeseries", None)



def _hour_in_window(hour: float, start: float, end: float) -> bool:
    """Return True if hour is inside a possibly overnight control window."""
    hour = float(hour) % 24.0
    start = float(start) % 24.0
    end = float(end) % 24.0
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _lookup_operation_schedule(operation_schedule_df: Optional[pd.DataFrame], hour_of_day: float, day_of_week: int) -> Dict[str, object]:
    """Return a custom schedule row for the current hour/day when available.

    Expected optional columns: day_type, start_hour, end_hour, occupied, occ_multiplier,
    setpoint_shift_C, airflow_factor, cooling_allowed, heating_allowed, demand_response.
    day_type may be Weekday, Weekend, All, or Custom.
    """
    if operation_schedule_df is None or len(operation_schedule_df) == 0:
        return {}
    df = operation_schedule_df.copy()
    cols = {str(c).strip().lower(): c for c in df.columns}
    sh = cols.get("start_hour")
    eh = cols.get("end_hour")
    if sh is None or eh is None:
        return {}
    day_type_col = cols.get("day_type")
    is_weekend = int(day_of_week) >= 5
    for _, row in df.iterrows():
        day_type = str(row[day_type_col]).strip().lower() if day_type_col else "all"
        if day_type not in ("all", "custom"):
            if is_weekend and "weekend" not in day_type:
                continue
            if (not is_weekend) and "weekday" not in day_type:
                continue
        try:
            if _hour_in_window(hour_of_day, float(row[sh]), float(row[eh])):
                return row.to_dict()
        except Exception:
            continue
    return {}


def apply_ems_control(
    cfg: HVACConfig,
    T_mean: float,
    occ: float,
    hour_of_day: float,
    day_of_week: int,
    T_sp: float,
    af: float,
    operation_schedule_df: Optional[pd.DataFrame] = None,
) -> Tuple[float, float, Dict[str, object]]:
    """Apply EMS/schedule overlays to setpoint and airflow.

    The default settings leave the original S0-S3 calculation unchanged. EMS rules are
    intentionally simple and transparent so they can be reported in a publication.
    """
    flags: Dict[str, object] = {
        "ems_active": 0,
        "ems_mode_applied": "None",
        "ems_occ_control": 0,
        "ems_night_setback": 0,
        "ems_demand_response": 0,
        "ems_economizer": 0,
        "ems_custom_schedule": 0,
        "ems_optimum_start": 0,
    }

    mode = str(getattr(cfg, "EMS_MODE", "Disabled") or "Disabled")
    mode_l = mode.lower()
    if mode_l in ("disabled", "none", "off"):
        return float(T_sp), float(af), flags

    flags["ems_active"] = 1
    flags["ems_mode_applied"] = mode

    # Custom operation schedule has priority because it represents explicit user intent.
    sched = _lookup_operation_schedule(operation_schedule_df, hour_of_day, day_of_week)
    if getattr(cfg, "EMS_CUSTOM_SCHEDULE_ENABLED", False) and sched:
        flags["ems_custom_schedule"] = 1
        try:
            T_sp += float(sched.get("setpoint_shift_C", 0.0))
        except Exception:
            pass
        try:
            af *= float(sched.get("airflow_factor", 1.0))
        except Exception:
            pass
        try:
            occ_mult = float(sched.get("occ_multiplier", 1.0))
            flags["ems_schedule_occ_multiplier"] = occ_mult
        except Exception:
            pass
        if bool(sched.get("demand_response", False)):
            flags["ems_demand_response"] = 1

    if getattr(cfg, "EMS_OCC_CONTROL", False) or "occupancy" in mode_l or "hybrid" in mode_l:
        if occ <= float(getattr(cfg, "EMS_LOW_OCC_THRESHOLD", 0.25)):
            flags["ems_occ_control"] = 1
            T_sp += float(getattr(cfg, "EMS_LOW_OCC_SETPOINT_SHIFT_C", 1.0))
            af *= float(getattr(cfg, "EMS_LOW_OCC_AIRFLOW_FACTOR", 0.65))

    if getattr(cfg, "EMS_NIGHT_SETBACK", False) or "night" in mode_l or "hybrid" in mode_l:
        if _hour_in_window(hour_of_day, getattr(cfg, "EMS_NIGHT_START_HOUR", 19.0), getattr(cfg, "EMS_NIGHT_END_HOUR", 6.0)):
            flags["ems_night_setback"] = 1
            T_sp += float(getattr(cfg, "EMS_NIGHT_SETPOINT_SHIFT_C", 2.0))
            af *= float(getattr(cfg, "EMS_NIGHT_AIRFLOW_FACTOR", 0.55))

    if getattr(cfg, "EMS_DEMAND_RESPONSE", False) or "demand" in mode_l or "hybrid" in mode_l:
        if _hour_in_window(hour_of_day, getattr(cfg, "EMS_DR_START_HOUR", 13.0), getattr(cfg, "EMS_DR_END_HOUR", 17.0)):
            flags["ems_demand_response"] = 1
            T_sp += float(getattr(cfg, "EMS_DR_SETPOINT_SHIFT_C", 1.5))
            af *= max(0.10, 1.0 - float(getattr(cfg, "EMS_DR_AIRFLOW_REDUCTION", 0.15)))

    if getattr(cfg, "EMS_ECONOMIZER", False) or "economizer" in mode_l or "hybrid" in mode_l:
        lo = float(getattr(cfg, "EMS_ECONOMIZER_TEMP_LOW_C", 16.0))
        hi = float(getattr(cfg, "EMS_ECONOMIZER_TEMP_HIGH_C", 22.0))
        if lo <= float(T_mean) <= hi and float(occ) > 0.05:
            flags["ems_economizer"] = 1
            # Encourage outdoor-air use but keep within AF_MAX; cooling reduction is applied later.
            af = max(af, min(float(getattr(cfg, "AF_MAX", 1.0)), 0.95))

    if getattr(cfg, "EMS_OPTIMUM_START", False) or "optimum" in mode_l or "hybrid" in mode_l:
        start_h = float(getattr(cfg, "EMS_OPTIMUM_START_HOUR", 7.0))
        if start_h - 1.0 <= float(hour_of_day) < start_h:
            flags["ems_optimum_start"] = 1
            T_sp += float(getattr(cfg, "EMS_PRECOOL_SHIFT_C", -0.8))
            af = max(af, 0.85)

    T_sp = float(np.clip(T_sp, getattr(cfg, "T_SP_MIN", 16.0), getattr(cfg, "T_SP_MAX", 30.0)))
    af = float(np.clip(af, getattr(cfg, "AF_MIN", 0.1), getattr(cfg, "AF_MAX", 1.5)))
    return T_sp, af, flags

def simulate_baseline_no_degradation(
    strategy: str,
    climate_name: str,
    bldg: BuildingSpec,
    base_cfg: HVACConfig,
    base_weather: pd.DataFrame,
    schedule_profile: Optional[Dict[str, float]] = None,
    random_state: int = 42,
):
    cfg = apply_hvac_preset(HVACConfig(**asdict(base_cfg)))
    derived = derive_building_numbers(bldg)
    duration_hours = resolve_time_step_hours(getattr(cfg, "TIME_STEP_HOURS", 24.0))
    steps_per_year = weather_steps_per_year(base_weather, duration_hours)
    rng = np.random.default_rng(random_state)

    daily_rows = []
    for step in range(cfg.years * steps_per_year):
        tf = step_time_fields_from_weather(step, duration_hours, base_weather)
        day_index = int(tf["day"] - 1)
        year = int(tf["year"])
        doy = int(tf["day_of_year"])
        T_mean, T_max, RH_mean, GHI_mean, occ = climate_and_operation_for_step(
            step, duration_hours, base_weather, climate_name, schedule_profile
        )

        T_sp = cfg.T_SET
        af = 1.0
        loads = cooling_heating_loads(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, T_sp, occ, doy)
        mode = loads["mode"]
        current_cop = cop_cooling(cfg, T_mean, 0.0, 0.0) if mode == "cooling" else cop_heating(cfg, T_mean, 0.0, 0.0)
        P_hvac = loads["Q_HVAC_kw"] / max(current_cop, 0.8)
        P_fan = 0.0
        if getattr(cfg, "USE_HVAC_FANS", True):
            P_fan = (derived["Q_air_nom_m3h"] * af / 3600.0 * cfg.DP_CLEAN / max(cfg.FAN_EFF, 1e-6)) / 1000.0
        power_terms = auxiliary_power_terms(bldg, cfg, occ, 0.0)
        P_pump = power_terms["P_pump"]
        P_aux = power_terms["P_aux"]
        P_tot = P_hvac + P_fan + P_pump + P_aux
        E_period = P_tot * duration_hours
        co2 = E_period * cfg.CO2_FACTOR if getattr(cfg, "USE_CARBON", True) else 0.0
        T_zone = (
            T_sp + 2.2 * (1.0 - af) * occ + 0.08 * max(T_mean - T_sp, 0.0)
            - 0.06 * max(T_sp - T_mean, 0.0) + cfg.HUMIDITY_COMFORT_FACTOR * max(RH_mean - 60.0, 0.0)
        )
        comfort_dev = abs(T_zone - cfg.T_SET)
        discomfort_flag = int((occ > 0.5) and (comfort_dev > 0.3))
        row = {
            "strategy": strategy,
            "severity": "Baseline_NoDegradation",
            "climate": climate_name,
            "scenario_combo_3axis": f"BASELINE_{strategy}_{climate_name}",
            "building_type": bldg.building_type,
            "area_m2": bldg.conditioned_area_m2,
            "floors": bldg.floors,
            "n_spaces": bldg.n_spaces,
            "hvac_system_type": cfg.hvac_system_type,
            "Q_cool_des_kw": derived["Q_cool_des_kw"],
            "Q_heat_des_kw": derived["Q_heat_des_kw"],
            "Q_air_nom_m3h": derived["Q_air_nom_m3h"],
            **tf,
            "T_amb_C": T_mean,
            "T_max_C": T_max,
            "RH_mean_pct": RH_mean,
            "GHI_mean_Wm2": GHI_mean,
            "occ": occ,
            "T_sp_C": T_sp,
            "alpha_flow": af,
            "R_f": 0.0,
            "dust_kg": 0.0,
            "dP_Pa": cfg.DP_CLEAN,
            "delta": 0.0,
            "COP_eff": current_cop,
            "mode": mode,
            "Q_cool_kw": loads["Q_cool_kw"],
            "Q_heat_kw": loads["Q_heat_kw"],
            "Q_HVAC_kw": loads["Q_HVAC_kw"],
            "P_hvac_kw": P_hvac,
            "P_fan_kw": P_fan,
            "P_pump_kw": P_pump,
            "P_auxiliary_kw": P_aux,
            "P_total_kw": P_tot,
            "thermal_hvac_kwh_period": P_hvac * duration_hours,
            "fan_kwh_period": P_fan * duration_hours,
            "pump_kwh_period": P_pump * duration_hours,
            "auxiliary_kwh_period": P_aux * duration_hours,
            "people_count": derived["N_people_max"] * occ,
            "internal_gains_kw": loads.get("internal_kw", 0.0),
            "sensible_people_kw": derived["N_people_max"] * occ * bldg.sensible_w_per_person / 1000.0,
            "energy_kwh_period": E_period,
            "energy_kwh_day": E_period,
            "co2_kg_period": co2,
            "co2_kg_day": co2,
            "comfort_dev_C": comfort_dev,
            "occupied_discomfort_flag": discomfort_flag,
            "occupied_discomfort_day_equiv": discomfort_flag * duration_hours / 24.0,
            "cost_usd_period": E_period * cfg.E_PRICE,
            "cost_usd_day": E_period * cfg.E_PRICE,
            "hx_cleaned": 0,
            "filter_replaced": 0,
            "baseline_flag": 1,
        }
        daily_rows.append(row)

    daily = pd.DataFrame(daily_rows)
    annual = daily.groupby(["strategy", "severity", "climate", "year"], as_index=False).agg(
        annual_energy_MWh=("energy_kwh_period", lambda ss: float(ss.sum() / 1000.0)),
        annual_thermal_hvac_MWh=("thermal_hvac_kwh_period", lambda ss: float(ss.sum() / 1000.0)),
        annual_fan_MWh=("fan_kwh_period", lambda ss: float(ss.sum() / 1000.0)),
        annual_pump_MWh=("pump_kwh_period", lambda ss: float(ss.sum() / 1000.0)),
        annual_auxiliary_MWh=("auxiliary_kwh_period", lambda ss: float(ss.sum() / 1000.0)),
        annual_cost_usd=("cost_usd_period", "sum"),
        annual_co2_tonne=("co2_kg_period", lambda ss: float(ss.sum() / 1000.0)),
        mean_COP=("COP_eff", "mean"),
        mean_delta=("delta", "mean"),
        mean_comfort_dev=("comfort_dev_C", "mean"),
        mean_Q_cool_kw=("Q_cool_kw", "mean"),
        mean_Q_heat_kw=("Q_heat_kw", "mean"),
        occupied_discomfort_days=("occupied_discomfort_day_equiv", "sum"),
    )
    summary = {
        "strategy": strategy,
        "severity": "Baseline_NoDegradation",
        "climate": climate_name,
        "scenario_combo_3axis": f"BASELINE_{strategy}_{climate_name}",
        "Building Area m2": bldg.conditioned_area_m2,
        "No. of Spaces": bldg.n_spaces,
        "HVAC System": cfg.hvac_system_type,
        "Cooling Design kW": derived["Q_cool_des_kw"],
        "Heating Design kW": derived["Q_heat_des_kw"],
        "Airflow m3h": derived["Q_air_nom_m3h"],
        "Time Step Hours": duration_hours,
        "Total Energy MWh": float(daily["energy_kwh_period"].sum() / 1000.0),
        "Total Thermal HVAC Energy MWh": float(daily["thermal_hvac_kwh_period"].sum() / 1000.0),
        "Total Fan Energy MWh": float(daily["fan_kwh_period"].sum() / 1000.0),
        "Total Pump Energy MWh": float(daily["pump_kwh_period"].sum() / 1000.0),
        "Total Auxiliary Energy MWh": float(daily["auxiliary_kwh_period"].sum() / 1000.0),
        "Total Cost USD": float(daily["cost_usd_period"].sum()),
        "Total CO2 tonne": float(daily["co2_kg_period"].sum() / 1000.0),
        "Mean COP": float(daily["COP_eff"].mean()),
        "Mean Degradation Index": 0.0,
        "Mean Comfort Deviation C": float(daily["comfort_dev_C"].mean()),
        "Mean Cooling Load kW": float(daily["Q_cool_kw"].mean()),
        "Mean Heating Load kW": float(daily["Q_heat_kw"].mean()),
        "Occupied Discomfort Days": float(daily["occupied_discomfort_day_equiv"].sum()),
        "Filter Replacements count": 0,
        "HX Cleanings count": 0,
    }
    return daily, annual, summary

def cop_cooling(cfg: HVACConfig, T_a: float, year_frac: float, rf: float) -> float:
    cop_aged = cfg.COP_COOL_NOM - cfg.COP_AGING_RATE * year_frac
    cop_foul = cop_aged / (1.0 + 0.45 * (rf / max(cfg.RF_STAR, 1e-12)))
    cop_amb = 1.0 - 0.018 * max(T_a - 25.0, 0.0)
    return min(cfg.COP_COOL_NOM, max(0.8, cop_foul * cop_amb))


def cop_heating(cfg: HVACConfig, T_a: float, year_frac: float, rf: float) -> float:
    cop_aged = cfg.COP_HEAT_NOM - 0.6 * cfg.COP_AGING_RATE * year_frac
    cop_foul = cop_aged / (1.0 + 0.30 * (rf / max(cfg.RF_STAR, 1e-12)))
    cop_amb = 1.0 - 0.010 * max(18.0 - T_a, 0.0)
    return min(cfg.COP_HEAT_NOM, max(0.8, cop_foul * cop_amb))


def cooling_heating_loads(bldg: BuildingSpec, cfg: HVACConfig, derived: Dict[str, float], T_mean: float, RH_mean: float, GHI_mean: float, T_sp: float, occ: float, doy: int) -> Dict[str, float]:
    # Optional native zone-by-zone load calculation. The run_scenario_model function attaches _ZONE_TABLE when available.
    zone_table = getattr(cfg, "_ZONE_TABLE", None)
    if getattr(cfg, "APPLY_NATIVE_ZONE_LOADS", False) and zone_table is not None and len(zone_table) > 0 and not getattr(cfg, "_IN_ZONE_RECURSION", False):
        total = {"Q_cool_kw": 0.0, "Q_heat_kw": 0.0, "people": 0.0, "internal_kw": 0.0, "lighting_kw": 0.0, "equipment_kw": 0.0, "latent_cooling_kw": 0.0, "capacity_unmet_kw": 0.0}
        old = getattr(cfg, "_IN_ZONE_RECURSION", False)
        cfg._IN_ZONE_RECURSION = True
        try:
            week = ((int(doy) - 1) // 7) % 52
            in_sem = (1 <= week <= 16) or (20 <= week <= 34)
            is_summer = 180 <= int(doy) <= 242
            for _, z in pd.DataFrame(zone_table).iterrows():
                area = float(z.get("area_m2", 0.0))
                if area <= 0:
                    continue
                z_occ_density = float(z.get("occ_density", bldg.occupancy_density_p_m2))
                if in_sem:
                    z_occ = float(z.get("term_factor", occ))
                elif is_summer:
                    z_occ = float(z.get("summer_factor", occ))
                else:
                    z_occ = float(z.get("break_factor", occ))
                z_occ = float(np.clip(z_occ, 0.0, 1.0))
                zb = BuildingSpec(**asdict(bldg))
                zb.conditioned_area_m2 = area
                zb.n_spaces = 1
                zb.occupancy_density_p_m2 = z_occ_density
                zd = derive_building_numbers(zb)
                zr = cooling_heating_loads(zb, cfg, zd, T_mean, RH_mean, GHI_mean, T_sp, z_occ, doy)
                for k in total:
                    total[k] += float(zr.get(k, 0.0))
        finally:
            cfg._IN_ZONE_RECURSION = old
        q_cool = max(total["Q_cool_kw"], 0.0)
        q_heat = max(total["Q_heat_kw"], 0.0)
        mode = "cooling" if q_cool >= q_heat else "heating"
        return {"Q_cool_kw": q_cool, "Q_heat_kw": q_heat, "Q_HVAC_kw": q_cool if mode == "cooling" else q_heat, "mode": mode,
                "people": total["people"], "internal_kw": total["internal_kw"], "lighting_kw": total["lighting_kw"], "equipment_kw": total["equipment_kw"],
                "latent_cooling_kw": total["latent_cooling_kw"], "zone_load_mode": "native_zone_sum", "hx_capacity_factor": 1.0, "capacity_unmet_kw": total.get("capacity_unmet_kw", 0.0)}

    q_cool_des = derived["Q_cool_des_kw"]
    q_heat_des = derived["Q_heat_des_kw"]
    people_enabled = bool(getattr(cfg, "USE_INTERNAL_GAINS", True) and getattr(cfg, "USE_PEOPLE_GAINS", True))
    lighting_enabled = bool(getattr(cfg, "USE_INTERNAL_GAINS", True) and getattr(cfg, "USE_LIGHTING_GAINS", True))
    equipment_enabled = bool(getattr(cfg, "USE_INTERNAL_GAINS", True) and getattr(cfg, "USE_EQUIPMENT_GAINS", True))
    solar_enabled = bool(getattr(cfg, "USE_SOLAR", True))
    infiltration_enabled = bool(getattr(cfg, "USE_INFILTRATION", True))

    n_people = derived["N_people_max"] * occ if people_enabled else 0.0
    lighting_kw = bldg.conditioned_area_m2 * bldg.lighting_w_m2 / 1000.0 if lighting_enabled else 0.0
    equipment_kw = bldg.conditioned_area_m2 * bldg.equipment_w_m2 / 1000.0 if equipment_enabled else 0.0
    internal_kw = (lighting_kw + equipment_kw) * max(0.20, occ * cfg.INTERNAL_USE_FACTOR + 0.20)
    dT_cool = max(T_mean - T_sp, 0.0)
    dT_heat = max(T_sp - T_mean, 0.0)
    ghi_norm = min(max(GHI_mean / 700.0, 0.0), 1.5)
    humidity_mult = 1.0 + cfg.HUMIDITY_COOL_FACTOR * max(RH_mean - 60.0, 0.0)
    envelope_mult = 1.0 if getattr(cfg, "USE_ENVELOPE", True) else 0.0
    q_cool_env = envelope_mult * cfg.A_COOL_ENV * q_cool_des * (dT_cool / max(cfg.DT_REF_COOL, 1e-9))
    solar_season_factor = 1.0 if getattr(cfg, "TIME_STEP_HOURS", 24.0) < 24.0 else max(math.sin(math.pi * doy / 365.0), 0.0)
    q_cool_solar = (cfg.SOLAR_COOL_FACTOR * q_cool_des * ghi_norm * solar_season_factor) if solar_enabled else 0.0
    q_cool_occ = n_people * bldg.sensible_w_per_person / 1000.0 if people_enabled else 0.0
    q_cool_inf = (cfg.INFIL_COOL_FACTOR * q_cool_des * (dT_cool / max(cfg.DT_REF_COOL, 1e-9))) if infiltration_enabled else 0.0
    latent_kw = estimate_latent_cooling_kw(bldg, cfg, derived, T_mean, RH_mean, occ, af=float(getattr(cfg, "_CURRENT_AF", 1.0))) if getattr(cfg, "APPLY_LATENT_LOAD_TO_CORE", False) else 0.0
    q_cool_raw = (q_cool_env + q_cool_solar + internal_kw + q_cool_occ + q_cool_inf) * humidity_mult + latent_kw
    q_heat_env = envelope_mult * cfg.A_HEAT_ENV * q_heat_des * (dT_heat / max(cfg.DT_REF_HEAT, 1e-9))
    q_heat_inf = (cfg.INFIL_HEAT_FACTOR * q_heat_des * (dT_heat / max(cfg.DT_REF_HEAT, 1e-9))) if infiltration_enabled else 0.0
    q_internal_credit = cfg.HEAT_INTERNAL_CREDIT * (internal_kw + q_cool_occ)
    q_heat_raw = q_heat_env + q_heat_inf - q_internal_credit

    deg_for_capacity = max(float(getattr(cfg, "_CURRENT_DELTA", 0.0)), 0.0)
    hx_capacity_factor = 1.0
    if getattr(cfg, "APPLY_HX_UA_TO_CAPACITY", False):
        hx_capacity_factor = max(0.40, 1.0 - float(getattr(cfg, "HX_UA_LOSS_FACTOR", 0.30)) * deg_for_capacity)
    cool_cap = 1.20 * q_cool_des * hx_capacity_factor
    heat_cap = 1.20 * q_heat_des * hx_capacity_factor
    q_cool = max(0.0, min(q_cool_raw, cool_cap))
    q_heat = max(0.0, min(q_heat_raw, heat_cap))
    capacity_unmet = max(q_cool_raw - cool_cap, 0.0) + max(q_heat_raw - heat_cap, 0.0)
    if not getattr(cfg, "USE_COOLING", True):
        q_cool = 0.0
    if not getattr(cfg, "USE_HEATING", True):
        q_heat = 0.0
    mode = "cooling" if q_cool >= q_heat else "heating"
    q_hvac = q_cool if mode == "cooling" else q_heat
    return {"Q_cool_kw": q_cool, "Q_heat_kw": q_heat, "Q_HVAC_kw": q_hvac, "mode": mode, "people": n_people, "internal_kw": internal_kw,
            "lighting_kw": lighting_kw, "equipment_kw": equipment_kw, "latent_cooling_kw": latent_kw, "zone_load_mode": "building_aggregate",
            "hx_capacity_factor": hx_capacity_factor, "capacity_unmet_kw": capacity_unmet}

def evaluate_controls(bldg: BuildingSpec, cfg: HVACConfig, derived: Dict[str, float], T_mean: float, RH_mean: float, GHI_mean: float, occ: float, year_frac: float, doy: int, rf: float, dust: float, T_sp: float, af: float, duration_hours: float = 24.0) -> Dict[str, float]:
    time_scale_days = duration_hours / 24.0
    if getattr(cfg, "USE_DEGRADATION", True):
        rf_next = cfg.RF_STAR - (cfg.RF_STAR - rf) * math.exp(-cfg.B_FOUL * time_scale_days)
        dust_next = dust + cfg.DUST_RATE * af * time_scale_days
        dp_next, deg_next = degradation_index(cfg, rf_next, dust_next)
    else:
        rf_next, dust_next, dp_next, deg_next = 0.0, 0.0, cfg.DP_CLEAN, 0.0
    cfg._CURRENT_DELTA = deg_next
    cfg._CURRENT_AF = af
    loads = cooling_heating_loads(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, T_sp, occ, doy)
    mode = loads["mode"]
    current_cop = cop_cooling(cfg, T_mean, year_frac, rf_next) if mode == "cooling" else cop_heating(cfg, T_mean, year_frac, rf_next)
    P_hvac = loads["Q_HVAC_kw"] / max(current_cop, 0.8)
    P_fan = 0.0
    if getattr(cfg, "USE_HVAC_FANS", True):
        P_fan = (derived["Q_air_nom_m3h"] * af / 3600.0 * dp_next / max(cfg.FAN_EFF, 1e-6)) / 1000.0
    power_terms = auxiliary_power_terms(bldg, cfg, occ, deg_next, af=af, q_hvac_kw=loads["Q_HVAC_kw"], mode=mode)
    P_pump = power_terms["P_pump"]
    P_aux = power_terms["P_aux"]
    P_tot = P_hvac + P_fan + P_pump + P_aux
    E_period = P_tot * duration_hours
    co2 = E_period * cfg.CO2_FACTOR if getattr(cfg, "USE_CARBON", True) else 0.0
    # Capacity unmet penalty is reflected in comfort when the coupled HX capacity switch is enabled.
    unmet = float(loads.get("capacity_unmet_kw", 0.0))
    cap_penalty = 0.015 * unmet / max(derived.get("Q_cool_des_kw", 1.0), 1e-9) * 100.0
    T_zone = T_sp + 2.2 * (1.0 - af) * occ + 0.08 * max(T_mean - T_sp, 0.0) - 0.06 * max(T_sp - T_mean, 0.0) + cfg.HUMIDITY_COMFORT_FACTOR * max(RH_mean - 60.0, 0.0) + 0.60 * deg_next * occ + cap_penalty
    comfort_dev = abs(T_zone - cfg.T_SET)
    e_n = E_period / max((derived["Q_cool_des_kw"] * duration_hours * 1.5), 1e-9)
    d_n = deg_next
    c_n = comfort_dev / 3.0
    co2_n = co2 / max((derived["Q_cool_des_kw"] * cfg.CO2_FACTOR * duration_hours * 1.5), 1e-9)
    J = cfg.W_ENERGY * e_n + cfg.W_DEGRAD * d_n + cfg.W_COMFORT * c_n + cfg.W_CARBON * co2_n
    res = {"rf_next": rf_next, "dust_next": dust_next, "dp_next": dp_next, "deg_next": deg_next, "cop": current_cop, "Q_cool_kw": loads["Q_cool_kw"], "Q_heat_kw": loads["Q_heat_kw"], "Q_HVAC_kw": loads["Q_HVAC_kw"], "mode": mode, "P_tot": P_tot, "P_fan": P_fan, "P_pump": P_pump, "P_aux": P_aux, "P_hvac": P_hvac, "E_hvac": P_hvac * duration_hours, "E_fan": P_fan * duration_hours, "E_pump": P_pump * duration_hours, "E_aux": P_aux * duration_hours, "E_day": E_period, "E_period": E_period, "co2": co2, "comfort_dev": comfort_dev, "objective": J, "people": loads.get("people", 0.0), "internal_kw": loads.get("internal_kw", 0.0)}
    return apply_core_coupled_corrections(bldg, cfg, derived, res, loads, T_mean, RH_mean, occ, T_sp, af, duration_hours)

def optimize_s3(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, prev_T_sp, prev_af, rng, duration_hours: float = 24.0):
    center = np.array([prev_T_sp, prev_af], dtype=float)
    sigma = np.array([1.4, 0.18], dtype=float)
    best_x = center.copy()
    best_obj = evaluate_controls(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, best_x[0], best_x[1], duration_hours=duration_hours)["objective"]
    for _ in range(cfg.APO_ITERS):
        pop = []
        candidates = [best_x, np.array([cfg.T_SET, 1.0]), np.array([prev_T_sp, prev_af])]
        while len(candidates) < cfg.APO_POP:
            x = center + rng.normal(0.0, 1.0, size=2) * sigma
            x[0] = float(np.clip(x[0], cfg.T_SP_MIN, cfg.T_SP_MAX))
            x[1] = float(np.clip(x[1], cfg.AF_MIN, cfg.AF_MAX))
            candidates.append(x)
        for x in candidates:
            obj = evaluate_controls(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, float(x[0]), float(x[1]), duration_hours=duration_hours)["objective"]
            pop.append((obj, x))
        pop.sort(key=lambda t: t[0])
        elite = pop[: max(3, cfg.APO_POP // 4)]
        elite_x = np.array([e[1] for e in elite])
        center = elite_x.mean(axis=0)
        center[0] = float(np.clip(center[0], cfg.T_SP_MIN, cfg.T_SP_MAX))
        center[1] = float(np.clip(center[1], cfg.AF_MIN, cfg.AF_MAX))
        if elite[0][0] < best_obj:
            best_x = elite[0][1].copy()
            best_obj = elite[0][0]
        sigma *= 0.72
    return float(best_x[0]), float(best_x[1])



def simulate_combo(
    strategy: str,
    severity: str,
    climate_name: str,
    bldg: BuildingSpec,
    base_cfg: HVACConfig,
    base_weather: pd.DataFrame,
    schedule_profile: Optional[Dict[str, float]] = None,
    random_state: int = 42,
    degradation_model: str = "physics",
    operation_schedule_df: Optional[pd.DataFrame] = None,
):
    cfg = apply_hvac_preset(apply_severity(base_cfg, severity))
    derived = derive_building_numbers(bldg)
    duration_hours = resolve_time_step_hours(getattr(cfg, "TIME_STEP_HOURS", 24.0))
    time_scale_days = duration_hours / 24.0
    steps_per_year = weather_steps_per_year(base_weather, duration_hours)
    steps_per_day = max(1, int(round(24.0 / duration_hours)))
    rng = np.random.default_rng(random_state + sum(ord(c) for c in strategy + severity + climate_name))
    rf = 0.0
    dust = 0.0
    delta_state = 0.0
    T_sp = cfg.T_SET
    af = 1.0
    daily_rows = []
    hx_count = 0
    filter_count = 0
    last_hx_day = None
    last_filter_day = None

    if not getattr(cfg, "USE_DEGRADATION", True):
        degradation_model = "none"

    for step in range(cfg.years * steps_per_year):
        tf = step_time_fields_from_weather(step, duration_hours, base_weather)
        day = int(tf["day"])
        day_index = day - 1
        year = int(tf["year"])
        doy = int(tf["day_of_year"])
        year_frac = float(tf["elapsed_days"]) / 365.0
        T_mean, T_max, RH_mean, GHI_mean, occ = climate_and_operation_for_step(step, duration_hours, base_weather, climate_name, schedule_profile)

        if strategy == "S3":
            T_sp, af = optimize_s3(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, T_sp, af, rng, duration_hours=duration_hours)
        else:
            T_sp = cfg.T_SET
            af = 1.0

        T_sp, af, ems_flags = apply_ems_control(
            cfg=cfg,
            T_mean=T_mean,
            occ=occ,
            hour_of_day=float(tf.get("hour_of_day", 0.0)),
            day_of_week=int(tf.get("day_of_week", 0)),
            T_sp=T_sp,
            af=af,
            operation_schedule_df=operation_schedule_df,
        )

        if degradation_model == "physics":
            res = evaluate_controls(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, T_sp, af, duration_hours=duration_hours)
            rf = res["rf_next"]
            dust = res["dust_next"]
            dp = res["dp_next"]
            deg = res["deg_next"]
            delta_state = deg
        elif degradation_model in ["linear_ts", "exponential_ts"]:
            rf, dust, dp, deg = ts_degradation_update(
                cfg=cfg,
                severity=severity,
                prev_delta=delta_state,
                T_mean=T_mean,
                RH_mean=RH_mean,
                GHI_mean=GHI_mean,
                model_name=degradation_model,
                time_scale_days=time_scale_days,
            )
            delta_state = deg
            cfg._CURRENT_DELTA = delta_state
            cfg._CURRENT_AF = af
            cfg._CURRENT_DELTA = delta_state
            cfg._CURRENT_AF = af
            loads = cooling_heating_loads(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, T_sp, occ, doy)
            mode = loads["mode"]
            current_cop = cop_cooling(cfg, T_mean, year_frac, rf) if mode == "cooling" else cop_heating(cfg, T_mean, year_frac, rf)
            P_hvac = loads["Q_HVAC_kw"] / max(current_cop, 0.8)
            P_fan = 0.0
            if getattr(cfg, "USE_HVAC_FANS", True):
                P_fan = (derived["Q_air_nom_m3h"] * af / 3600.0 * dp / max(cfg.FAN_EFF, 1e-6)) / 1000.0
            power_terms = auxiliary_power_terms(bldg, cfg, occ, delta_state)
            P_pump = power_terms["P_pump"]
            P_aux = power_terms["P_aux"]
            P_tot = P_hvac + P_fan + P_pump + P_aux
            E_period = P_tot * duration_hours
            co2 = E_period * cfg.CO2_FACTOR if getattr(cfg, "USE_CARBON", True) else 0.0
            T_zone = T_sp + 2.2 * (1.0 - af) * occ + 0.08 * max(T_mean - T_sp, 0.0) - 0.06 * max(T_sp - T_mean, 0.0) + cfg.HUMIDITY_COMFORT_FACTOR * max(RH_mean - 60.0, 0.0) + 0.60 * deg * occ
            comfort_dev = abs(T_zone - cfg.T_SET)
            e_n = E_period / max((derived["Q_cool_des_kw"] * duration_hours * 1.5), 1e-9)
            d_n = deg
            c_n = comfort_dev / 3.0
            co2_n = co2 / max((derived["Q_cool_des_kw"] * cfg.CO2_FACTOR * duration_hours * 1.5), 1e-9)
            J = cfg.W_ENERGY * e_n + cfg.W_DEGRAD * d_n + cfg.W_COMFORT * c_n + cfg.W_CARBON * co2_n
            res = {"rf_next": rf, "dust_next": dust, "dp_next": dp, "deg_next": deg, "cop": current_cop, "Q_cool_kw": loads["Q_cool_kw"], "Q_heat_kw": loads["Q_heat_kw"], "Q_HVAC_kw": loads["Q_HVAC_kw"], "mode": mode, "P_tot": P_tot, "P_fan": P_fan, "P_pump": P_pump, "P_aux": P_aux, "P_hvac": P_hvac, "E_hvac": P_hvac * duration_hours, "E_fan": P_fan * duration_hours, "E_pump": P_pump * duration_hours, "E_aux": P_aux * duration_hours, "E_day": E_period, "E_period": E_period, "co2": co2, "comfort_dev": comfort_dev, "objective": J, "people": loads.get("people", 0.0), "internal_kw": loads.get("internal_kw", 0.0)}
        elif degradation_model == "none":
            rf, dust, dp, delta_state = 0.0, 0.0, cfg.DP_CLEAN, 0.0
            res = evaluate_controls(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, T_sp, af, duration_hours=duration_hours)
        else:
            raise ValueError(f"Unsupported degradation_model: {degradation_model}")

        do_hx = do_filter = False
        maint_cost = 0.0
        if getattr(cfg, "USE_DEGRADATION", True):
            if strategy == "S0":
                # fixed calendar maintenance: trigger once when the selected time step first reaches the calendar day
                do_hx = (doy - 1 == 180) and (last_hx_day != day)
                do_filter = (doy - 1 in (0, 90, 180, 270)) and (last_filter_day != day)
            elif strategy == "S1":
                do_hx = rf >= cfg.RF_THRESH
                do_filter = dp >= cfg.DP_THRESH
            elif strategy == "S2":
                do_hx = (day_index % max(int(cfg.HX_INTERVAL), 1) == 0) and (last_hx_day != day)
                do_filter = (day_index % max(int(cfg.FILTER_INTERVAL), 1) == 0) and (last_filter_day != day)
            elif strategy == "S3":
                do_hx = (rf >= cfg.RF_WARN) or (delta_state >= cfg.DEG_TRIGGER)
                do_filter = (dp >= cfg.DP_WARN) or (delta_state >= cfg.DEG_TRIGGER)

        if do_hx:
            rf = 0.0
            hx_count += 1
            last_hx_day = day
            if getattr(cfg, "USE_MAINTENANCE_COST", True):
                maint_cost += cfg.COST_HX
        if do_filter:
            dust = 0.0
            filter_count += 1
            last_filter_day = day
            if getattr(cfg, "USE_MAINTENANCE_COST", True):
                maint_cost += cfg.COST_FILTER

        if degradation_model in ["linear_ts", "exponential_ts"]:
            if do_hx and do_filter:
                delta_state *= 0.40
            elif do_hx or do_filter:
                delta_state *= 0.65
            rf = min(cfg.RF_STAR, cfg.RF_STAR * min(delta_state * 1.20, 1.0))
            dp = min(cfg.DP_CLEAN + delta_state * (cfg.DP_MAX - cfg.DP_CLEAN), cfg.DP_MAX)
            dust = max((dp - cfg.DP_CLEAN) / max(cfg.K_CLOG, 1e-9), 0.0)

        # Re-evaluate after maintenance reset so the period energy reflects post-action operation, as in the original model pattern.
        if degradation_model in ["physics", "none"]:
            res = evaluate_controls(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, T_sp, af, duration_hours=duration_hours)
        else:
            cfg._CURRENT_DELTA = delta_state
            cfg._CURRENT_AF = af
            cfg._CURRENT_DELTA = delta_state
            cfg._CURRENT_AF = af
            loads = cooling_heating_loads(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, T_sp, occ, doy)
            mode = loads["mode"]
            current_cop = cop_cooling(cfg, T_mean, year_frac, rf) if mode == "cooling" else cop_heating(cfg, T_mean, year_frac, rf)
            P_hvac = loads["Q_HVAC_kw"] / max(current_cop, 0.8)
            P_fan = 0.0
            if getattr(cfg, "USE_HVAC_FANS", True):
                P_fan = (derived["Q_air_nom_m3h"] * af / 3600.0 * dp / max(cfg.FAN_EFF, 1e-6)) / 1000.0
            power_terms = auxiliary_power_terms(bldg, cfg, occ, delta_state)
            P_pump = power_terms["P_pump"]
            P_aux = power_terms["P_aux"]
            P_tot = P_hvac + P_fan + P_pump + P_aux
            E_period = P_tot * duration_hours
            co2 = E_period * cfg.CO2_FACTOR if getattr(cfg, "USE_CARBON", True) else 0.0
            T_zone = T_sp + 2.2 * (1.0 - af) * occ + 0.08 * max(T_mean - T_sp, 0.0) - 0.06 * max(T_sp - T_mean, 0.0) + cfg.HUMIDITY_COMFORT_FACTOR * max(RH_mean - 60.0, 0.0) + 0.60 * delta_state * occ
            comfort_dev = abs(T_zone - cfg.T_SET)
            e_n = E_period / max((derived["Q_cool_des_kw"] * duration_hours * 1.5), 1e-9)
            d_n = delta_state
            c_n = comfort_dev / 3.0
            co2_n = co2 / max((derived["Q_cool_des_kw"] * cfg.CO2_FACTOR * duration_hours * 1.5), 1e-9)
            J = cfg.W_ENERGY * e_n + cfg.W_DEGRAD * d_n + cfg.W_COMFORT * c_n + cfg.W_CARBON * co2_n
            res = {"rf_next": rf, "dust_next": dust, "dp_next": dp, "deg_next": delta_state, "cop": current_cop, "Q_cool_kw": loads["Q_cool_kw"], "Q_heat_kw": loads["Q_heat_kw"], "Q_HVAC_kw": loads["Q_HVAC_kw"], "mode": mode, "P_tot": P_tot, "P_fan": P_fan, "P_pump": P_pump, "P_aux": P_aux, "P_hvac": P_hvac, "E_hvac": P_hvac * duration_hours, "E_fan": P_fan * duration_hours, "E_pump": P_pump * duration_hours, "E_aux": P_aux * duration_hours, "E_day": E_period, "E_period": E_period, "co2": co2, "comfort_dev": comfort_dev, "objective": J, "people": loads.get("people", 0.0), "internal_kw": loads.get("internal_kw", 0.0)}

        # Apply optional coupled publication modules to manual time-series degradation branches as well.
        if degradation_model not in ["physics", "none"]:
            res = apply_core_coupled_corrections(bldg, cfg, derived, res, loads, T_mean, RH_mean, occ, T_sp, af, duration_hours)

        if int(ems_flags.get("ems_economizer", 0)) == 1 and res.get("mode") == "cooling":
            reduction = float(np.clip(getattr(cfg, "EMS_ECONOMIZER_COOLING_REDUCTION", 0.20), 0.0, 0.80))
            old_e_hvac = float(res.get("E_hvac", 0.0))
            new_e_hvac = old_e_hvac * (1.0 - reduction)
            delta_e = old_e_hvac - new_e_hvac
            res["E_hvac"] = new_e_hvac
            res["P_hvac"] = new_e_hvac / max(duration_hours, 1e-9)
            res["E_period"] = max(0.0, float(res.get("E_period", 0.0)) - delta_e)
            res["E_day"] = res["E_period"]
            res["P_tot"] = res["E_period"] / max(duration_hours, 1e-9)
            res["co2"] = res["E_period"] * cfg.CO2_FACTOR if getattr(cfg, "USE_CARBON", True) else 0.0

        discomfort_flag = int((occ > 0.5) and (res["comfort_dev"] > 0.3))
        energy_cost = res["E_period"] * cfg.E_PRICE
        cost_period = energy_cost + maint_cost
        daily_rows.append({
            "strategy": strategy, "severity": severity, "climate": climate_name,
            "scenario_combo_3axis": f"{strategy}_{severity}_{climate_name}",
            "building_type": bldg.building_type, "area_m2": bldg.conditioned_area_m2,
            "floors": bldg.floors, "n_spaces": bldg.n_spaces, "hvac_system_type": cfg.hvac_system_type,
            "Q_cool_des_kw": derived["Q_cool_des_kw"], "Q_heat_des_kw": derived["Q_heat_des_kw"], "Q_air_nom_m3h": derived["Q_air_nom_m3h"],
            **tf,
            "T_amb_C": T_mean, "T_max_C": T_max, "RH_mean_pct": RH_mean, "GHI_mean_Wm2": GHI_mean,
            "occ": occ, "T_sp_C": T_sp, "alpha_flow": af,
            "ems_active": ems_flags.get("ems_active", 0), "ems_mode_applied": ems_flags.get("ems_mode_applied", "None"),
            "ems_occ_control": ems_flags.get("ems_occ_control", 0), "ems_night_setback": ems_flags.get("ems_night_setback", 0),
            "ems_demand_response": ems_flags.get("ems_demand_response", 0), "ems_economizer": ems_flags.get("ems_economizer", 0),
            "ems_custom_schedule": ems_flags.get("ems_custom_schedule", 0), "ems_optimum_start": ems_flags.get("ems_optimum_start", 0),
            "R_f": rf, "dust_kg": dust,
            "dP_Pa": res["dp_next"], "dP_fan_Pa": res.get("dP_fan_Pa", res.get("dp_next", np.nan)), "dP_water_kPa": res.get("dP_water_kPa", 0.0), "water_flow_m3h": res.get("water_flow_m3h", 0.0),
            "delta": res["deg_next"], "COP_eff": res["cop"], "COP_base_before_PLR": res.get("COP_base_before_PLR", res.get("cop", np.nan)), "PLR": res.get("PLR", np.nan), "PLR_modifier": res.get("PLR_modifier", 1.0), "mode": res["mode"],
            "Q_cool_kw": res["Q_cool_kw"], "Q_heat_kw": res["Q_heat_kw"], "Q_HVAC_kw": res["Q_HVAC_kw"], "latent_cooling_kw": res.get("latent_cooling_kw", 0.0), "capacity_unmet_kw": res.get("capacity_unmet_kw", 0.0), "hx_capacity_factor": res.get("hx_capacity_factor", 1.0), "zone_load_mode": res.get("zone_load_mode", "building_aggregate"),
            "P_hvac_kw": res.get("P_hvac", np.nan), "P_fan_kw": res.get("P_fan", np.nan),
            "P_pump_kw": res.get("P_pump", 0.0), "P_auxiliary_kw": res.get("P_aux", 0.0), "P_total_kw": res.get("P_tot", np.nan),
            "coupled_modules_active": res.get("coupled_modules_active", "none"),
            "thermal_hvac_kwh_period": res.get("E_hvac", res.get("P_hvac", 0.0) * duration_hours),
            "fan_kwh_period": res.get("E_fan", res.get("P_fan", 0.0) * duration_hours),
            "pump_kwh_period": res.get("E_pump", res.get("P_pump", 0.0) * duration_hours),
            "auxiliary_kwh_period": res.get("E_aux", res.get("P_aux", 0.0) * duration_hours),
            "people_count": res.get("people", derived["N_people_max"] * occ),
            "internal_gains_kw": res.get("internal_kw", derived["Internal_kw_max"] * max(0.20, occ * cfg.INTERNAL_USE_FACTOR + 0.20)),
            "sensible_people_kw": derived["N_people_max"] * occ * bldg.sensible_w_per_person / 1000.0,
            "energy_kwh_period": res["E_period"], "energy_kwh_day": res["E_period"],
            "co2_kg_period": res["co2"], "co2_kg_day": res["co2"],
            "comfort_dev_C": res["comfort_dev"],
            "occupied_discomfort_flag": discomfort_flag,
            "occupied_discomfort_day_equiv": discomfort_flag * time_scale_days,
            "cost_usd_period": cost_period, "cost_usd_day": cost_period,
            "maintenance_cost_usd": maint_cost,
            "hx_cleaned": int(do_hx), "filter_replaced": int(do_filter),
        })
    daily = pd.DataFrame(daily_rows)
    annual = daily.groupby(["strategy", "severity", "climate", "year"], as_index=False).agg(
        annual_energy_MWh=("energy_kwh_period", lambda ss: float(ss.sum() / 1000.0)),
        annual_thermal_hvac_MWh=("thermal_hvac_kwh_period", lambda ss: float(ss.sum() / 1000.0)),
        annual_fan_MWh=("fan_kwh_period", lambda ss: float(ss.sum() / 1000.0)),
        annual_pump_MWh=("pump_kwh_period", lambda ss: float(ss.sum() / 1000.0)),
        annual_auxiliary_MWh=("auxiliary_kwh_period", lambda ss: float(ss.sum() / 1000.0)),
        annual_cost_usd=("cost_usd_period", "sum"),
        annual_co2_tonne=("co2_kg_period", lambda ss: float(ss.sum() / 1000.0)),
        mean_COP=("COP_eff", "mean"),
        mean_delta=("delta", "mean"),
        mean_comfort_dev=("comfort_dev_C", "mean"),
        mean_Q_cool_kw=("Q_cool_kw", "mean"),
        mean_Q_heat_kw=("Q_heat_kw", "mean"),
        occupied_discomfort_days=("occupied_discomfort_day_equiv", "sum"),
        filter_replacements=("filter_replaced", "sum"),
        hx_cleanings=("hx_cleaned", "sum"),
    )
    summary = {
        "strategy": strategy, "severity": severity, "climate": climate_name,
        "scenario_combo_3axis": f"{strategy}_{severity}_{climate_name}",
        "Building Area m2": bldg.conditioned_area_m2, "No. of Spaces": bldg.n_spaces, "HVAC System": cfg.hvac_system_type,
        "Cooling Design kW": derived["Q_cool_des_kw"], "Heating Design kW": derived["Q_heat_des_kw"], "Airflow m3h": derived["Q_air_nom_m3h"],
        "Time Step Hours": duration_hours,
        "Total Energy MWh": float(daily["energy_kwh_period"].sum() / 1000.0),
        "Total Thermal HVAC Energy MWh": float(daily["thermal_hvac_kwh_period"].sum() / 1000.0),
        "Total Fan Energy MWh": float(daily["fan_kwh_period"].sum() / 1000.0),
        "Total Pump Energy MWh": float(daily["pump_kwh_period"].sum() / 1000.0),
        "Total Auxiliary Energy MWh": float(daily["auxiliary_kwh_period"].sum() / 1000.0),
        "Total Cost USD": float(daily["cost_usd_period"].sum()),
        "Total CO2 tonne": float(daily["co2_kg_period"].sum() / 1000.0),
        "Mean COP": float(daily["COP_eff"].mean()),
        "Mean Degradation Index": float(daily["delta"].mean()),
        "Mean Comfort Deviation C": float(daily["comfort_dev_C"].mean()),
        "Mean Cooling Load kW": float(daily["Q_cool_kw"].mean()),
        "Mean Heating Load kW": float(daily["Q_heat_kw"].mean()),
        "Occupied Discomfort Days": float(daily["occupied_discomfort_day_equiv"].sum()),
        "Filter Replacements count": int(filter_count), "HX Cleanings count": int(hx_count),
    }
    return daily, annual, summary

def save_figure(df: pd.DataFrame, x: str, y: str, hue: Optional[str], title: str, out_png: Path, out_svg: Optional[Path] = None):
    plt.figure(figsize=(8, 5))
    if hue and hue in df.columns:
        for val, grp in df.groupby(hue):
            plt.plot(grp[x], grp[y], marker="o", label=str(val))
        plt.legend(frameon=False)
    else:
        plt.plot(df[x], df[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    if out_svg:
        plt.savefig(out_svg)
    plt.close()


def save_heatmap(summary_df: pd.DataFrame, climate_name: str, value_col: str, out_png: Path, out_svg: Optional[Path] = None):
    subset = summary_df[summary_df["climate"] == climate_name].copy()
    pivot = subset.pivot(index="severity", columns="strategy", values=value_col)
    order = [s for s in ["Mild", "Moderate", "Severe", "High"] if s in pivot.index]
    pivot = pivot.reindex(order)
    plt.figure(figsize=(7.5, 5.5))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(f"{value_col} | {climate_name}")
    plt.colorbar(label=value_col)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    if out_svg:
        plt.savefig(out_svg)
    plt.close()


def run_scenario_model(
    output_dir: str | Path,
    axis_mode: str,
    bldg: BuildingSpec,
    cfg: HVACConfig,
    weather_mode: str = "synthetic",
    epw_path: str | None = None,
    csv_path: str | None = None,
    weather_df: Optional[pd.DataFrame] = None,
    fixed_strategy: str = "S3",
    fixed_severity: str = "Moderate",
    fixed_climate: str = "C0_Baseline",
    zone_df: Optional[pd.DataFrame] = None,
    random_state: int = 42,
    include_baseline_layer: bool = True,
    include_baseline_as_scenario: bool = False,
    degradation_model: str = "physics",
    time_step_hours: float | None = None,
    operation_schedule_df: Optional[pd.DataFrame] = None,
    parameter_switches: Optional[Dict[str, bool]] = None,
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures_dir = out / "figures"
    figures_dir.mkdir(exist_ok=True)

    bldg, zone_meta = aggregate_zone_occupancy(bldg, zone_df)
    cfg = _clone_cfg(cfg)
    if time_step_hours is not None:
        cfg.TIME_STEP_HOURS = resolve_time_step_hours(time_step_hours)
    else:
        cfg.TIME_STEP_HOURS = resolve_time_step_hours(getattr(cfg, "TIME_STEP_HOURS", 24.0))
    if parameter_switches:
        key_map = {
            "sw_use_envelope": "USE_ENVELOPE",
            "sw_use_walls": "USE_WALLS",
            "sw_use_roof": "USE_ROOF",
            "sw_use_windows": "USE_WINDOWS",
            "sw_use_solar": "USE_SOLAR",
            "sw_use_infiltration": "USE_INFILTRATION",
            "sw_use_internal_gains": "USE_INTERNAL_GAINS",
            "sw_use_people_gains": "USE_PEOPLE_GAINS",
            "sw_use_lighting_gains": "USE_LIGHTING_GAINS",
            "sw_use_equipment_gains": "USE_EQUIPMENT_GAINS",
            "sw_use_hvac_fans": "USE_HVAC_FANS",
            "sw_use_hvac_pumps": "USE_HVAC_PUMPS",
            "sw_use_hvac_auxiliary": "USE_HVAC_AUXILIARY",
            "APPLY_PART_LOAD_COP_TO_CORE": "APPLY_PART_LOAD_COP_TO_CORE",
            "APPLY_LATENT_LOAD_TO_CORE": "APPLY_LATENT_LOAD_TO_CORE",
            "APPLY_HX_AIR_PRESSURE_TO_FAN": "APPLY_HX_AIR_PRESSURE_TO_FAN",
            "APPLY_HX_WATER_PRESSURE_TO_PUMP": "APPLY_HX_WATER_PRESSURE_TO_PUMP",
            "APPLY_HX_UA_TO_CAPACITY": "APPLY_HX_UA_TO_CAPACITY",
            "APPLY_NATIVE_ZONE_LOADS": "APPLY_NATIVE_ZONE_LOADS",
            "sw_use_cooling": "USE_COOLING",
            "sw_use_heating": "USE_HEATING",
            "sw_use_degradation": "USE_DEGRADATION",
            "sw_use_carbon": "USE_CARBON",
            "sw_use_maintenance_cost": "USE_MAINTENANCE_COST",
        }
        for k, v in parameter_switches.items():
            attr = key_map.get(k, k)
            if hasattr(cfg, attr):
                setattr(cfg, attr, bool(v))
    cfg = apply_hvac_preset(cfg)
    if getattr(cfg, "APPLY_NATIVE_ZONE_LOADS", False) and zone_meta.get("zone_table"):
        cfg._ZONE_TABLE = pd.DataFrame(zone_meta.get("zone_table", []))
    base_weather, weather_meta = _load_base_weather(weather_mode, epw_path, csv_path, weather_df, random_state, cfg.TIME_STEP_HOURS)

    combos = []
    if axis_mode == "baseline_scenario":
        combos = []
        dataset_name = "baseline_scenario_ml_dataset.csv"
        summary_name = "baseline_scenario_summary.csv"
        annual_name = "annual_baseline_scenario.csv"
        include_baseline_layer = True
    elif axis_mode == "one_severity":
        combos = [(fixed_strategy, sev, fixed_climate) for sev in SEVERITY_LEVELS.keys()]
        dataset_name = "one_axis_severity_ml_dataset.csv"
        summary_name = "one_axis_severity_summary.csv"
        annual_name = "annual_one_axis_severity.csv"
    elif axis_mode == "one_strategy":
        combos = [(stg, fixed_severity, fixed_climate) for stg in SCENARIOS.keys()]
        dataset_name = "one_axis_strategy_ml_dataset.csv"
        summary_name = "one_axis_strategy_summary.csv"
        annual_name = "annual_one_axis_strategy.csv"
    elif axis_mode == "two_axis":
        combos = [(stg, sev, fixed_climate) for sev in SEVERITY_LEVELS.keys() for stg in SCENARIOS.keys()]
        dataset_name = "matrix_ml_dataset.csv"
        summary_name = "matrix_summary.csv"
        annual_name = "annual_matrix.csv"
    elif axis_mode == "three_axis":
        combos = [(stg, sev, cli) for cli in CLIMATE_LEVELS.keys() for sev in SEVERITY_LEVELS.keys() for stg in SCENARIOS.keys()]
        dataset_name = "three_axis_ml_dataset.csv"
        summary_name = "three_axis_summary.csv"
        annual_name = "annual_three_axis.csv"
    else:
        raise ValueError(f"Unsupported axis_mode: {axis_mode}")

    all_daily, all_annual, summaries = [], [], []
    schedule_profile = zone_meta.get("schedule_profile", None)
    for strategy, severity, climate_name in combos:
        daily, annual, summary = simulate_combo(
            strategy=strategy,
            severity=severity,
            climate_name=climate_name,
            bldg=bldg,
            base_cfg=cfg,
            base_weather=base_weather,
            schedule_profile=schedule_profile,
            random_state=random_state,
            degradation_model=degradation_model,
            operation_schedule_df=operation_schedule_df,
        )
        all_daily.append(daily)
        all_annual.append(annual)
        summaries.append(summary)

    baseline_daily_df = pd.DataFrame()
    baseline_annual_df = pd.DataFrame()
    baseline_summary_df = pd.DataFrame()

    if include_baseline_layer:
        baseline_daily, baseline_annual, baseline_summary = simulate_baseline_no_degradation(
            strategy=fixed_strategy if axis_mode in ["one_severity", "baseline_scenario"] else "S2",
            climate_name=fixed_climate,
            bldg=bldg,
            base_cfg=cfg,
            base_weather=base_weather,
            schedule_profile=schedule_profile,
            random_state=random_state,
        )
        baseline_daily_df = baseline_daily.copy()
        baseline_annual_df = baseline_annual.copy()
        baseline_summary_df = pd.DataFrame([baseline_summary])
        baseline_daily_df.to_csv(out / "baseline_no_degradation_daily.csv", index=False)
        baseline_annual_df.to_csv(out / "baseline_no_degradation_annual.csv", index=False)
        baseline_summary_df.to_csv(out / "baseline_no_degradation_summary.csv", index=False)

    if axis_mode == "baseline_scenario":
        daily_df = baseline_daily_df.copy()
        annual_df = baseline_annual_df.copy()
        summary_df = baseline_summary_df.copy()
    else:
        daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
        annual_df = pd.concat(all_annual, ignore_index=True) if all_annual else pd.DataFrame()
        summary_df = pd.DataFrame(summaries)
        if include_baseline_as_scenario and include_baseline_layer and not baseline_daily_df.empty:
            daily_df = pd.concat([baseline_daily_df, daily_df], ignore_index=True)
            annual_df = pd.concat([baseline_annual_df, annual_df], ignore_index=True)
            summary_df = pd.concat([baseline_summary_df, summary_df], ignore_index=True)

    daily_df.to_csv(out / dataset_name, index=False)
    daily_df.to_csv(out / "matrix_ml_dataset.csv", index=False)
    annual_df.to_csv(out / annual_name, index=False)
    summary_df.to_csv(out / summary_name, index=False)
    base_weather.to_csv(out / "weather_timeseries.csv", index=False)
    # Backward-compatible alias used by older notebooks/apps. It may contain sub-daily rows in the journal-hourly version.
    base_weather.to_csv(out / "baseline_daily_weather.csv", index=False)

    meta = {
        "building_spec": asdict(bldg),
        "hvac_config": asdict(cfg),
        "weather_summary": weather_meta,
        "zone_occupancy_meta": zone_meta,
        "axis_mode": axis_mode,
        "fixed_strategy": fixed_strategy,
        "fixed_severity": fixed_severity,
        "fixed_climate": fixed_climate,
        "available_hvac_types": list(HVAC_PRESETS.keys()),
        "available_severity_levels": list(SEVERITY_LEVELS.keys()),
        "available_climate_levels": list(CLIMATE_LEVELS.keys()),
        "degradation_model": degradation_model,
        "ems_mode": getattr(cfg, "EMS_MODE", "Disabled"),
        "operation_schedule_rows": int(len(operation_schedule_df)) if operation_schedule_df is not None else 0,
        "include_baseline_layer": include_baseline_layer,
        "include_baseline_as_scenario": include_baseline_as_scenario,
        "time_step_hours": cfg.TIME_STEP_HOURS,
        "parameter_switches": parameter_switches or {},
    }
    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # figures
    if not summary_df.empty:
        if axis_mode in ["one_severity", "one_strategy"]:
            key = "severity" if axis_mode == "one_severity" else "strategy"
            save_figure(summary_df, key, "Total Energy MWh", None, f"Total Energy vs {key.title()}", figures_dir / "energy_vs_axis.png", figures_dir / "energy_vs_axis.svg")
            save_figure(summary_df, key, "Mean Degradation Index", None, f"Degradation vs {key.title()}", figures_dir / "degradation_vs_axis.png", figures_dir / "degradation_vs_axis.svg")
            save_figure(summary_df, key, "Mean Comfort Deviation C", None, f"Comfort vs {key.title()}", figures_dir / "comfort_vs_axis.png", figures_dir / "comfort_vs_axis.svg")
        elif axis_mode == "two_axis":
            save_figure(summary_df, "strategy", "Total Energy MWh", "severity", "Energy by Strategy and Severity", figures_dir / "energy_by_strategy_severity.png", figures_dir / "energy_by_strategy_severity.svg")
            save_figure(summary_df, "strategy", "Mean Degradation Index", "severity", "Degradation by Strategy and Severity", figures_dir / "degradation_by_strategy_severity.png", figures_dir / "degradation_by_strategy_severity.svg")
        elif axis_mode == "three_axis":
            for cli in CLIMATE_LEVELS.keys():
                save_heatmap(summary_df, cli, "Total Energy MWh", figures_dir / f"heatmap_energy_{cli}.png", figures_dir / f"heatmap_energy_{cli}.svg")
                save_heatmap(summary_df, cli, "Mean Degradation Index", figures_dir / f"heatmap_degradation_{cli}.png", figures_dir / f"heatmap_degradation_{cli}.svg")
                save_heatmap(summary_df, cli, "Mean Comfort Deviation C", figures_dir / f"heatmap_comfort_{cli}.png", figures_dir / f"heatmap_comfort_{cli}.svg")

    export_excel_report(out, summary_df, annual_df, daily_df, meta)
    export_pdf_report(out, summary_df, annual_df, meta)

    return {
        "dataset_csv": str(out / dataset_name),
        "matrix_ml_dataset_csv": str(out / "matrix_ml_dataset.csv"),
        "summary_csv": str(out / summary_name),
        "annual_csv": str(out / annual_name),
        "excel_report": str(out / "results_export.xlsx"),
        "pdf_report": str(out / "results_report.pdf"),
        "figures_dir": str(figures_dir),
        "baseline_daily_csv": str(out / "baseline_no_degradation_daily.csv") if include_baseline_layer else "",
        "baseline_summary_csv": str(out / "baseline_no_degradation_summary.csv") if include_baseline_layer else "",
        "time_step_hours": str(cfg.TIME_STEP_HOURS),
    }

def export_excel_report(out: Path, summary_df: pd.DataFrame, annual_df: pd.DataFrame, daily_df: pd.DataFrame, meta: Dict[str, object]):
    with pd.ExcelWriter(out / "results_export.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([meta]).to_excel(writer, sheet_name="run_metadata", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        annual_df.to_excel(writer, sheet_name="annual", index=False)
        daily_df.head(5000).to_excel(writer, sheet_name="dataset_head", index=False)  # keeps workbook manageable


def export_pdf_report(out: Path, summary_df: pd.DataFrame, annual_df: pd.DataFrame, meta: Dict[str, object]):
    pdf_path = out / "results_report.pdf"
    figs = sorted((out / "figures").glob("*.png"))
    with PdfPages(pdf_path) as pdf:
        # cover
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        txt = (
            "HVAC Research Modeling Suite v3\n\n"
            f"Axis mode: {meta.get('axis_mode')}\n"
            f"Building type: {meta.get('building_spec', {}).get('building_type')}\n"
            f"HVAC system: {meta.get('hvac_config', {}).get('hvac_system_type')}\n"
            f"Area (m²): {meta.get('building_spec', {}).get('conditioned_area_m2')}\n"
            f"Spaces: {meta.get('building_spec', {}).get('n_spaces')}\n"
            f"Weather: {meta.get('weather_summary', {}).get('source_mode')}\n"
        )
        plt.text(0.08, 0.92, txt, va="top", fontsize=14)
        pdf.savefig(fig, dpi=300); plt.close(fig)

        # summary page
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.text(0.05, 0.97, "Summary table (top rows)", va="top", fontsize=14)
        show_df = summary_df.head(18).copy()
        tbl = plt.table(cellText=show_df.values, colLabels=show_df.columns, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6)
        tbl.scale(1, 1.2)
        pdf.savefig(fig, dpi=300); plt.close(fig)

        # annual page
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.text(0.05, 0.97, "Annual table (top rows)", va="top", fontsize=14)
        show_df = annual_df.head(18).copy()
        tbl = plt.table(cellText=show_df.values, colLabels=show_df.columns, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6)
        tbl.scale(1, 1.2)
        pdf.savefig(fig, dpi=300); plt.close(fig)

        for img in figs[:8]:
            arr = plt.imread(img)
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.imshow(arr)
            plt.axis("off")
            plt.title(img.name)
            pdf.savefig(fig, dpi=300); plt.close(fig)



# ---------- Early benchmark sensitivity and robustness ----------
SENSITIVITY_PARAMETERS = [
    {"group": "building", "attr": "conditioned_area_m2", "label": "Conditioned area", "min": 100.0},
    {"group": "building", "attr": "occupancy_density_p_m2", "label": "Occupancy density", "min": 0.0001},
    {"group": "building", "attr": "lighting_w_m2", "label": "Lighting power density", "min": 0.0},
    {"group": "building", "attr": "equipment_w_m2", "label": "Equipment power density", "min": 0.0},
    {"group": "building", "attr": "sensible_w_per_person", "label": "Sensible heat/person", "min": 1.0},
    {"group": "building", "attr": "airflow_m3h_m2", "label": "Airflow intensity", "min": 0.01},
    {"group": "building", "attr": "cooling_intensity_w_m2", "label": "Cooling design intensity", "min": 1.0},
    {"group": "building", "attr": "heating_intensity_w_m2", "label": "Heating design intensity", "min": 1.0},
    {"group": "building", "attr": "wall_u", "label": "Wall U-value", "min": 0.01},
    {"group": "building", "attr": "roof_u", "label": "Roof U-value", "min": 0.01},
    {"group": "building", "attr": "window_u", "label": "Window U-value", "min": 0.01},
    {"group": "building", "attr": "shgc", "label": "SHGC", "min": 0.01, "max": 0.95},
    {"group": "building", "attr": "glazing_ratio", "label": "Glazing ratio", "min": 0.01, "max": 0.95},
    {"group": "building", "attr": "infiltration_ach", "label": "Infiltration ACH", "min": 0.0},
    {"group": "config", "attr": "COP_COOL_NOM", "label": "Cooling COP", "min": 0.8},
    {"group": "config", "attr": "COP_HEAT_NOM", "label": "Heating COP", "min": 0.8},
    {"group": "config", "attr": "FAN_EFF", "label": "Fan efficiency", "min": 0.1, "max": 0.95},
    {"group": "config", "attr": "COP_AGING_RATE", "label": "COP aging rate", "min": 0.0},
    {"group": "config", "attr": "RF_STAR", "label": "Fouling asymptote RF*", "min": 1e-8},
    {"group": "config", "attr": "B_FOUL", "label": "Fouling growth constant", "min": 0.0},
    {"group": "config", "attr": "DUST_RATE", "label": "Dust accumulation rate", "min": 0.0},
    {"group": "config", "attr": "K_CLOG", "label": "Clogging coefficient", "min": 0.0},
]

SENSITIVITY_KPIS = ["Total Energy MWh", "Total CO2 tonne", "Mean Degradation Index", "Mean Comfort Deviation C", "Total Cost USD"]


def _set_nested_param(bldg: BuildingSpec, cfg: HVACConfig, spec: Dict[str, object], value: float) -> Tuple[BuildingSpec, HVACConfig]:
    b2, c2 = _clone_bldg(bldg), _clone_cfg(cfg)
    value = float(value)
    if "min" in spec:
        value = max(float(spec["min"]), value)
    if "max" in spec:
        value = min(float(spec["max"]), value)
    if spec["group"] == "building":
        setattr(b2, str(spec["attr"]), value)
    else:
        setattr(c2, str(spec["attr"]), value)
        if str(spec["attr"]) in ["COP_COOL_NOM", "COP_HEAT_NOM", "FAN_EFF"]:
            c2.USE_HVAC_PRESET = False
            c2.hvac_system_type = "Custom"
    return b2, c2


def _single_summary_for_analysis(
    bldg: BuildingSpec,
    cfg: HVACConfig,
    base_weather: pd.DataFrame,
    fixed_strategy: str,
    fixed_severity: str,
    fixed_climate: str,
    zone_df: Optional[pd.DataFrame],
    degradation_model: str,
    random_state: int,
) -> Dict[str, float]:
    b2, zone_meta = aggregate_zone_occupancy(bldg, zone_df)
    schedule_profile = zone_meta.get("schedule_profile", None)
    daily, annual, summary = simulate_combo(
        strategy=fixed_strategy,
        severity=fixed_severity,
        climate_name=fixed_climate,
        bldg=b2,
        base_cfg=cfg,
        base_weather=base_weather,
        schedule_profile=schedule_profile,
        random_state=random_state,
        degradation_model=degradation_model,
    )
    return summary


def run_early_sensitivity_analysis(
    output_dir: str | Path,
    bldg: BuildingSpec,
    cfg: HVACConfig,
    weather_mode: str = "synthetic",
    epw_path: str | None = None,
    csv_path: str | None = None,
    weather_df: Optional[pd.DataFrame] = None,
    fixed_strategy: str = "S2",
    fixed_severity: str = "Moderate",
    fixed_climate: str = "C0_Baseline",
    zone_df: Optional[pd.DataFrame] = None,
    degradation_model: str = "physics",
    perturbation_pct: float = 0.10,
    analysis_years: int = 1,
    random_state: int = 42,
    time_step_hours: float | None = None,
    parameter_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """One-at-a-time early screening analysis.

    Each selected input is perturbed down and up around the baseline. The ranking
    is a dimensionless central elasticity: percentage KPI change divided by
    percentage input change. It is designed as a fast early benchmark, not a
    replacement for full global sensitivity analysis.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    cfg0 = _clone_cfg(cfg)
    cfg0.years = int(max(1, analysis_years))
    if time_step_hours is not None:
        cfg0.TIME_STEP_HOURS = resolve_time_step_hours(time_step_hours)
    cfg0 = apply_hvac_preset(cfg0)
    base_weather, weather_meta = _load_base_weather(weather_mode, epw_path, csv_path, weather_df, random_state, cfg0.TIME_STEP_HOURS)

    specs = SENSITIVITY_PARAMETERS
    if parameter_names:
        wanted = set(parameter_names)
        specs = [sp for sp in specs if str(sp["attr"]) in wanted or str(sp["label"]) in wanted]

    base_summary = _single_summary_for_analysis(bldg, cfg0, base_weather, fixed_strategy, fixed_severity, fixed_climate, zone_df, degradation_model, random_state)
    pd.DataFrame([base_summary]).to_csv(out / "sensitivity_base_summary.csv", index=False)

    rows = []
    detail_rows = []
    pct = float(perturbation_pct)
    for spec in specs:
        base_val = float(getattr(bldg if spec["group"] == "building" else cfg0, str(spec["attr"])))
        if not np.isfinite(base_val):
            continue
        delta = abs(base_val) * pct if abs(base_val) > 1e-12 else pct
        low_val = base_val - delta
        high_val = base_val + delta
        b_low, c_low = _set_nested_param(bldg, cfg0, spec, low_val)
        b_high, c_high = _set_nested_param(bldg, cfg0, spec, high_val)
        low_summary = _single_summary_for_analysis(b_low, c_low, base_weather, fixed_strategy, fixed_severity, fixed_climate, zone_df, degradation_model, random_state)
        high_summary = _single_summary_for_analysis(b_high, c_high, base_weather, fixed_strategy, fixed_severity, fixed_climate, zone_df, degradation_model, random_state)
        for label, summary, value in [("low", low_summary, low_val), ("high", high_summary, high_val)]:
            drow = {"parameter": spec["attr"], "label": spec["label"], "case": label, "value": value}
            for k in SENSITIVITY_KPIS:
                drow[k] = summary.get(k, np.nan)
            detail_rows.append(drow)
        row = {"parameter": spec["attr"], "label": spec["label"], "group": spec["group"], "base_value": base_val, "low_value": low_val, "high_value": high_val}
        kpi_indices = []
        for kpi in SENSITIVITY_KPIS:
            base_k = float(base_summary.get(kpi, np.nan))
            low_k = float(low_summary.get(kpi, np.nan))
            high_k = float(high_summary.get(kpi, np.nan))
            denom_k = max(abs(base_k), 1e-9)
            elasticity = ((high_k - low_k) / denom_k) / (2.0 * pct)
            row[f"elasticity_{kpi}"] = elasticity
            row[f"abs_elasticity_{kpi}"] = abs(elasticity)
            kpi_indices.append(abs(elasticity))
        row["composite_importance"] = float(np.nanmean(kpi_indices)) if kpi_indices else np.nan
        rows.append(row)

    ranking = pd.DataFrame(rows).sort_values("composite_importance", ascending=False)
    details = pd.DataFrame(detail_rows)
    ranking.to_csv(out / "early_sensitivity_ranking.csv", index=False)
    details.to_csv(out / "early_sensitivity_details.csv", index=False)

    if not ranking.empty:
        top = ranking.head(15).iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(top["label"], top["composite_importance"])
        plt.xlabel("Composite absolute elasticity")
        plt.title("Early benchmark sensitivity ranking")
        plt.tight_layout()
        plt.savefig(figs / "early_sensitivity_ranking.png", dpi=600)
        plt.close()

    meta = {"method": "one-at-a-time central elasticity", "perturbation_pct": pct, "analysis_years": cfg0.years, "fixed_strategy": fixed_strategy, "fixed_severity": fixed_severity, "fixed_climate": fixed_climate, "weather_summary": weather_meta, "time_step_hours": cfg0.TIME_STEP_HOURS}
    with open(out / "early_sensitivity_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return {
        "ranking_csv": str(out / "early_sensitivity_ranking.csv"),
        "details_csv": str(out / "early_sensitivity_details.csv"),
        "base_summary_csv": str(out / "sensitivity_base_summary.csv"),
        "figures_dir": str(figs),
    }


def run_robustness_analysis(
    output_dir: str | Path,
    bldg: BuildingSpec,
    cfg: HVACConfig,
    weather_mode: str = "synthetic",
    epw_path: str | None = None,
    csv_path: str | None = None,
    weather_df: Optional[pd.DataFrame] = None,
    fixed_strategy: str = "S2",
    fixed_severity: str = "Moderate",
    fixed_climate: str = "C0_Baseline",
    zone_df: Optional[pd.DataFrame] = None,
    degradation_model: str = "physics",
    n_samples: int = 20,
    uncertainty_pct: float = 0.10,
    analysis_years: int = 1,
    random_state: int = 42,
    time_step_hours: float | None = None,
    parameter_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Monte-Carlo robustness analysis using bounded uniform input perturbations."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    cfg0 = _clone_cfg(cfg)
    cfg0.years = int(max(1, analysis_years))
    if time_step_hours is not None:
        cfg0.TIME_STEP_HOURS = resolve_time_step_hours(time_step_hours)
    cfg0 = apply_hvac_preset(cfg0)
    base_weather, weather_meta = _load_base_weather(weather_mode, epw_path, csv_path, weather_df, random_state, cfg0.TIME_STEP_HOURS)
    rng = np.random.default_rng(random_state)
    specs = SENSITIVITY_PARAMETERS
    if parameter_names:
        wanted = set(parameter_names)
        specs = [sp for sp in specs if str(sp["attr"]) in wanted or str(sp["label"]) in wanted]

    sample_rows = []
    pct = float(uncertainty_pct)
    for i in range(int(n_samples)):
        b_i, c_i = _clone_bldg(bldg), _clone_cfg(cfg0)
        row = {"sample": i + 1}
        for spec in specs:
            current_obj = b_i if spec["group"] == "building" else c_i
            base_val = float(getattr(current_obj, str(spec["attr"])))
            factor = rng.uniform(1.0 - pct, 1.0 + pct)
            value = base_val * factor
            b_i, c_i = _set_nested_param(b_i, c_i, spec, value)
            row[f"input_{spec['attr']}"] = value
        summary = _single_summary_for_analysis(b_i, c_i, base_weather, fixed_strategy, fixed_severity, fixed_climate, zone_df, degradation_model, random_state + i)
        for kpi in SENSITIVITY_KPIS:
            row[kpi] = summary.get(kpi, np.nan)
        sample_rows.append(row)

    samples = pd.DataFrame(sample_rows)
    samples.to_csv(out / "robustness_samples.csv", index=False)

    summary_rows = []
    for kpi in SENSITIVITY_KPIS:
        if kpi in samples.columns:
            vals = pd.to_numeric(samples[kpi], errors="coerce").dropna()
            if len(vals):
                summary_rows.append({
                    "kpi": kpi,
                    "n": int(len(vals)),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "cv_pct": float(100.0 * vals.std(ddof=1) / max(abs(vals.mean()), 1e-9)) if len(vals) > 1 else 0.0,
                    "p05": float(vals.quantile(0.05)),
                    "p50": float(vals.quantile(0.50)),
                    "p95": float(vals.quantile(0.95)),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                })
    robust_summary = pd.DataFrame(summary_rows)
    robust_summary.to_csv(out / "robustness_summary.csv", index=False)

    if not robust_summary.empty:
        plt.figure(figsize=(9, 5.5))
        plot_data = [pd.to_numeric(samples[k], errors="coerce").dropna().to_numpy() for k in SENSITIVITY_KPIS if k in samples.columns]
        labels = [k.replace(" ", "\n") for k in SENSITIVITY_KPIS if k in samples.columns]
        plt.boxplot(plot_data, labels=labels, showmeans=True)
        plt.ylabel("KPI value")
        plt.title("Robustness analysis KPI spread")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(figs / "robustness_kpi_boxplot.png", dpi=600)
        plt.close()

    meta = {"method": "Monte Carlo bounded uniform perturbation", "uncertainty_pct": pct, "n_samples": int(n_samples), "analysis_years": cfg0.years, "fixed_strategy": fixed_strategy, "fixed_severity": fixed_severity, "fixed_climate": fixed_climate, "weather_summary": weather_meta, "time_step_hours": cfg0.TIME_STEP_HOURS}
    with open(out / "robustness_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return {"samples_csv": str(out / "robustness_samples.csv"), "summary_csv": str(out / "robustness_summary.csv"), "figures_dir": str(figs)}

# ---------- Surrogate + SHAP ----------
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"RMSE": float(np.sqrt(mse)), "MAE": float(mean_absolute_error(y_true, y_pred)), "R2": float(r2_score(y_true, y_pred))}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "day_of_year" not in out.columns and "day" in out.columns:
        out["day_of_year"] = ((out["day"] - 1) % 365) + 1
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.0)
    if "time_idx" not in out.columns and "day" in out.columns:
        out["time_idx"] = out["day"].astype(int)
    if "scenario_key" not in out.columns:
        if "scenario_combo_3axis" in out.columns:
            out["scenario_key"] = out["scenario_combo_3axis"].astype(str)
        else:
            parts = []
            for c in ["strategy", "severity", "climate"]:
                if c in out.columns:
                    parts.append(out[c].astype(str))
            if parts:
                val = parts[0]
                for p in parts[1:]:
                    val = val + "_" + p
                out["scenario_key"] = val
            else:
                out["scenario_key"] = "case"
    return out


def add_group_lags(df, group_col, cols, lags):
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby(group_col)[col].shift(lag)
    return out


def prepare_dataset_for_ml(df):
    out = df.copy()
    for c in ["strategy", "severity", "climate", "scenario_key", "hvac_system_type", "mode"]:
        if c in out.columns:
            out[c] = out[c].astype(str)
    lag_cols = ["energy_kwh_day", "delta", "comfort_dev_C", "COP_eff", "T_amb_C", "occ", "R_f", "dP_Pa", "hx_cleaned", "filter_replaced", "alpha_flow", "T_sp_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2", "Q_cool_kw", "Q_heat_kw"]
    out = add_group_lags(out, "scenario_key", lag_cols, [1, 7])
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def feature_map(df):
    cats = [c for c in ["strategy", "severity", "climate", "scenario_key", "hvac_system_type", "mode"] if c in df.columns]
    common = cats + [c for c in ["year", "day_of_year", "doy_sin", "doy_cos", "occ", "T_amb_C", "T_sp_C", "alpha_flow", "hx_cleaned_lag1", "filter_replaced_lag1", "hx_cleaned_lag7", "filter_replaced_lag7", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2", "Q_cool_kw", "Q_heat_kw"] if c in df.columns]
    fmap = {}
    if "energy_kwh_day" in df.columns:
        fmap["energy_kwh_day"] = common + [c for c in ["R_f", "dP_Pa", "delta_lag1", "delta_lag7", "energy_kwh_day_lag1", "energy_kwh_day_lag7", "COP_eff_lag1", "COP_eff_lag7", "T_amb_C_lag1", "T_amb_C_lag7", "occ_lag1", "occ_lag7", "T_sp_C_lag1", "alpha_flow_lag1", "RH_mean_pct_lag1", "GHI_mean_Wm2_lag1"] if c in df.columns]
    if "delta" in df.columns:
        fmap["delta"] = common + [c for c in ["energy_kwh_day_lag1", "energy_kwh_day_lag7", "delta_lag1", "delta_lag7", "COP_eff_lag1", "COP_eff_lag7", "T_amb_C_lag1", "T_amb_C_lag7", "occ_lag1", "occ_lag7", "R_f_lag1", "R_f_lag7", "dP_Pa_lag1", "dP_Pa_lag7", "T_sp_C_lag1", "alpha_flow_lag1", "RH_mean_pct_lag1", "GHI_mean_Wm2_lag1"] if c in df.columns]
    if "comfort_dev_C" in df.columns:
        fmap["comfort_dev_C"] = common + [c for c in ["R_f", "dP_Pa", "delta", "energy_kwh_day_lag1", "energy_kwh_day_lag7", "COP_eff_lag1", "COP_eff_lag7", "T_amb_C_lag1", "T_amb_C_lag7", "occ_lag1", "occ_lag7", "T_sp_C_lag1", "alpha_flow_lag1", "RH_mean_pct_lag1", "GHI_mean_Wm2_lag1"] if c in df.columns]
    return fmap


def auto_year_split(df):
    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    n = len(years)
    if n < 3:
        raise ValueError("Need at least 3 years in the dataset.")
    if n >= 20 and years[:20] == list(range(1, 21)):
        train_years, valid_years, test_years = list(range(1, 15)), [15, 16], [17, 18, 19, 20]
    else:
        n_train = max(1, int(round(n * 0.6)))
        n_valid = max(1, int(round(n * 0.2)))
        n_test = n - n_train - n_valid
        if n_test < 1:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_valid -= 1
        train_years = years[:n_train]
        valid_years = years[n_train:n_train+n_valid]
        test_years = years[n_train+n_valid:]
        if not valid_years:
            valid_years = [train_years.pop()]
        if not test_years:
            test_years = [valid_years.pop()]
            if not valid_years:
                valid_years = [train_years.pop()]
    return (
        df[df["year"].isin(train_years)].copy(),
        df[df["year"].isin(valid_years)].copy(),
        df[df["year"].isin(test_years)].copy(),
        {"train_years": train_years, "valid_years": valid_years, "test_years": test_years},
    )


def train_surrogate_models(input_csv: str | Path, output_dir: str | Path, n_iter_search: int = 6, shap_sample: int = 1000, random_state: int = 42) -> Dict[str, str]:
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost is not installed. Install dependencies from requirements_v3.txt")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    figs.mkdir(exist_ok=True)

    raw = pd.read_csv(input_csv)
    data = prepare_dataset_for_ml(add_time_features(raw))
    data.to_csv(out / "prepared_dataset.csv", index=False)

    fmap = feature_map(data)
    overall_rows = []
    shap_notes = []

    for target, feats in fmap.items():
        df = data.dropna(subset=list(set(feats + [target]))).copy()
        train_df, valid_df, test_df, split_info = auto_year_split(df)
        cat_features = [c for c in ["strategy", "severity", "climate", "scenario_key", "hvac_system_type", "mode"] if c in feats]

        param_dist = {
            "iterations": [300, 600, 1000],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "depth": [4, 5, 6, 8],
            "l2_leaf_reg": [1, 3, 5, 7, 10],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "random_strength": [0.0, 0.5, 1.0, 2.0],
        }
        best = None
        for params in ParameterSampler(param_dist, n_iter=n_iter_search, random_state=random_state):
            model = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", random_seed=random_state, verbose=False, **params)
            model.fit(train_df[feats], train_df[target], cat_features=cat_features, eval_set=(valid_df[feats], valid_df[target]), use_best_model=True, early_stopping_rounds=80, verbose=False)
            pred_valid = model.predict(valid_df[feats])
            valid_metrics = regression_metrics(valid_df[target].to_numpy(), pred_valid)
            if best is None or valid_metrics["RMSE"] < best["valid_metrics"]["RMSE"]:
                best = {"model": model, "params": params, "valid_metrics": valid_metrics}

        model = best["model"]
        pred_test = model.predict(test_df[feats])
        test_metrics = regression_metrics(test_df[target].to_numpy(), pred_test)
        overall_rows.append({"target": target, **best["valid_metrics"], "test_RMSE": test_metrics["RMSE"], "test_MAE": test_metrics["MAE"], "test_R2": test_metrics["R2"], **split_info})
        model.save_model(str(out / f"{target}_catboost_model.cbm"))

        pred_df = test_df.copy()
        pred_df["actual"] = test_df[target].to_numpy()
        pred_df["predicted"] = pred_test
        keep_cols = [c for c in ["strategy", "severity", "climate", "scenario_key", "year", "day_of_year", "actual", "predicted"] if c in pred_df.columns]
        pred_df[keep_cols].to_csv(out / f"{target}_test_predictions.csv", index=False)

        save_scatter(pred_df["actual"].to_numpy(), pred_df["predicted"].to_numpy(), f"{target}: Actual vs Predicted", figs / f"{target}_actual_vs_pred.png")
        plt.figure(figsize=(8, 6))
        importances = model.get_feature_importance()
        imp_df = pd.DataFrame({"feature": feats, "importance": importances}).sort_values("importance", ascending=False)
        top = imp_df.head(15).iloc[::-1]
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Importance")
        plt.title(f"{target}: CatBoost feature importance")
        plt.tight_layout()
        plt.savefig(figs / f"{target}_feature_importance.png", dpi=600)
        plt.close()
        imp_df.to_csv(out / f"{target}_feature_importance.csv", index=False)

        for group_col in ["strategy", "severity", "climate", "scenario_key"]:
            if group_col in pred_df.columns:
                rows = []
                for g, grp in pred_df.groupby(group_col):
                    rows.append({group_col: g, **regression_metrics(grp["actual"].to_numpy(), grp["predicted"].to_numpy()), "n": len(grp)})
                pd.DataFrame(rows).sort_values("RMSE").to_csv(out / f"{target}_metrics_by_{group_col}.csv", index=False)

        if SHAP_AVAILABLE:
            shap_df = test_df[feats].sample(n=min(shap_sample, len(test_df)), random_state=random_state).reset_index(drop=True)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(shap_df)
            shap_values = np.array(shap_values)
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(-1, 1)
            mean_abs = np.abs(shap_values).mean(axis=0)
            shap_imp = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
            shap_imp.to_csv(out / f"{target}_mean_abs_shap.csv", index=False)

            plt.figure(figsize=(8, 6))
            top = shap_imp.head(15).iloc[::-1]
            plt.barh(top["feature"], top["mean_abs_shap"])
            plt.xlabel("Mean |SHAP value|")
            plt.title(f"{target}: Top SHAP features")
            plt.tight_layout()
            plt.savefig(figs / f"{target}_shap_bar.png", dpi=600)
            plt.close()

            shap.summary_plot(shap_values, shap_df, show=False, max_display=15)
            plt.title(f"{target}: SHAP summary")
            plt.tight_layout()
            plt.savefig(figs / f"{target}_shap_summary.png", dpi=600, bbox_inches="tight")
            plt.close()

            shap_notes.append(f"{target}: top SHAP features = " + ", ".join(shap_imp['feature'].head(5).tolist()))

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out / "axis_catboost_overall_metrics.csv", index=False)

    export_surrogate_excel_report(out, overall_df)
    export_surrogate_pdf_report(out, overall_df, shap_notes)

    return {
        "metrics_csv": str(out / "axis_catboost_overall_metrics.csv"),
        "excel_report": str(out / "surrogate_export.xlsx"),
        "pdf_report": str(out / "surrogate_report.pdf"),
        "figures_dir": str(figs),
    }


def save_scatter(y_true, y_pred, title, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.35)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def export_surrogate_excel_report(out: Path, overall_df: pd.DataFrame):
    with pd.ExcelWriter(out / "surrogate_export.xlsx", engine="openpyxl") as writer:
        overall_df.to_excel(writer, sheet_name="overall_metrics", index=False)


def export_surrogate_pdf_report(out: Path, overall_df: pd.DataFrame, shap_notes: List[str]):
    figs = sorted((out / "figures").glob("*.png"))
    with PdfPages(out / "surrogate_report.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.text(0.06, 0.96, "Surrogate Model Report", va="top", fontsize=16)
        plt.text(0.06, 0.90, overall_df.to_string(index=False), va="top", family="monospace", fontsize=8)
        if shap_notes:
            plt.text(0.06, 0.52, "Key SHAP notes:\n- " + "\n- ".join(shap_notes), va="top", fontsize=10)
        pdf.savefig(fig, dpi=300); plt.close(fig)
        for img in figs[:8]:
            arr = plt.imread(img)
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.imshow(arr)
            plt.axis("off")
            plt.title(img.name)
            pdf.savefig(fig, dpi=300); plt.close(fig)
