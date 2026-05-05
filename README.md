# HVAC ROM-Degradation Suite — Strong Coupled Publication Version

This deployable Streamlit package extends the Publication Plus + EMS/Optimization version with **optional fully coupled diagnostic modules**. In earlier versions, the heat-exchanger, part-load COP, latent-load, and zone-level tabs mainly worked as post-processing/diagnostic layers. In this version, these modules can also modify the official Scenario Modeling outputs when their switches are enabled.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

or:

```bash
python -m streamlit run streamlit_app.py
```

## Main entry files

- `streamlit_app.py` — full graphical interface
- `hvac_v3_engine.py` — numerical engine and coupled calculations
- `report_addons.py` — reporting, plotting, validation, diagnostics, optimization helpers
- `requirements.txt` — dependencies

## Strong coupled switches

Open **Parameter Switches** and enable one or more of:

1. **Apply part-load COP to core**
   - Applies the selected PLR curve to `COP_eff` inside the time-step simulation.
   - Affects thermal HVAC energy, total energy, CO₂, cost, and objective value.

2. **Apply latent cooling to core**
   - Calculates humidity-ratio-based latent cooling from outdoor weather, indoor RH target, ventilation fraction, and infiltration.
   - Adds latent cooling to `Q_cool_kw` before calculating compressor/thermal HVAC energy.

3. **Apply HX air ΔP to fan**
   - Replaces/supplements the simple fan pressure term with a heat-exchanger air-side pressure-drop model:
     `ΔP_air = ΔP_clean × AF² × (1 + fouling_factor × degradation)`.
   - Affects fan power and total energy.

4. **Apply HX water ΔP to pump**
   - Replaces simple pump W/m² power with detailed water-side pressure-drop pump power:
     `P_pump = Vdot_water × ΔP_water / pump_efficiency`.
   - Affects pump energy and total energy.

5. **Apply HX UA degradation to capacity**
   - Reduces available cooling/heating capacity using:
     `capacity_factor = max(0.40, 1 − UA_loss_factor × degradation)`.
   - Can increase `capacity_unmet_kw` and comfort deviation.

6. **Use native zone-by-zone load sum**
   - If a zone table is supplied, loads are calculated per zone using each zone area, occupancy density, and schedule factors, then summed to building-level load.
   - Affects `Q_cool_kw`, `Q_heat_kw`, energy, CO₂, and comfort.

All coupled switches default to **OFF** to preserve the earlier core-model results. Turn them ON for the stronger publication formulation.

## New coupled output columns

The official time-step dataset can now include:

- `COP_base_before_PLR`
- `PLR`
- `PLR_modifier`
- `latent_cooling_kw`
- `dP_fan_Pa`
- `dP_water_kPa`
- `water_flow_m3h`
- `hx_capacity_factor`
- `capacity_unmet_kw`
- `zone_load_mode`
- `coupled_modules_active`

Energy-balance columns remain:

- `thermal_hvac_kwh_period`
- `fan_kwh_period`
- `pump_kwh_period`
- `auxiliary_kwh_period`
- `energy_kwh_period`

where:

```text
energy_kwh_period = thermal_hvac_kwh_period + fan_kwh_period + pump_kwh_period + auxiliary_kwh_period
```

## Recommended publication workflow

1. Run the baseline with all coupled switches OFF.
2. Enable one coupled switch at a time and compare outputs.
3. Enable the final set of coupled switches for the publication scenario.
4. Use **Sensitivity & Robustness** to quantify uncertainty.
5. Use **Advanced Plot Studio** to create final figures.
6. Use **Model Validation** when measured, EnergyPlus, DesignBuilder, BMS, or utility data are available.

## Honest limitation

This is still a reduced-order HVAC model. The coupled modules improve physical consistency and make diagnostics affect the official energy results, but they do not turn the model into a full heat-balance simulator like EnergyPlus. Case-specific validation is still required before making strong absolute-performance claims.

## Advanced HVAC Control Library update

This bundle adds a new **Advanced HVAC Control Library** tab. It includes deployable control templates for:

- ASHRAE Guideline 36-style high-performance control sequence approximation
- Demand-controlled ventilation through occupancy/airflow reset
- Fault-adaptive control recommendations
- Degradation-aware maintenance optimization recommendations
- Carbon-aware / price-aware control
- Peak demand limiting
- Zone priority control
- Hybrid degradation-aware EMS controller

The tab can score and rank control templates using energy, comfort, degradation, carbon, and fault-risk weights. A selected control template can be applied to the EMS settings and used in the main Scenario Modeling tab.

### Experimental controllers

The same tab includes two experimental research structures:

- **MPC experimental template:** prepares forecast and decision columns for future model predictive control work. It is not a validated live MPC solver.
- **RL experimental dataset specification:** defines state/action/reward fields for offline reinforcement learning dataset generation. It does not train or deploy an RL control policy.

These experimental tabs are intentionally separated from deployable control templates to avoid claiming real-time autonomous control without validation.
