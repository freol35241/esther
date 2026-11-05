# esther v1 — specification

Status: **draft for review.** Nothing in this document is built yet. The v0 line
(single-state model, hand-set parameters, SLSQP, MQTT service) is preserved at
tag `v0-legacy`; the unmerged dev line ends at `v0.4.0_dev9`. v1 is a rewrite in
the same repository, not an increment.

This file is the contract for the build. Where it says MUST, a change needs an
edit here first. Where it says *hypothesis*, the claim is to be tested on data
in milestone M1 before anything depends on it.

---

## 1. Purpose

esther plans the heating of a house: when to run which heat source, how hard,
and how to use the house's own thermal storage — the building fabric and the
hot-water tank — to buy energy when it is cheap, while keeping the people in it
comfortable and the hot water hot.

It does that with **economic model predictive control** over a **grey-box
thermal model that learns its parameters from the house's own history**.

### 1.1 Goals

- **G1. Composable.** A house is described, not programmed. Any house built
  from the component library (§4) is a configuration file; a new kind of
  equipment is one new component class, registered without touching the core.
- **G2. Self-learning.** Starting from physical priors (rated powers, volumes,
  datasheet curves), esther fits its model to measured data, re-fits it as data
  accumulates, and adopts a new fit only when it predicts better on data it has
  not seen.
- **G3. Multi-source.** Several heat sources with different costs, dynamics and
  discreteness — a heat pump, an electric heater, a pellet stove — are planned
  together, from the first version.
- **G4. Hot water is part of the plan.** The tank is a store to be charged when
  energy is cheap, against a forecast of when hot water will be drawn.
- **G5. Safe by construction.** esther proposes; executors outside it act. A
  plan that stops arriving MUST degrade every actuator to a defined, safe
  behaviour without esther's help.
- **G6. Verifiable.** Every model and every controller change can be replayed
  against recorded history, including the forecasts that were available at the
  time, and scored.

### 1.2 Non-goals

- **No I/O in the core.** esther does not speak MQTT, HTTP or any bus. It is a
  library; an integration layer (§9) feeds it and carries out its plans. (v0
  was an MQTT service; that shape is retired.)
- **No device safety.** Device safety layers (a stove's own cutouts, a heat
  pump's pressure switches) stay where they are. esther's constraints are
  economy and comfort, never the last line of defence.
- **No cloud dependency** in the planning path.
- **Not a building simulator.** The models are as simple as the data can
  identify, not as detailed as physics allows.

### 1.3 Terminology

| Term | Meaning |
|---|---|
| **Node** | A lumped thermal capacity with one temperature state (a room's air, the building mass, a tank layer) |
| **Coupling** | A heat path between two nodes, or a node and a boundary, with a conductance UA |
| **Boundary** | A temperature imposed from outside the model (outdoor air, ground, cold mains water) |
| **Source** | A component that injects heat into a node, at a cost, under decisions |
| **Load** | Heat removed by something other than a coupling (a hot-water draw, ventilation) |
| **Decision** | A variable the controller chooses per step: continuous (heat rate) or integer (on/off, a level, a mode) |
| **Exogenous input** | A variable the controller does not choose but must forecast: weather, price, occupancy, draws |
| **Actuation** | The translation of a decision into what a device actually accepts (a fake sensor value, a button press) |
| **Executor** | The thing outside esther that holds the device and carries out actuation, with its own fallback |
| **Plan** | The controller's output: decisions over the horizon, the predicted trajectories, and their provenance |

---

## 2. Requirements

### 2.1 Functional

- **F1.** Load a house description (§5) and build the composed model from it.
- **F2.** Build a training dataset from a history source at a fixed resolution,
  with gaps marked and never interpolated across (§6.1).
- **F3.** Fit model parameters by maximum a posteriori estimation over the
  priors, with uncertainty (§6).
- **F4.** Validate a fit by multi-step simulation on held-out data and report
  the metrics in §6.5; refuse a fit that fails the acceptance gate.
- **F5.** Estimate the current state, including unmeasured states (building
  mass, tank layers), from live measurements (§6.6).
- **F6.** Forecast exogenous inputs over the horizon (§7).
- **F7.** Solve the planning problem (§8) and return a Plan.
- **F8.** Translate the Plan's first step(s) into per-actuator commands (§8.7).
- **F9.** Replay any past period as if esther had been running — backtest —
  using the forecasts that were available then (§10).

### 2.2 Non-functional

- **N1.** A plan for a 36 h horizon at 15 min steps MUST solve in under 10 s on
  a small x86 server, and SHOULD solve in under 2 s. A solve that exceeds its
  time limit returns the incumbent if one exists, flagged, or no plan.
- **N2.** Deterministic: the same inputs, parameters and solver version give the
  same plan.
- **N3.** Python ≥ 3.11, pure-Python dependencies plus wheels from PyPI. No
  compiled code of our own.
- **N4.** Internal units are SI: W, J, K, s, kg. °C appears only at the I/O
  boundary and in configuration.
- **N5.** Every public function has a test; every model component has a test
  that recovers known parameters from data it simulated itself (§10.1).

---

## 3. Architecture

```
            ┌──────────── integration layer (outside esther) ─────────────┐
            │  history reader   live inputs   executors (one per device)  │
            └─────┬──────────────────┬──────────────────▲─────────────────┘
                  │ dataset          │ measurements      │ commands
                  ▼                  ▼                   │
  ┌──────────────────────────── esther ──────────────────┴──────────────┐
  │  house  ──► composed model ──► estimator (fit, filter)              │
  │  (config)        │                     │ params, state              │
  │                  ▼                     ▼                            │
  │            forecasters ─────────► planner (MILP) ──► plan ──► actuation
  │                                                                     │
  │  backtest harness: replays history through all of the above         │
  └─────────────────────────────────────────────────────────────────────┘
```

### 3.1 Package layout

```
esther/
  core/          model graph, variables, parameter/prior types, units
  components/    the component library (§4); each module registers its types
  house.py       house description loading and validation (§5)
  data/          dataset building, resampling, gap handling (§6.1)
  identify/      Kalman filter, likelihood, MAP fit, validation (§6)
  forecast/      forecaster interface and implementations (§7)
  plan/          problem assembly, solver backend, plan type (§8)
  actuate/       actuation translators (§8.7)
  backtest/      replay harness and scoring (§10)
  io/            OPTIONAL readers, e.g. io.homeostat (history store) — extras only
```

`core`, `components`, `identify`, `forecast`, `plan`, `actuate` and `backtest`
MUST NOT import anything from `io`. Readers are optional extras
(`pip install esther[homeostat]`).

### 3.2 Dependencies

numpy, scipy, pandas (dataset handling only), cvxpy with HiGHS for the MILP
(open: cvxpy vs linopy, decided in M0 by a timing spike on the reference
problem), tomllib (stdlib). No CasADi, no JAX in v1; revisit if the fit needs
automatic differentiation.

---

## 4. Component model

### 4.1 The contract

Every component is a class implementing one protocol. A component contributes
some or all of:

| Contribution | Form |
|---|---|
| **States** | named, with units, initial-state prior |
| **Parameters** | named, with a prior (§4.2), bounds and a transform |
| **Ports** | the nodes it attaches to, by name, resolved at composition |
| **Dynamics** | its terms of the continuous-time model `C dT/dt = Σ heat flows`, as linear contributions to A, B, E (§4.3) |
| **Decisions** | per-step variables, continuous or integer, with bounds |
| **Exogenous inputs** | named signals it needs, and which forecaster supplies them |
| **Constraints** | per-step and inter-step (minimum run times, ramps), linear in decisions and states |
| **Cost** | per-step terms, linear in decisions (and in states for comfort) |
| **Measurements** | which states or outputs a named signal observes, with noise prior |
| **Actuation** | optional: how its decisions become device commands (§8.7) |

The core knows nothing about heat pumps or stoves. It composes contributions:
it stacks states, sums heat flows into nodes, concatenates constraints and adds
costs.

### 4.2 Parameters and priors

Every parameter is declared as

```python
Param("UA_env", unit="W/K", prior=LogNormal(median=150, gsd=2.0), bounds=(10, 2000))
```

- Positive physical quantities are fitted in log space.
- Priors come from the house description (datasheet values, rough sizes); a
  component supplies a weak default when the house gives none.
- The prior enters the fit as a penalty (MAP). With little data the fit stays
  near the prior; with good data the data dominates. This is how a "new house"
  starts.
- A parameter MAY be pinned (`fixed = true`) — a measured tank volume, a rated
  power — and is then not fitted.

### 4.3 The dynamics are linear in the states

For a fixed parameter vector the composed thermal model is

```
dx/dt = A(θ) x + B(θ) u + E(θ) d + w,      y = H(θ) x + v
```

with `x` the node temperatures, `u` the heat-rate decisions (W), `d` the
exogenous inputs (boundary temperatures, solar, internal gains, loads), and
`w, v` process and measurement noise. Exact zero-order-hold discretisation per
step (matrix exponential; Van Loan for the noise covariance), which handles
uneven steps exactly — v0's one good numerical idea, generalised.

Non-linearities live **outside** the state equation, where they can be handled
by precomputation per step:

- **COP** depends on temperatures. It is evaluated per step on the forecast
  boundary and the previous plan's predicted sink temperature, and enters the
  problem as a known coefficient (sequential linearisation, at most two passes;
  §8.5).
- **Discrete behaviour** (on/off, levels, start-up) is expressed with integer
  decisions and linear constraints, never inside A.
- **Mixing limits** (a shunt cannot deliver hotter than its source) are linear
  inequalities on heat rates and temperatures (§4.5.3).

### 4.4 Composition

Components attach by **port name**. A node is created by the component that
owns it (a `Zone` owns its air node, a `Tank` its layer nodes) and referenced by
others (`Radiators.zone = "house"`, `Stove.zone = "living"`). Composition fails
— with a message naming the component and the port — if a referenced node does
not exist, if a node has no path to a boundary (its temperature would be
unbounded), or if a source has no node to heat.

### 4.5 The library (v1)

Each entry gives the component's equations. Symbols: `T` temperature, `C` heat
capacity (J/K), `UA` conductance (W/K), `Q` heat rate (W), `P` electrical power
(W), `Δt` step length.

#### 4.5.1 Envelope

- **`Zone`** — an air node. `C_air dT_z/dt = Σ Q_in`. Measured by one or more
  room sensors, each with its own bias parameter (a sensor on a cold wall reads
  low; that is a bias, not a dynamic).
- **`Mass`** — a building-fabric node coupled to a zone: `C_m dT_m/dt =
  UA_zm (T_z − T_m) + f_solar_m · Q_sol`. Unmeasured; it is the house's heat
  store and the reason a 1-state model cannot plan pre-heating.
- **`Coupling`** — `Q = UA (T_a − T_b)` between any two nodes or a node and a
  boundary.
- **`Ambient`** — a boundary: outdoor air, from a measurement or forecast.
- **`Ventilation`** — a coupling to ambient whose conductance is
  `ρ c_p V̇` with the flow `V̇` known or fitted. For an exhaust-air heat pump the
  extracted air passes the evaporator; see 4.5.3.

The default house is `Zone + Mass + Coupling(zone, ambient) + Coupling(mass,
ambient)` — the classic 2R2C. M2 compares 1-, 2- and 3-node structures and the
house description chooses.

#### 4.5.2 Gains

- **`SolarGain`** — `Q_sol = g_A · I` where `I` is an irradiance proxy (a PV
  inverter's output, a lux sensor, or a clear-sky model times cloud cover);
  `g_A` is fitted and split between zone and mass by `f_solar_m`.
- **`InternalGain`** — a time-of-week profile (24 × 7 hourly values, smoothed,
  fitted with a smoothness prior), optionally scaled by occupancy. Absorbs
  people, cooking and appliances.
- **`ElectricalLoadGain`** — optional: a fraction of a measured household
  electrical load ends up as heat.

#### 4.5.3 Hydronic system

- **`Radiators`** — heat from a supply to a zone. Two forms:
  - *Decision form (planning):* the decision is `Q_rad ≥ 0`; the supply
    temperature it needs is derived after the solve as
    `T_supply = T_z + Q_rad / UA_rad` (linearised emitter; the exponent-n
    emitter law is a later refinement).
  - *Constraint:* `Q_rad ≤ UA_rad (T_source − T_z)` where `T_source` is the
    node the circuit mixes from — the shunt can only mix down, never up.
  - *Measurement form (identification):* with the supply temperature measured,
    `Q_rad = UA_rad (T_supply,meas − T_z)` is an input, and `UA_rad` is fitted.
- **`JacketTank`** — a double-jacket tank: an outer jacket of heating water and
  an inner vessel of domestic hot water. Two nodes (`jacket`, `dhw`), coupled by
  `UA_jd`. Sources heat the jacket; the radiator circuit draws from the jacket;
  draws cool the inner vessel. Losses to its room.
- **`StratifiedTank`** — `n` layers (n = 1–3) with inter-layer conductance and
  draw-driven advection; for plain DHW cylinders and buffer tanks.

#### 4.5.4 Sources

- **`HeatPump`** — generic. Decision: heat rate `Q_hp ∈ [0, Q_max(T_src, T_sink)]`,
  or for a fixed-speed machine an on-fraction `δ ∈ [0, 1]` per step with
  `Q_hp = δ · Q_rated`. Electrical power `P = Q_hp / COP(T_src, T_sink)`.
  `COP` and `Q_max` are a two-variable table or a Carnot-fraction model
  `COP = η · T_sink / (T_sink − T_src)`, with `η` fitted and clipped to the
  datasheet range. `T_src` is outdoor air, exhaust air, brine or ground,
  according to the type.
  - Minimum run and minimum off time as constraints on the on-fraction (§8.3).
  - Starts per day MAY be penalised.
- **`ElectricHeater`** — an immersion heater or flow heater: `P = Q`
  (COP 1), in `k` discrete steps or continuous, with a maximum.
- **`Stove`** — air-heating combustion stove with automatic fuel feed. Heats its
  zone's air directly (`f_rad` of it to mass, fitted). Decision per step: an
  integer level `ℓ ∈ {off, 1, …, L}`, each level with rated heat output and fuel
  rate. Start-up: output ramps over `t_ign` (fitted, prior from the manual);
  restart requires `t_cool` since the last stop (a minimum off time); minimum
  burn time `t_min_on`. Cost per kWh of heat = fuel price / (fuel energy density
  × efficiency). **Manual firing** (wood) is not planned: it is detected from
  measurements (§7.5) and enters as an exogenous heat input.

#### 4.5.5 Loads

- **`HotWaterDraw`** — energy drawn from a DHW node:
  `Q_draw = ṁ c_p (T_dhw − T_mains)`. In planning, a forecast draw energy per
  step (§7.4). In identification, draws are **detected** from the tank's
  temperature signature (§6.3) — no flow meter is assumed.

#### 4.5.6 Controller-wrapped devices

Many installed heat sources run their own controller, which esther cannot
replace and can only steer, usually by falsifying one of its inputs. These are
**wrappers**: a component that contains a physical source plus a model of the
device's own control logic, and whose actuation translates esther's decision
into the input that makes the device do it.

- **`CurveControlledHeatSource`** — a device that computes its supply target
  from a heating curve over an outdoor sensor. esther's decision (the supply
  temperature, via `Q_rad`) is translated to a **fake outdoor temperature**
  through the inverse curve. The wrapper also models what the device does on
  its own (auxiliary-heat timers, summer mode) as constraints or as a cost on
  the states that trigger it.
- **`ThermostatChargedTank`** — a device that charges a tank between a start and
  a stop temperature on its own tank sensor. esther's integer decision
  `mode ∈ {block, auto, boost}` is translated to a fake tank-sensor value.

The reference plant in Appendix A is two such wrappers around one compressor.

#### 4.5.7 Economics and comfort

- **`Tariff`** — electricity: spot price per step plus fixed per-kWh fees plus
  VAT, as configured; fuel: price per kg. The price forecast supplies the spot
  part (§7.2). Export price, if any, for houses with PV.
- **`ComfortBand`** — per zone: a setpoint schedule and a band. Soft: linear
  penalty per K·h outside the band, quadratic optional. Hard limits (a floor
  that is never planned below) are separate and wider.
- **`DhwService`** — per DHW node: minimum temperature at the tap, enforced as a
  soft constraint whose penalty is large, and evaluated at the forecast draw
  times (§8.4).

---

## 5. House description

A house is one TOML file. Every value with a unit carries it in the key name or
the value is in the default unit for that kind of quantity. Example (the
reference plant of Appendix A, abbreviated):

```toml
schema = 1
name = "reference"
step = "15min"
horizon = "36h"

[[component]]
type = "Zone"
name = "house"
prior.C = { median = "8 MJ/K", gsd = 2.0 }

[[component]]
type = "Mass"
name = "fabric"
zone = "house"
prior.C = { median = "60 MJ/K", gsd = 3.0 }

[[component]]
type = "Ambient"
name = "outdoor"
signal = "outdoor_temperature"

[[component]]
type = "Coupling"
between = ["house", "outdoor"]
prior.UA = { median = "120 W/K", gsd = 2.0 }

[[component]]
type = "JacketTank"
name = "tank"
volume_jacket = "55 l"
volume_dhw = "145 l"
room = "house"

[[component]]
type = "Radiators"
zone = "house"
source = "tank.jacket"
supply_signal = "supply_temperature"

[[component]]
type = "Stove"
name = "stove"
zone = "house"
levels = [ { heat = "0.7 kW", fuel = "0.15 kg/h" },
           { heat = "3.4 kW", fuel = "0.78 kg/h" },
           { heat = "6.8 kW", fuel = "1.56 kg/h" } ]
efficiency = 0.87
fuel = "pellets"
prior.t_ign = { median = "20 min", gsd = 1.5 }

[[component]]
type = "ivt490.Plant"          # a registered wrapper, Appendix A
tank = "tank"
radiators = "radiators"
curve_slope = 3.0
heater_max = "9 kW"
```

Rules:

- Component types are resolved by name from a registry. Third-party packages
  register types through the `esther.components` entry point; nothing in the
  core changes to add one.
- Validation runs before anything else: unknown keys, missing ports, unit
  mismatches and unreachable nodes are errors with the file position.
- **Signals are named, not addressed.** The house description says
  `signal = "outdoor_temperature"`; mapping that name to a bus topic, an entity
  or a database column is the integration layer's business (§9). The same house
  file therefore works against a live bus, a history store and a simulator.

---

## 6. Identification

### 6.1 Dataset

- Fixed resolution, **5 min** for identification (the planning step is 15 min;
  identification needs the finer step to see compressor cycling and stove
  start-up).
- Per signal and step: mean, and for binary signals the on-fraction (a
  compressor on for 3 of 5 minutes is `δ = 0.6`).
- **Gaps are segments.** A step with any required signal missing for more than
  its tolerance ends the segment. The filter restarts at the next segment from
  its prior; nothing is interpolated across a gap.
- Signals carry a validity mask from the integration layer (device available,
  value plausible). Implausible values (a disconnected sensor reading −51 °C, an
  ADC spike) are masked, not clipped.
- Periods the house description marks as non-representative (an outage, a
  manual override, a deliberately wrong setpoint) are excluded by date range.

### 6.2 Estimation

- **Method:** continuous-discrete linear Kalman filter; negative log-likelihood
  of the innovations plus the negative log-prior; minimised with L-BFGS-B over
  the transformed parameters; multi-start (≥ 8 starts drawn from the prior).
  This is the CTSM approach, without the dependency.
- **Uncertainty:** the inverse Hessian at the optimum, in transformed space.
  Reported per parameter as a 95 % interval.
- **Identifiability check:** condition number of the Hessian and pairwise
  correlation. A parameter whose interval is not narrower than its prior's by a
  configured factor is reported as *not identified by the data* — the fit still
  stands, resting on the prior for that parameter, and the report says so.

### 6.3 Draw detection (hot water without a flow meter)

A draw is a drop in the DHW node's measured temperature faster than standby
loss explains, not coinciding with a charge. Detector:

1. Residual of the tank sub-model's one-step prediction with draws set to zero.
2. A draw event where the residual is below `−k σ` for at least one step;
   energy = `C_dhw × Δ T_residual` integrated over the event.
3. Events are fitted jointly with the tank parameters in an outer loop (fit
   with events, re-detect, refit; at most three passes).

Validated against a labelled sample: the house owner records a handful of
known draws (a shower, a bath) with times, and the detector MUST find them with
the right order of magnitude of energy. Without labels the detector's output is
reported, not trusted.

### 6.4 The fitting split

Fitted jointly where they share data, separately where they do not:

- Envelope + gains + radiators: from zone temperatures, supply temperature,
  weather. Sources enter as measured heat inputs where measured.
- Tank: from tank sensors, compressor and heater state, detected draws.
- Source performance (COP, heater step size, stove output per level): from
  electrical and fuel measurements where they exist; otherwise from the prior
  alone, flagged.

### 6.5 Validation and the acceptance gate

Held-out data: the most recent 20 % of the dataset by time (never random rows;
autocorrelation makes random splits optimistic).

Metrics per measured signal:

- RMSE of **open-loop simulation** from each segment start at 1 h, 6 h and 24 h
  horizons, with measured inputs. Open-loop is the metric that matters; it is
  what the planner relies on.
- One-step innovation RMSE (for the filter).
- Residual autocorrelation (Ljung–Box at lags 1–24 h); reported, not gating.
- Bias per time-of-day (catches a missing gain).

A new fit is **accepted** only if its 24 h open-loop RMSE on the zone
temperature is not worse than the current fit's on the same held-out window,
and no parameter sits on a bound. Otherwise the current fit stays and the
refusal is reported with the numbers.

### 6.6 Online state estimation

The same Kalman filter, run forward on live measurements at the identification
resolution, gives the state estimate (including unmeasured mass and tank
nodes) at plan time. Parameters are not estimated online in v1: joint state and
parameter estimation drifts quietly and is hard to audit. Re-fitting (G2) is a
scheduled batch job — weekly by default — through the acceptance gate.

---

## 7. Forecasting

Every forecaster implements one interface: `forecast(now, horizon, step) →
(values, provenance)`. Provenance records what was used and when it was
issued, so a backtest can prove it used only information available at `now`.

### 7.1 Weather

Outdoor temperature and cloud or irradiance from a numerical forecast,
resampled to the step **using its own timestamps** (v0 assumed one value per
hour and discarded the times; forecast resolution coarsens with lead time). The
first steps are blended from the current measurement to the forecast over a
configured time constant (a forecast is often biased at the house for the
first hours).

### 7.2 Price

Day-ahead spot prices at their native resolution (15 min). Beyond the published
horizon: a forecast from the last 4 weeks' time-of-week profile, flagged as
estimated. The planner treats estimated prices with a terminal value (§8.6)
rather than optimising hard against a guess.

### 7.3 Occupancy

From the integration layer's presence signal: a time-of-week probability profile
learned from history, overridden by the current state for the first hours
(someone at home now is likely home in 30 minutes).

### 7.4 Hot-water draws

From detected draw events (§6.3): energy per step as a time-of-week empirical
distribution, conditioned on occupancy. The planner uses a **high quantile**
(default 80th) — a cold shower costs more goodwill than the energy saved. The
quantile is a house setting.

### 7.5 Manual firing and other unplanned heat

A stove lit by hand is detected from its flue or combustion-chamber temperature
rising without a planned start, or above the manual's wood threshold. While
detected it is an exogenous heat input at the stove's wood output, decaying
over a fitted time after the signature ends. It is never forecast ahead.

### 7.6 Device-internal events

Events the device schedules itself (a periodic anti-legionella raise) are
forecast from their observed period and last occurrence, and enter as a fixed
load on the plan — esther MUST NOT plan to suppress them (§8.4).

---

## 8. Planning

### 8.1 Horizon and steps

Default horizon 36 h at 15 min (144 steps), aligned to the price grid; the first
step is shortened to the next grid boundary. Re-planned every step, and on any
major input change (a new price issue, a stove lit by hand, a state estimate
jumping more than a configured amount).

### 8.2 Variables

- Continuous: `Q_hp,k` or `δ_k`, `Q_heater,k`, `Q_rad,k`, slack variables for
  comfort and DHW service, and the state trajectory `x_k`.
- Integer: stove level per step (one-hot binaries), stove start indicator,
  heat-pump on indicator where minimum run/off times apply, tank mode per step
  for `ThermostatChargedTank`.
- Problem size for the reference plant: about 144 × (6 continuous + 6 binary)
  — a small MILP; HiGHS solves problems of this size in well under a second.
  M0's spike verifies that.

### 8.3 Constraints

- Dynamics: `x_{k+1} = A_d x_k + B_d u_k + E_d d_k` (from §4.3).
- Source limits: `0 ≤ Q ≤ Q_max,k`; on-fractions in [0, 1].
- Minimum run/off: for any source with them, on/off indicators and the
  standard `t_min` window constraints. A minimum run time shorter than one step
  is handled by the on-fraction.
- Stove: one level per step; start indicator `s_k ≥ on_k − on_{k−1}`; minimum
  burn time; minimum off time `t_cool` after a stop; ignition ramp as reduced
  output in the first `⌈t_ign/Δt⌉` steps after a start. Permitted steps only
  (§8.4).
- Mixing: `Q_rad,k ≤ UA_rad (T_jacket,k − T_z,k)`.
- Device envelopes: supply temperature limits; tank maxima per source (a heat
  pump that cannot lift the tank above its condensing limit).
- Initial state: the filter's estimate; the comfort band widened to include
  the current temperature (v0's rule, kept: the problem must be feasible from
  wherever the house is).

### 8.4 Policy constraints

House policy is configuration, not code:

- `Stove.allowed`: a predicate over steps — by default "someone forecast home
  with probability ≥ p", and never between configured hours. Steps where it is
  false force the level to off.
- `DhwService.min_temp` at forecast draw times.
- Device-internal events (§7.6): the steps they occupy fix the affected
  decisions to "let the device do it" (`mode = auto` for the tank).
- Manual overrides from the integration layer: a pinned decision for a period.

### 8.5 Objective

```
minimise  Σ_k [ c_el,k · Σ_sources P_k Δt              electricity at the step's price
              + c_fuel · Σ_stove fuel_k Δt            fuel
              + c_start · s_k                          per start (wear, emissions)
              + w_comfort · (σ⁺_k + σ⁻_k) Δt           comfort slack, K·h
              + w_dhw · σ_dhw,k ]                      DHW service slack
          − V_terminal(x_N)                            value of heat stored at the end
```

`P_k = Q_hp,k / COP_k` with `COP_k` fixed per step for the solve. After the
solve, `COP_k` is recomputed on the planned sink temperatures and the problem
re-solved once if any step's COP moved by more than 5 %.

### 8.6 Terminal value

Without it the plan empties every store at the horizon. `V_terminal` values the
energy in each storage node above its band floor at the mean of the estimated
part of the price forecast, divided by the source COP that would replace it.
Linear in the terminal state.

### 8.7 Actuation

The Plan carries the full decision trajectory. Actuators consume the first step
(or the stove's start/stop times, which may lie further ahead) through
translators:

- **Fake outdoor temperature** for curve-controlled devices: the supply target
  `T_supply` (from §4.5.3) through the inverse curve, clamped to the device's
  accepted range. Rewritten at the executor's cadence for as long as the plan
  is fresh.
- **Fake tank-sensor value** for thermostat-charged tanks: the mode, as the
  device's own interface defines it.
- **Discrete schedule** for stoves: a start time and level, a stop time — as
  absolute times, so that an executor can hold them if the planner falls
  silent.

Every translated command carries the plan's id and expiry.

### 8.8 Failure behaviour

- **No feasible plan or solver timeout without an incumbent:** no plan is
  published. Executors fall back (§9.2).
- **Inputs stale** (a forecast older than its maximum age, a required
  measurement silent): no plan. v0's staleness gate, kept, and made explicit
  per input.
- **Model not accepted** (no fit has passed the gate): planning runs in
  **advisory mode** — plans are published, marked advisory, and executors MUST
  NOT act on them.

---

## 9. Integration contract

esther ships no integration. The integration layer — for the reference plant a
homeostat unit — owns:

### 9.1 Inputs it provides

- A **history reader** that returns named signals with timestamps and validity,
  for the dataset builder.
- **Live measurements** by signal name, with their age.
- **Forecast sources** (raw forecasts with issue times) for esther's
  forecasters.
- **Presence**, manual overrides, and the policy predicate inputs.

### 9.2 Executors it runs

One executor per device, each with its own fallback that does not depend on
esther:

| Device kind | On a fresh plan | When the plan goes stale |
|---|---|---|
| Curve-controlled heat source | rewrite the fake outdoor temperature on cadence | stop rewriting; the device's own validity expires and it falls back to its curve |
| Thermostat-charged tank | hold the mode | stop; the device's mode expires to auto |
| Stove | start at the planned time and level; **stop at the planned stop time even if no newer plan arrives** | no new starts; a burn in progress still stops at its planned stop time, and the device's own rules and cutouts remain |

The asymmetry is deliberate: for a heat pump silence is safe, for a lit stove it
is not, so the stove's stop is part of the start.

### 9.3 What esther publishes

The Plan: decisions and predicted trajectories over the horizon, the current
state estimate (including unmeasured nodes), the model version, the fit's
validation metrics, provenance of every forecast used, and the mode (active or
advisory). The integration layer records all of it, so that every plan can be
compared with what happened.

---

## 10. Verification

### 10.1 Synthetic recovery (the positive control)

For every component and for the composed reference plant: simulate data from
known parameters with realistic noise, gaps and excitation; fit; require the
true parameters inside the reported 95 % intervals. A fit routine that cannot
recover parameters it generated is not used on real data.

### 10.2 Backtest

Replay a past period step by step: state estimate from the measurements up to
`now`, forecasts as issued before `now`, plan, then score:

- predicted vs measured zone and tank trajectories at 1/6/24 h;
- the plan's cost evaluated on the realised prices.

A backtest cannot score the counterfactual cost of acting on the plan (the house
did something else); that is what §10.4 is for.

### 10.3 Shadow

Live, advisory mode: plans published and recorded, nothing acted on. Exit
criterion per actuator: two weeks of shadow with the 24 h prediction error
inside the acceptance gate and no plan the owner judges unreasonable.

### 10.4 Evaluation of savings

Alternating days (or alternating 48 h blocks) between esther and the incumbent
control, with cost normalised by heating degree-hours. Reported with a
confidence interval; the winter is long enough for an answer.

---

## 11. Milestones

| | Deliverable | Exit |
|---|---|---|
| **M0** | Repo reset: legacy code removed (kept at `v0-legacy`), package skeleton, CI, house loader with validation, solver spike on a synthetic reference MILP | spike solves 144 steps in < 2 s |
| **M1** | Dataset builder + `io.homeostat` reader; plant characterisation report on real data (Appendix A hypotheses tested) | each hypothesis confirmed, refuted or re-stated |
| **M2** | Components: Zone, Mass, Coupling, Ambient, gains, Radiators; KF + MAP fit; synthetic recovery; envelope fit on real data with validation report | 24 h open-loop zone RMSE reported; structure chosen |
| **M3** | Tank components, draw detection, source components, reference-plant wrappers; fit on real data | draw detector validated on labelled draws |
| **M4** | Forecasters; planner (MILP) with all constraints; actuation translators; backtest harness | backtest over ≥ 4 weeks runs and scores |
| **M5** | Integration (outside esther): integration unit, executors with fallbacks; shadow | §10.3 exit, per actuator |
| **M6** | Cutover per actuator: supply temperature → tank mode → stove; alternating-day evaluation | owner's decision per actuator |
| **M7** | Scheduled re-fit with acceptance gate | first automatic re-fit accepted or refused with a report |

---

## 12. Open questions

1. **Modelling language.** cvxpy vs linopy for the MILP (M0 spike).
2. **Emitter law.** Linear `UA_rad` vs `UA (ΔT)^n`; decided in M2 on residuals.
3. **Zones.** One zone, or a zone per heat-source location (a stove heats one
   room hard). M2 compares; the description supports both.
4. **Ventilation and exhaust-air sources.** Whether an exhaust-air heat pump's
   source temperature can be taken as the zone temperature (it is the extract
   air) — probably, M1 checks.
5. **Draw labels.** How many labelled draws are needed to trust the detector.
6. **Evaluation length.** How many alternating blocks give a useful interval.

---

## Appendix A — Reference plant: exhaust-air heat pump with a jacket tank and a pellet stove

This appendix specifies the wrappers for the plant esther is first built
against. Facts carry their source; *hypotheses* are to be tested in M1.

### A.1 The heat pump: IVT 490 (controller Rego 401)

- Exhaust-air heat pump, fixed-speed piston compressor, R-134a.
  **Heat output 1.7–2.0 kW; electrical input 0.7–0.9 kW** including circulation
  pump and fan (installer manual, technical data). Compressor 0.5–0.7 kW. No
  COP is published; the figures imply 2.2–2.4. The manual expects the heat pump
  alone to carry a house to about +5 °C outdoors.
- **Source: the house's exhaust air**, 20–70 l/s at a fixed fan speed — so the
  source temperature is close to the zone temperature, not the outdoor
  temperature, and the COP depends mainly on the sink temperature (hypothesis
  H-A1).
- Compressor maximum supply about 58 °C; with electric heat 80 °C. When the
  tank bottom (GT3:3) exceeds 60 °C the compressor stops and only the electric
  heater continues.
- **Electric immersion heater**: 6, 9 or 12/13.5 kW variants; connected power and
  maximum are settings (factory 9 kW). Engaged by the controller on **timers**,
  not degree-minutes: heating timer 120 min, hot-water timer 30–120 min,
  disconnect timer 15 min; a fast-start rule below a set outdoor temperature
  (factory +5 °C). Step sizes are not documented. *Hypothesis H-A2:* the
  controller reports connected heater power in six equal steps (observed values
  are multiples of 4.62 in the serial field); the scale to kW is to be
  calibrated from the household load.
- **Sensors:** GT1 supply; GT2 outdoor; GT3:1 tank top (tap water); GT3:2 tank
  middle ("internal functions"); GT3:3 tank bottom, the heating water in the
  jacket; GT3:4 optional external; GT5 room (accessory); GT6 compressor hot gas.
- **Tank:** stainless double-jacket, 145 l inner (domestic hot water) and 55 l
  jacket (heating water). Hot water has priority: the controller closes the
  heating shunt until hot-water demand is met. Hot-water setpoint factory 52 °C,
  differential 5 K.
- **Heating circuit:** shunt valve SV1 mixing from the jacket to the supply.
  *Confirmed on data:* the supply GT1 is at or below the jacket GT3:3 in 96.6 %
  of minutes, a median 23.6 K below — the radiators are fed by mixing down from
  the jacket, so the jacket is the heating system's buffer and `Radiators.source
  = tank.jacket` is right.
- **Heating curve:** `T_supply = 20 + 0.16 · k · (20 − T_out) + parallel offset`,
  slope `k` 0–10 (fits the manual's four factory points exactly at k = 4;
  linearity in k assumed). Supply limits: start limiting 54 °C, maximum 56 °C
  (settings). Summer mode at a set outdoor temperature (factory 17 °C) with
  60 min / 360 min switching delays.
- **Anti-legionella** ("varmvattenspets"): a periodic raise, interval settable
  in days (factory 7). Temperature and duration not documented.

### A.2 How it is steered (the actuators that exist)

The installed interface emulates two of the controller's sensors:

- **Outdoor sensor (GT2)** → `CurveControlledHeatSource`. The interface accepts
  a supply-temperature target and computes the fake outdoor temperature
  through the inverse curve itself; while that target is valid the interface's
  own other corrections are bypassed. The value expires if not rewritten.
- **Tank middle sensor (GT3:2)** → `ThermostatChargedTank`. Three modes: pass
  the real value (auto), show 1 K above the upper limit (block: the controller
  sees a full tank and does not charge), show 1 K below the lower limit while
  the real value is below the upper limit (boost: the controller charges).
  The mode expires to auto if not rewritten.

*Hypothesis H-A3:* because the jacket is both the hot-water heat exchanger and
the heating buffer, **block and boost steer the compressor for space heating as
well as hot water.** If so, the two wrappers are one compressor seen through
two inputs, and `ivt490.Plant` must model them jointly: the compressor charges
the jacket when (tank demand) or (supply demand the jacket cannot meet), and the
two fake sensors decide which demands the controller sees. M1 tests this on
the recorded mode changes.

*Hypothesis H-A4:* the electric heater is triggered by the supply or tank
timers when the compressor cannot keep up. Since it is the expensive source
(COP 1 vs ~2.3), **avoiding it is the planner's main lever on cost in cold
weather,** and pre-charging the tank and the fabric before the timers would fire
is the mechanism. M1 measures how often it engages and on what.

### A.3 The stove: Aduro H2

- Convection stove, air heating only, no water jacket. Pellets with automatic
  feed and ignition; wood by hand (hybrid).
- **Pellets:** nominal 6.8 kW, reduced 3.4 kW, efficiency 87 %. Wood nominal
  7.0 kW, 81 %. Three pellet levels: 10 %, 50 %, 100 %. Per-level output is not
  documented; field reports give 50 % → 3.5 kW and 100 % → 7 kW; 10 % is a
  prior to be fitted.
- **Fuel rate:** maximum feed about 1.5 kg/h; hopper about 15 kg.
- **Start-up:** flames after about 6 min, normal operation after 15–20 min (at
  90 °C smoke temperature). Restart from standby only once the smoke
  temperature is below 100 °C — the `t_cool` minimum off time is fitted from
  smoke-temperature decay.
- **Wood detection:** at about 280 °C smoke temperature the stove itself stops
  pellet feed and switches to wood mode — the manual's own threshold, used for
  §7.5.
- Door open > 3 min shuts the pellet burner down; "FUEL" alarm after an hour of
  too-low smoke temperature. These are device behaviour and stay there.

### A.4 What the reference plant does not have

No heat-pump electricity meter: heat-pump power is the rated figure times the
measured compressor on-fraction, plus the heater's calibrated steps. No
hot-water flow meter: draws are detected (§6.3). A grid meter that measures net
import of a house with PV and a battery — not usable as consumption on its own.
