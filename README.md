# esther

esther plans the heating of a house: when to run which heat source, how hard,
and how to use the house's own thermal storage — the building fabric and the
hot-water tank — to buy energy when it is cheap, while keeping the people in it
comfortable and the hot water hot.

It does that with economic model predictive control over a grey-box thermal
model that learns its parameters from the house's own history.

**v1 is being rewritten.** The specification is [`docs/spec.md`](docs/spec.md);
it is the contract for the build. The v0 line (single-state model, SLSQP, MQTT
service) is preserved at tag [`v0-legacy`](../../tree/v0-legacy).

esther is a library. It does no I/O of its own: an integration layer feeds it
measurements and forecasts and carries out its plans.

## Status

Milestones M0–M4 (spec §11): package skeleton, CI, house loader with
validation; the identification dataset builder (`esther.data`) with an
optional reader for a homeostat history store (`pip install
'esther[homeostat]'`, `esther.io.homeostat`); the model and its
identification (`esther.core.model`, `esther.identify`): composition into a
linear state-space model, a batched continuous-discrete Kalman filter, MAP
fit over physical priors with uncertainty and identifiability, held-out
validation with the acceptance gate, draw detection, synthetic recovery
tests; and planning (M4): forecasters that learn from the house's own history
(`esther.forecast`), the MILP planner over every component's contribution
(`esther.plan`), actuation translators (`esther.actuate`) and a backtest
harness with a digital twin and a leak check (`esther.backtest`). Nothing
commands a real house yet: integration and executors are M5, outside esther.

```python
from esther import load_house
from esther.core.model import compose
from esther.identify import IdData, fit, validate, acceptance_gate, report_markdown

model = compose(load_house("house.toml"))  # the components that have dynamics
data = IdData.from_dataset(dataset, model.required_signals())
train, test = data.split(0.2)  # the last 20 % by time
res = fit(model, train)  # MAP, ≥ 8 starts
val = validate(model, res, test)  # open-loop 1/6/24 h vs persistence
print(report_markdown(res, val, acceptance_gate(res, val)))
```

Planning, given a fit's values and forecasts made at `now`:

```python
from esther.forecast import combine
from esther.plan import Planner
from esther.actuate import translate

planner = Planner(house, dict(zip(res.names, res.theta)), accepted=gate.accepted)
forecast = combine([f.forecast(now, house.horizon, house.step) for f in forecasters])
plan = planner.plan(now, state_estimate, forecast, previous=last_plan)
commands = translate(plan, house)  # advisory plans give commands marked not actionable
```

Set `OPENBLAS_NUM_THREADS=1` when fitting with `FitOptions(workers=n)`: the
filter's matrices are tiny and BLAS threads per process only compete.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,spike]'
pytest
ruff check .
python benchmarks/reference_milp.py     # solver spike, prints timings
```

## License

Apache 2.0, see [LICENSE](LICENSE).
