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

Milestones M0 and M1 (spec §11): package skeleton, CI, house loader with
validation, a solver spike on a synthetic reference MILP, and the
identification dataset builder (`esther.data`) with an optional reader for a
homeostat history store (`pip install 'esther[homeostat]'`, `esther.io.homeostat`).
Nothing plans a real house yet.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
python benchmarks/reference_milp.py     # solver spike, prints timings
```

## License

Apache 2.0, see [LICENSE](LICENSE).
