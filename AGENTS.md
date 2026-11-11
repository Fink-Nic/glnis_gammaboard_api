# Working on this project

## Project map

- `src/gammaboard_sampler.py`: GammaBoard adapter, public domains, mapping
  corrections, density queries, snapshots, and stage handoff.
- `src/glnis/core/nn/orchestrator.py`: generation windows, optimization, runtime
  settings, and training-state persistence.
- `src/glnis/core/nn/sampler_module.py`: sampler-module interfaces and mixed flows.
- `src/glnis/core/nn/flow/`: active continuous/discrete flows, spline transforms,
  and subnet constructors. Use `flow/subnet_constructor.py`; do not introduce
  imports from the retired `core/nn/subnet_constructor.py` location.
- `src/glnis/core/mappings.py`: coordinate transforms and separate integration
  corrections; `parser.py` resolves explicit or GammaLoop graph properties.
- `src/glnis/core/config.py`, `sampler_aggregator.py`, and `integrand.py`: local
  run configuration, concurrent evaluation, and numerical integrands.
- `src/glnis/scripts/`: integration, comparison, and PDF-slice entry points.
- `native/gammaloop/`: optional Rust/PyO3 companion, built independently.
- `examples/local/`: current local-run examples; `ops/ubelix/`: HPC build tooling.

## Implementation conventions

- Preserve the separation between the public adapter, orchestrator, proposal
  module, and coordinate mappings. Put new behavior in the component that owns it.
- Select replaceable components through `BuildFromKind` and a unique `KIND`.
  Registries discover imported classes; ensure new implementations are imported
  before configuration construction. Do not add parallel string-dispatch systems.
- Keep configuration in the corresponding dataclasses. Preserve caller-owned
  configuration dictionaries, coordinate arrays, and feedback arrays.
- Public sample dimensions include singleton discrete axes. Internal proposal
  dimensions omit singleton axes and may add hidden mapping channels. Keep
  generation, density queries, and discrete constraints consistent about this.
- Feedback contains raw evaluator values. Preserve GammaBoard's stratum mask.
  Apply mapping corrections once; do not apply the proposal's importance weight
  again during feedback.
- Keep coordinate Jacobians, integration/partition weights, and probability
  densities distinct. `Mapping.forward`/`backward` return `(jacobian, continuous)`;
  `LayeredMappingResult` exposes `continuous`, `jacobian`, and `integration_weight`
  separately. An identity transform has unit Jacobians in both directions.
  Do not add separate density-transform aliases or density-mode flags.
- Exact restore retains optimizer/scheduler state, RNG state, unfinished training
  buffers, and pending mapping weights. Fresh stages retain compatible proposal
  parameters and reset training state. Keep these two operations distinct.
- Register PyTorch parameters and buffers appropriately, and respect configured
  device and precision without changing process-wide defaults as a side effect.
- Keep GammaLoop imports lazy. Core imports, numerical mappings with explicit
  graph properties, and packaging must work without the native companion or
  Symbolica. `glnis.core.nn` is an implicit namespace package: preserve packaging
  support for it.
- Do not revive retired `MadnisSampler` classes or sign-strategy names to make
  stale tests pass. Port relevant coverage to the current interfaces instead.

## Verification

Use Python 3.13 and the existing `.venv` when available. For a CPU check:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
```

Run the affected tests while developing; run the full suite for changes spanning
the sampler's contracts. The full command includes `ops/ubelix/tests`. Add tests
for observable numerical behavior and invariants, especially sample/probability
agreement after parameter updates, conditioning, singleton axes, fragmented
feedback windows, mapping factors, and recovery with outstanding feedback.

Use temporary paths for checkpoints and outputs. The manual Gaussian scripts
write `saved_state.pkl` and `snapshot.json` in their working directory, so run
them in a temporary directory rather than over existing project files.

Optional native/state tests may skip when the backend or required fixtures are
absent. Report those skips; do not hide failures with broader skips or unrelated
expectation changes. See [development instructions](docs/development.md).

## Documentation and scope

- Keep the top-level README restricted to project building and executable entry
  points, with links into [the documentation](docs/README.md).
- Put workflows in `docs/usage/`, option reference in `docs/configuration/`, and
  implementation contracts in `docs/implementation/`. Update the relevant pages
  when changing a configuration key or a public contract.
- Check defaults, supported kinds, shapes, and limitations against the code.
  Distinguish implemented functionality from proposed extensions.
- Link to runnable examples and existing source/tests rather than copying large
  configurations or maintaining duplicate platform guides. Keep the canonical
  UBELIX guide in `ops/ubelix/README.md`.
- Preserve existing user edits. Keep changes scoped to the request; documentation
  work should not change numerical behavior or configuration semantics.
