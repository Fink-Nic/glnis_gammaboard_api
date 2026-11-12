# Running with GammaBoard

Build or install a worker using the [project README](../../README.md). The worker
entry point is `glnis-gammaboard-sampler`, also available as `python -u -m run_sampler`.
It speaks the GammaBoard process protocol; it is not an interactive integration CLI.

## Select a worker command

For a sibling checkout named `glnis_gammaboard_api`, examples of GammaBoard process
launch settings are:

```toml
command = ["$resources/../../glnis_gammaboard_api/.venv/bin/glnis-gammaboard-sampler"]
cwd = "$resources/.."
```

For a Nix runtime built into `result`:

```toml
command = ["$resources/../../glnis_gammaboard_api/result/bin/python", "-u", "-m", "run_sampler"]
cwd = "$resources/.."
```

For the Apptainer image:

```toml
command = ["apptainer", "exec", "--cleanenv", "--nv", "--bind", "$resources/../..:$resources/../..", "$resources/../../glnis_gammaboard_api/glnis.sif", "glnis-gammaboard-sampler"]
cwd = "$resources/.."
```

Adjust paths to the actual checkout layout. `--nv` exposes the host GPU driver
and devices; the image supplies PyTorch's CUDA runtime. Use a GPU allocation when
requesting CUDA execution. With `--cleanenv`, forward the allocation's mask in the
worker job: `export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"`.
Forward `CUDA_DEVICE_ORDER` the same way when the job sets it.
Forward `SYMBOLICA_LICENSE` using `APPTAINERENV_SYMBOLICA_LICENSE` only when the
optional native backend is needed. The [UBELIX guide](../../ops/ubelix/README.md)
covers these settings, compiler caches, and storage bindings.

## Supply process arguments

GammaBoard supplies `domain` and `evaluator_metadata`. The process `args` object
contains arguments accepted by `GammaBoardSampler`, for example this JSON object:

```json
{
  "save_path": "checkpoints/sampler.pt",
  "orchestrator": {
    "kind": "rectangular",
    "training_steps": 10,
    "training_batch_size": 128,
    "loss": {"kind": "kl_divergence"}
  },
  "module": {"kind": "mixed flow"}
}
```

There is no local-run `[sampler]` wrapper around this object. See the
[configuration overview](../configuration/README.md) for the two nesting conventions.
Process arguments pass through directly: relative paths such as `save_path` are
relative to the configured worker `cwd`, rather than automatically relative to
a local TOML file.

## Training and recovery

Each draw reports `training_remaining` while training. GammaBoard then supplies
one raw feedback value per sample, with optional requested/actual stratum
assignments. The sampler applies mapping corrections internally. Do not enable
an additional evaluator-side importance-weight correction for training.

A completed training window advances the optimizer. Generation reports `WAITING`
when a full window is outstanding, and inference draws stop requesting feedback
once training completes. See [sampling and feedback](../implementation/sampling-and-feedback.md).

GammaBoard uses `from_snapshot()` for recovery and `from_stage()` for stage
transfer. Their different state and override rules are documented in
[checkpoints and stages](../implementation/checkpoints-and-stages.md).

## GammaLoop inputs

A mapping can obtain graph properties from GammaLoop evaluator metadata with
`kind="gammaloop"`, `state_folder`, `process_id`, and `integrand_name`. State parsing
requires the optional companion and symbolic dependencies. Explicit
`graph_properties` avoid state parsing. Configure coordinate mappings explicitly;
evaluator metadata alone does not enable a mapping.
Saved states must match the GammaLoop version used by GammaBoard. A state-format
mismatch requires regenerating that state with the matching GammaLoop build.

Supply the license to the environment launching physics operations:

```bash
export SYMBOLICA_LICENSE='your-license-key'
```

See [mapping configuration](../configuration/mappings.md) and
[optional-backend development](../development.md#optional-backend-checks).

## Run and task cards

The [cards](../../examples/templates/) are adapted from the sibling
`gammaboard_os_ct/resources/templates` installation and use its current
`task_queue.sampler.config` / `accumulator={config=...}` definitions:

- [GammaLoop baseline](../../examples/templates/runs/gammaloop_train_sample.toml):
  training and inference for the MSbar ttH NLO GL30 group, selected as graph group
  8 in `NLO_MSbar`. It exposes training/device/precision settings, losses,
  schedulers, continuous/discrete networks, splines, execution policy, and queue
  sizes. Adjust state paths and graph selections for your generated state.
- [OS GL02](../../examples/templates/runs/madnis_NLO_GL02_OS.toml) and
  [sign-stratified OS GL02](../../examples/templates/runs/madnis_NLO_GL02_OS_sgn_stratified.toml):
  graph group 10 in `NLO_OS`, retaining the supplied real target and Voronoi
  partition. Stratification belongs to GammaBoard's process-sampler configuration.
- [NLO campaign](../../examples/templates/runs/epem_a_ttxh_NLO_campaign.toml):
  eleven graph groups with the supplied MSbar targets. Children use the expanded
  baseline, and graph group is included in run/checkpoint names.

Every process command assumes the installation is named `glnis_gammaboard_api`,
as a sibling of GammaBoard. State paths are interpreted by GammaBoard relative
to its resource root; sampler checkpoint paths are relative to process `cwd`.
Training excludes its estimates from published results. Inference restores the
completed training sampler with `from_name` and starts a fresh accumulator.
Keep `train_samples = train_steps * train_batch_size` when editing replacements.

GammaBoard expands an entire `$(name:default)` placeholder as a typed value.
The baseline's `loss`, `scheduler`, and `execution` replacements are complete
configuration values, so `scheduler=false` and `execution={kind="eager"}` work
without leaving incompatible suboptions behind. The commented compiled execution
table exposes both methods' graph, shape, and mode options. Optional mapping
configuration is shown separately; a hypercube evaluator needs no extra mapping.

Validate a card with `gammaboard run validate CARD`, then submit it with
`gammaboard run create CARD`. Validation reads the referenced GammaLoop state;
`--probe` additionally starts runtimes and requires the configured worker command
to exist. It is not necessary to create database records to validate a card.

The [task cards](../../examples/templates/tasks/) come from the same installation:
scalar Monte Carlo, native Havana four-phase training/inference, and two PDF
diagnostics. Append with `gammaboard run task append RUN CARD`. The PDF examples
target a two-dimensional domain and a source task named `inference`; adjust their
geometry/discrete coordinates and source name for another run (the flow baseline
names its inference task `sample`). Task names must be unique in the destination
queue. The Havana append card intentionally uses GammaBoard's native sampler.

## Compilation and the process deadline

The referenced GammaBoard installation hardcodes **300 seconds per process
request**, including initialization, generation, feedback, and checkpoint queries.
It has no run-card timeout setting. `shutdown_grace_seconds` controls shutdown
and does not extend this deadline. The GammaBoard checkout is unchanged here.

The sampler creates compiled callables during initialization but compiles lazily
on first generation/probability use. A new shape or the transition to inference
can trigger more work. The baseline defaults to eager execution; enable its
compiled table when these requests fit within the deadline. Reusing a persistent
`TORCHINDUCTOR_CACHE_DIR` in the worker-launch environment reduces recompilation
cost, but does not remove first-use startup. Warm the cache with the actual
model, device, precision and batch shapes. If cold compilation exceeds five
minutes, use eager execution or change GammaBoard's request-timeout implementation
before enabling compilation for that workload. A 1,800-second configurable
request window would need an upstream change; it cannot be enabled in these cards.
See the [native measurements](../implementation/compilation-native-6d.md) for
startup, shape changes, and recovery costs.
