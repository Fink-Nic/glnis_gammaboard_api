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
command = ["apptainer", "exec", "--no-mount", "/etc/localtime", "--nv", "--bind", "$resources/../..:$resources/../..", "$resources/../../glnis_gammaboard_api/glnis.sif", "glnis-gammaboard-sampler"]
cwd = "$resources/.."
```

Adjust paths to the actual checkout layout. `--nv` exposes the host GPU stack;
use a GPU allocation when requesting CUDA execution. The [UBELIX guide](../../ops/ubelix/README.md)
covers image builds and cluster deployment.

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

Supply the license to the environment launching physics operations:

```bash
export SYMBOLICA_LICENSE='your-license-key'
```

See [mapping configuration](../configuration/mappings.md) and
[optional-backend development](../development.md#optional-backend-checks).

The files under [deployment templates](../../examples/templates/) predate the
current sampler configuration schema. Migrate their sampler arguments before
launching them; retired `sgn_strat_naive` and `sgn_strat_full` kinds are unavailable.
