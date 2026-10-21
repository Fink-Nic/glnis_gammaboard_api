# glNIS GammaBoard API

MadNIS sampler implementation for GammaBoard using the
`gammaboard_process.run_sampler(...)` Python wrapper. Includes support for additional coordinate mappings schemes that allow the direct sampling of momentum space.

## Local sampling from TOML

Run a sampler and integrand locally with asynchronous evaluation:

```bash
.venv/bin/python tests/run_from_toml.py examples/local/ghostbumps.toml --history /tmp/ghostbumps-history.json
```

The test script defaults to `examples/local/ghostbumps.toml` when no path is
provided. It defaults OpenMP/MKL to one thread for the small CPU examples,
preserving any explicit environment settings. Other runnable examples are
`examples/local/sine.toml` and `examples/local/gaussian.toml`; the Gaussian uses
the existing spherical mapping to integrate over all of R³.

The same runner is available as `python -m glnis.scripts.run run.toml`, or
`glnis-integrate run.toml` after installing the updated package. The existing
`glnis-gammaboard-sampler` command continues to serve the GammaBoard protocol.

Each TOML has `[sampler]`, `[integrand]`, and `[aggregator]` tables. Sampler
settings use the existing MadNIS names, including nested `[sampler.flow_config]`,
`[sampler.transformer_config]`, and `[sampler.made_config]`. Dimensions are
derived from the integrand. Mapping configuration belongs under
`[sampler.mapping]`; graph data required by a mapping belongs under
`[sampler.graph_properties]`. Integrands do not depend on `GraphProperties`.
The mapping runs in the sampler, so its Jacobian is included exactly once in
the sample weights.

`sampler.kind` selects the MadNIS strategy (`default`, `sgn_strat_naive`, or
`sgn_strat_full`); the earlier `madnis` spelling remains an alias for `default`.
Stratified strategy implementations are still under development. The aggregator
uses `sampler.get_strat_cardinalities()` to configure every accumulator. When
that method returns a list, generation must return `(SampleBatch, strata)`:
integer labels with shape `(N,)` for one stratification dimension or `(N, K)`
for multiple dimensions. Strata remain separate from integrand discrete inputs
and follow the original samples through evaluation chunks and ordered feedback.
Before accumulation, `strat_channel_transform` receives the unweighted real,
imaginary, or magnitude projection selected by `aggregator.training_phase`.
Where it returns zero, both components of the complex contribution are zeroed.
The same selection applies during inference. A sampler returning `None` from
`get_strat_cardinalities()` continues to return a plain `SampleBatch`.

| Integrand kind | Settings |
| --- | --- |
| `gaussian` | Required `continuous_dims`; `sigma = 1.0` or a list of component widths; optional common/per-component `offset`; `const_f = false` |
| `sine` | `continuous_dims = 1`, integer `omega = 1`, `offset = 0.0` |
| `ghostbumps` | `centers = [[0.75, 0.75], [0.25, 0.25]]`, `sigma = 0.15`; dimensions follow centers |

Multiple Gaussian widths introduce a discrete component channel. The sum over
channels integrates to one on all real coordinates, using the original shared
normalization. A Gaussian on the unit cube has a different integral, so no
reference target is assumed. An explicit `[integrand.target]` table can supply
`real_mean`, `imag_mean`, and the existing `Result.from_kwargs` statistics.
Positive reference errors require `n_points >= 2` in that table.
Sine's default target equals its offset on the unit cube; use an explicit target
if changing its domain through a mapping.

| Aggregator setting | Default and meaning |
| --- | --- |
| `mode` | `"train_then_infer"`; also `"train"` and `"infer"` |
| `inference_samples` | `0`; additional samples to accumulate in inference |
| `generation_batch_size` | Omit to inherit `sampler.max_batch_size`; draws also stop at training-step boundaries |
| `evaluation_batch_size` | Omit to use the generation size; optionally split draws into smaller evaluation chunks |
| `evaluation_workers` | `2`; NumPy evaluation threads, independent of the serialized sampler thread |
| `max_in_flight_batches` | `4`; bounds outstanding draws, including results awaiting ordered feedback |
| `training_phase` | `"real"`; also `"imag"` and `"abs"` for weighted feedback |
| `log_every` | `10` completed training steps; `0` disables printing |

Training feedback preserves the exact generated sample order within and across
draws, even when chunks complete out of order. Every training step has a fresh
accumulator and one history record. Inference merges completed evaluation
chunks without reordering, extending one accumulator and recording every
update. Inference requires completed training or `sampler.training_steps = 0`.
Training starts at a step boundary; interrupted steps are not resumable through
the aggregator. Worker exceptions propagate and resources are closed; construct
a fresh run after a failure. Nonfinite evaluations are counted as failed samples
with zero contribution, preserving the accumulator's convention. Failures are
visible in the recorded observables and should be inspected when interpreting
an integral.

The Python API shares the same configuration path:

```python
from glnis.core.sampler_aggregator import SamplerAggregator

run = SamplerAggregator.from_toml("examples/local/ghostbumps.toml")
accumulator = run.run()  # In an event loop: await run.run_async()
training = run.training_history
inference = run.inference_history
run.run(mode="infer", inference_samples=1000)  # Extend the same accumulator.
```

`SamplerAggregator.from_config({...})` accepts the same three tables as a
dictionary. For preconstructed objects, use
`SamplerAggregator(sampler, integrand, config, accumulator=existing)` with an
optional `DefaultAccumulator` with matching stratification channels to continue
inference. Each history
record contains its phase, update number, sample count, elapsed wall time, and
an independent observable snapshot. Training records also contain the step and
loss; inference records identify the completed batch and sample slice. Raw
training samples are released after the step. Histories remain in memory;
`--history` writes both histories as JSON after the run, representing nonfinite
diagnostic values as `null`.

TOML filesystem paths such as `sampler.save_path` are resolved relative to the
configuration file. Unknown settings and invalid dimensions/batch sizes fail
validation. Custom integrands must support concurrent read-only evaluation and
return one unweighted real or complex value per sample.

### Sampler checkpoints

Set `sampler.save_path` and call `sampler.snapshot()` to save the sampler. The
returned metadata is JSON-compatible and points to one checkpoint file. For
`sgn_strat_full`, that file includes both child models and optimizers, their
configurations and training buffers, the parent training status, and the queue
of strata awaiting feedback. The default and naive strategies use the same
checkpoint interface. Writes replace the previous checkpoint atomically.

```python
from madnis_sampler import MadnisSampler

snapshot = sampler.snapshot()
restored = MadnisSampler.from_snapshot(
    snapshot=snapshot,
    continuous_dims=integrand.continuous_dims,
    discrete_cardinalities=integrand.discrete_cardinalities,
)
```

Pass the integrand's original dimensions, without a prepended sign channel.
New checkpoints contain the configuration and mapping metadata, so `init_args`
is optional; legacy checkpoints still accept their original `init_args`.
Recovery restores CPU and used CUDA random generators after reconstructing all
models, preserving continuation on the same device setup. A checkpoint taken
with evaluations outstanding retains that training state; supply their feedback
in the original generation order before advancing past the training window.
Sampler recovery supports partial windows even though a fresh `SamplerAggregator`
currently requires training to start at a step boundary.

The aggregator exposes the same services through `aggregator.snapshot(path)`
and `SamplerAggregator.from_snapshot(metadata, run_config)`. Passing a path to
`snapshot` overrides `sampler.save_path` for that call only. PDF evaluation is
available as `aggregator.pdf(discrete, continuous, batch_size=None)`, without
generating samples or changing histories. Its coordinate dimensions are exposed
as `pdf_continuous_dims` and `pdf_discrete_cardinalities`. Discrete PDF inputs
include sign strata and mapping channels in the sampler's PDF input order;
continuous inputs use the sampler's PDF coordinate space. Calls are batched and
preserve input arrays. PDF evaluation and checkpoint creation require the
aggregator to be idle.

## Comparing sampler configurations

Run the comparison script with a TOML containing named sampler settings:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m glnis.scripts.sampler_comparison examples/local/sampler_comparison.toml
```

After installing the updated package, the equivalent command is
`glnis-compare-samplers examples/local/sampler_comparison.toml`. The thread
settings above keep the small CPU example efficient; select appropriate values
for larger runs.

The comparison shares `[integrand]` and `[aggregator]` across all runs. An
optional `[sampler]` supplies defaults; each `[[samplers]]` entry has a unique
`name` followed by overrides using the usual sampler configuration keys.
Nested tables are merged, so overriding `flow_config.units` preserves the
other flow settings. Lists, including mapping layer lists, are replaced.
For example:

```toml
[integrand]
kind = "ghostbumps"

[aggregator]
inference_samples = 100_000

[sampler]
seed = 42
training_steps = 100
training_batch_size = 1024

[[samplers]]
name = "small-flow"
flow_config = { layers = 3, units = 32 }

[[samplers]]
name = "large-flow"
flow_config = { layers = 5, units = 128 }
```

Each entry constructs a fresh `SamplerAggregator`, completes its training and
inference, and releases the sampler before starting the next entry. Training
budgets may differ between entries; the plots use cumulative training sample
counts. Shared defaults also apply to seeds, so every run uses the same seed
unless its entry overrides it. There are no preliminary or independent
checkpoint evaluation draws: all training curves come from the recorded
training-step samples, which describe the distribution before that step's
optimizer update. No discrete-probability probing is performed.

`[comparison]` accepts `run_name`, `output_dir`, `comment`, `save_data` (default
`true`), `save_checkpoints` (default `false`), and `plot` (default `true`). `[comparison.plotting]` accepts `dpi`
(default `150`) and `figsize` (default `[10, 8]`). Filesystem paths are relative
to the TOML file. By default, output goes to `outputs/sampler_comparison` beside
that file; the supplied example instead selects the repository's output folder.
The runner saves a timestamped `SamplerCompData` pickle, a text summary, and PNGs
for training progression and final inference results. Empty training/inference
phases produce no corresponding PNG. `--no-plot` skips PNGs; `--no-output`
suppresses all files. `save_data = false` suppresses only the pickle.

The existing `TrainingProgress` arrays are populated once per completed training
step. Mean/error/RSD/TVAR use the selected real, imaginary, or complex-magnitude
training phase; ARSD uses its absolute values. Time variance uses accumulated
processing time, not elapsed wall time. `SamplerCompData.result`, `observables`,
and `training_progress` are keyed by configured names. `run_configs` stores each
resolved configuration, and `training_histories`/`inference_histories` retain the
complete observable histories. The comparison pickle is written once after all
runs finish. `run_sampler_comp()` returns this container and exposes saved paths
through `artifact_paths`.

With `save_checkpoints = true`, each sampler is checkpointed after its training
and inference finish, before constructing the next sampler. Checkpoints use
unique indexed filenames in a directory beside the comparison pickle. Their
metadata is stored in `SamplerCompData.integrator_states`; restore a named run
with `data.restore_aggregator("naive", use_gpu=False)`. Restored aggregators have
fresh accumulators/histories; the comparison's recorded results stay on `data`.
Keep the pickle and its checkpoint directory together when moving them. Relative
checkpoint references resolve against the loaded comparison file's location.
`--no-output` suppresses checkpoints too; `save_data = false` still permits
checkpoint creation and restoration through the returned in-memory object.

Replot a saved comparison without generating samples:

```bash
.venv/bin/python -m glnis.scripts.sampler_comparison /path/to/comparison.pkl --replot
```

Pickle files should only be loaded from trusted sources. Results saved through
the module/CLI use stable class import paths and can also be loaded with
`glnis.scripts.sampler_comparison.load_comparison(path)`.

## PDF slice plots from comparison checkpoints

`examples/local/sampler_comparison_slices.toml` is a small CPU example with
default, naive sign-stratified, and full sign-stratified samplers, checkpoint
creation enabled, and 1D/2D definitions under `[slice_plot]`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m glnis.scripts.sampler_comparison examples/local/sampler_comparison_slices.toml
.venv/bin/python -m glnis.scripts.slice_plot /path/to/saved_comparison.pkl
```

The slice runner uses the `[slice_plot]` settings saved in the comparison.
`--settings run.toml` replaces them with another TOML's table. It also accepts a
standalone TOML containing `slice_plot.comparison_file`:

```toml
[slice_plot]
comparison_file = "saved_comparison.pkl"
output_dir = "slices"
normalize = true
use_gpu = false
batch_size = 4096

[slice_plot.plotting]
plot_singles = true
plot_comparison = true
dpi = 150
cmap_name = "plasma"
table_precision = 3

[[slice_plot.slices]]
name = "positive-plane"
origin = [0.5, 0.5]
discrete = [0]
dirs = [0, 1]
grid = [{start = 0.0, stop = 1.0, num = 64}]
```

`origin` supplies the continuous dimension; `discrete` supplies all fixed PDF
channel indices (default `[]`). Sign channels come first, so `[0]` and `[1]`
select the positive and negative strata when there are no other channels.
Each entry in `dirs` is an axis index or a direction vector. Integer axes use the
grid value as that coordinate, replacing the corresponding origin coordinate;
vectors use `origin + u * direction`. Two directions form an affine plane.
`grid` uses `LinSpace` fields `start`, `stop`, and `num`; one grid is reused for
both directions, or supply two grids. 2D arrays have shape `(v.num, u.num)`.

Compatibility is checked separately for every restored sampler. Dimension,
direction, or cardinality mismatches are printed and recorded in
`SlicePlotData.skipped`, while compatible samplers continue. Slices use the
sampler's returned densities without changing joint/conditional normalization.
Function overlays use the configured training phase and sign-channel filtering.
Sampling and training are never called.

By default `domain = "auto"` masks unit-cube boundary points using `EPS = 1e-6`
and permits unbounded coordinates for momentum-space mappings; explicit
`"unit_cube"` and `"unbounded"` overrides are available. Masked points remain NaN
in saved arrays. `normalize = true` uses a positive absolute-integral reference
or comparison estimate when available; `normalization = 2.0` supplies an explicit
divisor, and `normalize = false` keeps raw function values. PDFs are unchanged.
No integral or variance estimate is inferred from the plotting grid.

Output contains a `SlicePlotData` pickle plus PNGs. `save_data = false`,
`plot = false`, `--no-output`, and `--no-plot` control artifacts. Paths in TOML
are relative to that file; `--output-dir` overrides the destination. Use
`python -m glnis.scripts.slice_plot saved_slices.pkl --replot` to render stored
arrays without accessing sampler checkpoints. The plural `slice_plots` module
and installed `glnis-slice-plot` command expose the same runner.

## Runtime Options

Supply your own Symbolica license in the environment used to launch GammaBoard
or the sampler, including for editable development and Apptainer:

```bash
export SYMBOLICA_LICENSE='your-license-key'
```

### Nix

Python is pinned to the same Nixpkgs revision as GammaBoard: Python 3.13.12 with
the matching PyTorch/CUDA 12 stack. Update both repositories' pins together.
Rust uses a separate Fenix toolchain because Symbolica 3.0.1 requires Rust 1.96
or newer. Rerun `glnis-dev` after changing the pin to refresh an editable venv.

Build the project environment:

```bash
nix build .#runtime
```

This builds a locked runtime with the Rust dependency artifacts cached separately
from the Python package. `nix run` also
launches the sampler, and `nix build .#wheel` builds the Nix-native Python wheel.

Use this GammaBoard process command:

```toml
command = ["$resources/../../glnis_gammaboard_api/result/bin/python", "-u", "-m", "run_sampler"]
cwd = "$resources/.."
```

### Apptainer

On UBELIX, clone this repository and run `python3 ubelix.py` to submit a Slurm
build of the sampler image. See [the UBELIX guide](ops/ubelix/README.md) for
resource options, build monitoring, and using the image with GammaBoard.

Apptainer/SIF is the portable runtime path for HPC systems that do not provide
Nix. The definition uses Nix only inside the image build container to reproduce
`nix build .#runtime`; the final SIF exposes normal executables and does not
require Nix on the host.

Build from this directory so the `%files` entries in `apptainer.def` resolve to
the local checkout:

```bash
apptainer build --force glnis.sif apptainer.def
```

Use this GammaBoard process command:

```toml
command = ["apptainer", "exec", "--no-mount", "/etc/localtime", "--nv", "--bind", "$resources/../..:$resources/../..", "$resources/../../glnis_gammaboard_api/glnis.sif", "glnis-gammaboard-sampler"]
cwd = "$resources/.."
```

### Editable development

Enter the development shell and install the checkout in editable mode:

```bash
nix develop
glnis-dev
source .venv/bin/activate
```

The shell provides Python 3.13, Rust, and native dependencies without building
the runtime. `glnis-dev [venv-directory]` creates or updates the venv and records
the selected native library/compiler paths so its sampler can also run outside
the shell. Python edits take effect immediately. Rerun `glnis-dev` after Rust or
dependency changes; Cargo reuses the local `target` directory. Editable builds
use the development profile; packaged Nix builds use the release profile.

After updating `flake.lock`, rerun `glnis-dev`. It also refreshes the venv's base
interpreter in place when the Nix Python changes, keeping Python, glibc, and the
Rust extension compatible while preserving installed packages.

An ordinary editable install is also supported inside the shell, or on a Linux
host with Python 3.13, Rust, a C/C++ compiler, and the required native libraries:

```bash
uv venv --python python3.13 .venv
uv pip install --python .venv/bin/python -e .
```

On Nix hosts, use `glnis-dev` when the venv must run outside `nix develop`.

Use this GammaBoard process command:

```toml
command = ["$resources/../../glnis_gammaboard_api/.venv/bin/glnis-gammaboard-sampler"]
cwd = "$resources/.."
```

With `cwd = "$resources/.."`, sampler `save_path` values should be relative to
the GammaBoard workspace, for example:

```toml
save_path = "integrations/glnis_gammaboard_api/checkpoints/ghost_bump_madnis"
```

### Updating dependencies

Change a Rust dependency version or Git revision in `Cargo.toml`, then update the
lockfile using Cargo, for example:

```bash
nix develop
cargo update -p gammalooprs
glnis-dev                 # incremental editable build
# Or build the deployable runtime:
nix build .#runtime
```

Commit `Cargo.toml` and `Cargo.lock` together. Maturin and Nix both build with
`--locked`. The GammaLoop revision is also used for its embedded assets, with no
separate source hash to maintain. Cargo uses this project's lockfile, not the
lockfile inside a Git dependency. GammaLoop `2e22d46` uses Rust Symbolica and
Numerica 3.0.1, as recorded in `Cargo.lock`.

Use `nix flake update <input-name>` to update a Nix input and commit `flake.lock`.
The editable Python dependencies use the sources declared in `pyproject.toml`;
the Nix runtime uses the source revisions in `flake.lock` and Nixpkgs packages.
The Python Symbolica wheel remains pinned to 3.0.0. When updating the Python
wheel, update its version in `pyproject.toml`
and the version, Python ABI tag, and checksum in `flake.nix` using the release
files published on PyPI.

Changing the Rust dependency set can rebuild Nix's dependency derivation. Local
editable builds retain Cargo's incremental cache across dependency updates.


## Use With GammaBoard

Run templates live in `examples/templates/runs/`. Load them directly from the
GammaBoard checkout, for example:

```bash
./gammaboard run create ../glnis_gammaboard_api/examples/templates/runs/ghost_bump_madnis.toml
```

Campaigns resolve their child template relative to the campaign file. Keep the
base `madnis_gammaloop_train_sample.toml` beside the campaign when copying files.
The templates launch the built Nix runtime by default.

Training feedback follows `SampleBatch.training_remaining`; these templates do
not need `requires_training_values`. Keep the training task's `max_samples`
equal to `training_steps * training_batch_size` so inference restores a fully
trained snapshot. Bulk generation is now automatic; `bulk_sample_generation`
is retired.

The physics templates expose independent `max_batch_size` (32,768 evaluator
samples) and `max_generation_size` (262,144 samples per sampler draw) replacements.
Training draws also stop at the minibatch boundary. Leave evaluator batch sizing
adaptive; try `sampler_aggregator_runner_params.io_threads = 2` or `4` only when
transport is limiting throughput and those CPU threads are available. The default
is one I/O thread per active sampler, so account for concurrent campaign children.

For GammaLoop x-space PDF images, every coordinate must be strictly inside
`(0, 1)`, including coordinates held fixed by a slice. The task template
`examples/templates/tasks/gg_hhh_1L_x_space_pdf.toml` supplies three interior
128-by-128 slices with the fixed coordinate at 0.5. Physical-momentum runs use
the separate slices and spherical mapping in `gg_hhh_1L.toml`.

The sampler command uses `$resources/..` because GammaBoard expands
`$resources` to the default resource directory. Sampler `args` are passed
through unchanged, so `save_path` should be relative to the configured process
`cwd`.

The process entrypoint is:

```bash
python -u -m run_sampler
```
