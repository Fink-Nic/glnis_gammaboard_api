# Local run and analysis options

Local runs have required `[sampler]` and `[integrand]` tables and an optional
`[aggregator]`. See [sampler configuration](sampler.md) for sampler options.
Sources: [run configuration](../../src/glnis/core/config.py),
[integrands](../../src/glnis/core/integrand.py),
[comparison configuration](../../src/glnis/scripts/comparison_config.py), and
[slice configuration](../../src/glnis/scripts/slice_plots.py).

## Integrands

The local factory's `kind` values are `gaussian`, `sine`, and `ghostbumps`.
These names differ from the descriptive `KIND` attributes on the integrand
classes. All integrands return one unweighted real or complex value per sample;
their discrete axes use counting measure.

| `kind` | Option | Default | Meaning / constraint |
| --- | --- | --- | --- |
| `gaussian` | `continuous_dims` | required | Positive dimension |
| `gaussian` | `sigma` | `1.0` | Positive width or nonempty list of positive widths |
| `gaussian` | `offset` | zero vectors | Shared vector or one vector per component |
| `gaussian` | `const_f` | `false` | Return the constant 1 instead of Gaussian values |
| `sine` | `continuous_dims` | `1` | Positive dimension |
| `sine` | `omega` | `1` | Integer frequency |
| `sine` | `offset` | `0.0` | Constant offset |
| `ghostbumps` | `centers` | `[[0.75, 0.75], [0.25, 0.25]]` | Nonempty matrix; columns determine continuous dimension |
| `ghostbumps` | `sigma` | `0.15` | Positive width |

Multiple Gaussian widths add one discrete component axis. Components share the
sum of their unnormalized integrals as normalization, so the total integral is
one over all real coordinates. A unit-cube Gaussian run has no automatic unit
target. Sine's unit-cube reference integral is its offset.

Every integrand accepts an optional `[integrand.target]` table with `n_points`,
`total_time`, and `{real,imag,abs_real,abs_imag}_{mean,error}` values. Errors and
time are nonnegative; positive errors require `n_points >= 2`. A supplied target
is a reference for reporting, not an instruction to normalize evaluator output.

## Aggregator

| Option under `aggregator` | Default | Meaning / constraint |
| --- | --- | --- |
| `mode` | `train_then_infer` | `train`, `infer`, or `train_then_infer` |
| `inference_samples` | `0` | Nonnegative inference budget |
| `generation_batch_size` | inherit orchestrator `max_batch_size` | Positive number of samples requested per draw |
| `evaluation_batch_size` | generation size | Positive number of samples per evaluator chunk |
| `evaluation_workers` | `2` | Positive worker-thread count |
| `max_in_flight_batches` | `4` | Positive bound on outstanding draws |
| `log_every` | `10` | Completed-step reporting interval; zero disables printing |
| `training_phase` | `real` | Train on `real`, `imag`, or complex magnitude `abs` |

Generation also stops at training-window boundaries. Inference requires a
completed proposal, or zero configured training steps for a fresh proposal.
Training starts at a step boundary; the sampler can recover partial windows,
but a fresh local aggregator does not resume their partial statistics.

Evaluation chunks may finish out of order. Training feedback is restored to
sample/draw order; inference statistics are updated as chunks complete. Failed
or nonfinite evaluations remain in the sample count with zero contributions
and are exposed by failure observables. Worker failures propagate; construct a
fresh local run after an execution failure.

## Comparisons

A comparison has one shared integrand and aggregator, optional shared sampler
defaults, and a nonempty `[[samplers]]` list. Each entry requires a unique,
nonempty `name`, followed by sampler overrides in the same schema as `[sampler]`.
Names are display labels, not checkpoint filenames.

| Option under `comparison` | Default | Meaning |
| --- | --- | --- |
| `run_name` | `sampler_comparison` | Nonempty filename component, without path separators |
| `output_dir` | `outputs/sampler_comparison` | Output directory relative to the settings file |
| `comment` | empty string | Free-text comparison annotation |
| `save_data` | `true` | Save the comparison pickle |
| `save_checkpoints` | `false` | Save trained proposals for later restoration/slices |
| `plot` | `true` | Generate comparison figures |

`comparison.plotting` accepts `dpi=150` and `figsize=[10.0, 8.0]`, with positive
finite sizes. A fresh comparison in `aggregator.mode="infer"` requires each
sampler's `orchestrator.training_steps=0`.

Nested sampler tables merge recursively; lists replace their defaults. Every
named configuration is parsed before executing the first run. See the
[current example](../../examples/local/sampler_comparison.toml) and
[analysis workflow](../usage/analysis.md).

## PDF slices

Slice settings live under `[slice_plot]`, either in a comparison file or in a
separate settings file. At least one `[[slice_plot.slices]]` entry is required.

| Option under `slice_plot` | Default | Meaning / constraint |
| --- | --- | --- |
| `comparison_file` | unspecified | Saved comparison input, required when not supplied through the CLI |
| `output_dir` | derived from comparison output | Optional output directory |
| `run_name` | `slice_plot` | Filename component |
| `EPS` | `0.000001` | Unit-cube boundary exclusion, in `[0, 0.5)` |
| `domain` | `auto` | `auto`, `unit_cube`, or `unbounded` |
| `normalize` | `true` | Normalize the selected integrand overlay when a normalization is available |
| `normalization` | inferred when possible | Explicit positive finite overlay normalization |
| `batch_size` | aggregator PDF default | Positive PDF query size |
| `use_gpu` | saved proposal preference | Optional boolean device override |
| `save_data` | `true` | Save evaluated slice arrays |
| `plot` | `true` | Generate figures |

`auto` uses the output mapping space, or a unit-cube domain when no mapping is
configured. Normalization affects the integrand overlay once; it does not rescale
the sampler's PDF. Without an explicit normalization, the code looks for a
positive absolute-integral reference among the target/results.

| Option per slice | Default | Meaning |
| --- | --- | --- |
| `name` | `slice-<index>` | Unique nonempty label |
| `origin` | required | Finite vector in the public continuous domain |
| `discrete` | `[]` | Fixed public discrete coordinates; excludes hidden mapping channels |
| `dirs` | `[0]` | One or two axis indices or nonzero direction vectors |
| `grid` | `[{start=0.0, stop=1.0, num=1000}]` | One grid reused for all directions, or one grid per direction |

Each grid requires `stop > start` and integer `num >= 2`. Explicit direction
vectors must match the origin width; two vectors must be linearly independent.
An axis-index direction replaces that origin component with the grid coordinate.
Vector directions instead add displacement to the origin. Incompatible slices
are recorded in the output's `skipped` mapping.

`slice_plot.plotting` accepts:

| Option | Default | Meaning |
| --- | --- | --- |
| `plot_singles` | `true` | Figures for individual proposals |
| `plot_comparison` | `true` | Figures comparing proposals |
| `dpi` | `150` | Positive image DPI |
| `cmap_name` | `plasma` | Installed Matplotlib colormap |
| `table_precision` | `3` | Nonnegative precision for displayed coordinate metadata |

See the [analysis workflow](../usage/analysis.md) for CLI commands and replotting.
