# Comparisons and PDF slices

## Compare named proposals

Start from the [current comparison example](../../examples/local/sampler_comparison.toml):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 glnis-compare-samplers examples/local/sampler_comparison.toml
```

Every `[[samplers]]` entry overrides the shared `[sampler]` table recursively.
Runs execute sequentially with fresh proposals. Lists replace earlier lists;
shared nested settings remain unless overridden. Configure losses and schedulers
under each run's `orchestrator` table, and proposal architecture under its
`sampler` table. See the [analysis option reference](../configuration/local-runs.md#comparisons).

Outputs can include a comparison pickle, text summary, and PNGs. Training curves
use the recorded training steps; the comparison does not generate extra
checkpoint-evaluation draws. Useful CLI switches are `--no-plot`, `--no-output`,
and `--comment "..."`. Replot saved results without sampling:

```bash
glnis-compare-samplers path/to/comparison.pkl --replot
```

Set `comparison.save_checkpoints=true` before running if later PDF queries or
slices will need the trained proposals. `save_data=false` disables the comparison
pickle; `--no-output` suppresses all output, including checkpoints.

## Restore a comparison proposal

```python
from glnis.scripts.sampler_comparison import load_comparison

comparison = load_comparison("path/to/comparison.pkl")
run = comparison.restore_aggregator("small-flow", use_gpu=False)
```

Recorded checkpoint paths are relative to the comparison file. Moving the whole
output directory preserves restoration. A restored aggregator starts with empty
histories and accumulator; the recorded comparison statistics remain on
`comparison`. Load only trusted comparison and checkpoint files, since their
serialization uses pickle/PyTorch.

## Evaluate a slice

Create a TOML settings file pointing to a saved comparison. This fragment is a
complete slice settings file; replace the comparison path with your result:

```toml
[slice_plot]
comparison_file = "results/comparison.pkl"
normalize = false

[[slice_plot.slices]]
name = "line"
origin = [0.5, 0.3]
discrete = []
dirs = [0]
grid = [{ start = 0.1, stop = 0.9, num = 100 }]
```

Then run:

```bash
glnis-slice-plot slices.toml
glnis-slice-plot path/to/comparison.pkl --settings slices.toml
```

The origin and discrete vector must match the proposal's public PDF domain.
A direction is an axis index or an explicit vector. Supply one direction for a
line or two independent directions for a plane. Hidden mapping channels are
summed by the sampler and are absent from the slice's discrete vector.

Mapped physical coordinates can lie outside the unit cube. The default `auto`
domain mask inspects the output mapping space; `unit_cube` excludes boundary
coordinates according to `EPS`. Incompatible slices are reported and skipped.
See [slice settings](../configuration/local-runs.md#pdf-slices) for normalization,
grids, and plotting controls.

Saved slice data contain evaluated arrays. Replot them without accessing the
proposal checkpoints:

```bash
glnis-slice-plot path/to/slices.pkl --replot
```

The Python entry points are `run_sampler_comp`, `load_comparison`,
`run_slice_plots`, and `load_slice_data` in
[sampler comparison](../../src/glnis/scripts/sampler_comparison.py) and
[slice plotting](../../src/glnis/scripts/slice_plots.py).
