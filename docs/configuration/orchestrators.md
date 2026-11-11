# Orchestrators, losses, and schedulers

The current orchestrator is `kind="rectangular"`. Options below belong under
`[sampler.orchestrator]` in local TOML. Source:
[RectangularOrchestratorConfig](../../src/glnis/core/nn/orchestrator.py).

| Option | Type | Default | Meaning / constraint |
| --- | --- | --- | --- |
| `kind` | string | `rectangular` | Registered orchestrator implementation |
| `seed` | nonnegative integer or Python `None` | `42` | Seeds initialization and draws; fresh stages seed a new run |
| `training_steps` | integer ≥ 0 | `100` | Number of optimization windows; zero starts in inference |
| `training_batch_size` | integer > 0 | `1000` | Samples collected before an optimization step |
| `max_batch_size` | integer > 0 | `100000` | Maximum model forward-pass size; also caps an inference draw |
| `use_gpu` | boolean | `true` | Use CUDA when available; otherwise use CPU |
| `use_f64` | boolean | `true` | Float64 model/training buffers; false uses float32 |
| `cuda_id` | integer ≥ 0 | `1` | CUDA device index, clamped to the last available device |
| `learning_rate` | positive finite number | `0.001` | Initial Adam learning rate |
| `loss` | table | `{kind="kl_divergence"}` | Registered objective |
| `scheduler` | table or Python `None` | `{kind="cosineannealing"}` | Registered learning-rate scheduler |

A training draw can exceed `max_batch_size` by combining several forward passes,
but cannot exceed the ungenerated remainder of its training window. Optimizer
selection is currently fixed to Adam in this orchestrator. A different optimizer
or training strategy can be implemented in another orchestrator.

## Losses

Source: [losses.py](../../src/glnis/core/nn/losses.py). Each loss receives a training
status and differentiable probabilities of the saved generated samples.

| `kind` | Options | Purpose |
| --- | --- | --- |
| `kl_divergence` | `threshold`, `normalize_across_training_steps` | Forward KL objective for the absolute integrand target |
| `rkl_divergence` | `threshold`, `normalize_across_training_steps` | Reverse KL objective with sampling-density correction |
| `variance` | `normalize_across_training_steps` | Integration-variance objective |

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `normalize_across_training_steps` | boolean | `false` | Use accumulated integration estimates instead of just the current window's normalization |
| `threshold` | positive finite number or Python `None` | `None` | Apply an arcsinh soft clip to normalized importance ratios in KL objectives |

`variance` does not accept `threshold`. All-zero windows advance training with
loss `0.0` and skip optimizer/scheduler updates. Nonfinite losses or gradients
skip the update and emit a runtime warning; the window still finishes.

## Schedulers

Source: [schedulers.py](../../src/glnis/core/nn/schedulers.py). Implementations use
PyTorch's `step`, `state_dict`, and `load_state_dict` interface.

| `kind` | Option | Default | Meaning |
| --- | --- | --- | --- |
| `cosineannealing` | `T_max` | `max(training_steps, 1)` | Positive integer horizon |
| `cosineannealing` | `eta_min` | `0.0` | Minimum learning rate |
| `step` | `step_size` | required | Positive integer number of updates between LR changes |
| `step` | `gamma` | `0.1` | Multiplicative factor |
| `exponential` | `gamma` | required | Multiplicative factor per update |

Python callers can use `scheduler=None`. To disable the scheduler in TOML, use
the retained `use_scheduler=false` option. Do not combine that spelling with a
conflicting scheduler table.

## Retained option spellings

`loss_type` and `loss_kwargs` normalize to the `loss` table. `use_scheduler` and
`scheduler_type` normalize to the `scheduler` table. New configurations should
use component tables, except the TOML disable case above. Conflicting old and new
settings raise an error. Snapshots store the normalized settings.

For which settings can change during restore or stage transfer, see
[checkpoints and stages](../implementation/checkpoints-and-stages.md).
