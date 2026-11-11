# Orchestrators, losses, and schedulers

The current orchestrator is `kind="rectangular"`. Options below belong under
`[sampler.orchestrator]` in local TOML. Source:
[RectangularOrchestratorConfig](../../src/glnis/core/nn/orchestrator.py).

Direct orchestrator construction accepts `module_kwargs` for the proposal
configuration. Through `GammaBoardSampler`, set the adapter's `module` argument
(`[sampler.module]` in local TOML); the adapter routes it to `module_kwargs`.

| Option | Type | Default | Meaning / constraint |
| --- | --- | --- | --- |
| `kind` | string | `rectangular` | Registered orchestrator implementation |
| `seed` | nonnegative integer or Python `None` | `42` | Seeds initialization and draws; fresh stages seed a new run |
| `training_steps` | integer ≥ 0 | `100` | Number of optimization windows; zero starts in inference |
| `training_batch_size` | integer > 0 | `1000` | Samples collected before an optimization step |
| `max_batch_size` | integer > 0 | `100000` | Maximum model forward-pass size; also caps an inference draw |
| `use_gpu` | boolean | `true` | Use CUDA when available; otherwise use CPU |
| `use_f64` | boolean | `true` | Float64 model/training buffers; false uses float32 |
| `gpu_id` | `"auto"` or integer ≥ 0 | `"auto"` | Select a visible CUDA device automatically or by index |
| `learning_rate` | positive finite number | `0.001` | Initial Adam learning rate |
| `loss` | table | `{kind="kl_divergence"}` | Registered objective |
| `scheduler` | table, `false`, or Python `None` | `{kind="cosineannealing"}` | Registered learning-rate scheduler; false/None disables it |
| `execution` | table | `{kind="eager"}` | Runtime sampling/probability execution policy |

A training draw can exceed `max_batch_size` by combining several forward passes,
but cannot exceed the ungenerated remainder of its training window. Optimizer
selection is currently fixed to Adam in this orchestrator. A different optimizer
or training strategy can be implemented in another orchestrator.

## GPU selection

`gpu_id="auto"` selects the first visible GPU with no active processes, using
`nvidia-smi` process information matched by UUID. CUDA visibility and ordering
are preserved, including reordered `CUDA_VISIBLE_DEVICES`. If all GPUs are busy,
process information is unavailable, or a MIG UUID cannot be matched, selection
uses the largest free VRAM reported by CUDA. Ties choose the first visible GPU.
The process query has a five-second timeout and needs no NVML Python dependency.

An explicit integer selects that visible device; an out-of-range index raises
an error instead of selecting another GPU. `use_gpu=false`, or unavailable CUDA,
selects CPU. Automatic selection reflects occupancy at construction time and
does not reserve the device against other workers starting concurrently.

## Execution and compilation

Source: [execution.py](../../src/glnis/core/nn/execution.py). Execution is owned by
the orchestrator; the proposal remains a regular `nn.Module` for parameters,
device moves, optimizer state, and serialization.

`kind="eager"` accepts no additional options. `kind="torch_compile"` accepts:

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `backend` | nonempty string | `inductor` | PyTorch compiler backend |
| `sample` | table or boolean | `{fullgraph=true, dynamic=true}` | Compile sampling; `false` keeps it eager, `true` uses defaults |
| `prob` | table or boolean | `{fullgraph=true, dynamic=false}` | Compile complete-point training probabilities; `false` keeps them eager |

Method tables accept `torch.compile` keyword arguments, including `mode`,
`fullgraph`, `dynamic`, and `options`. Their backend inherits the outer setting
unless overridden in the method table. Compilation is lazy, with no constructor
warmup, and errors propagate rather than silently falling back. If explicitly
using `fullgraph=false`, PyTorch may split execution at graph breaks.

```toml
[sampler.orchestrator.execution]
kind = "torch_compile"

[sampler.orchestrator.execution.sample]
dynamic = true

[sampler.orchestrator.execution.prob]
dynamic = false
```

For fixed sampling batches, `sample={dynamic=false, mode="reduce-overhead"}`
enables the CUDA-graph experiment. The orchestrator marks execution boundaries
and copies retained samples/probabilities into owned training buffers before
another compiled call. NumPy conversion, feedback, losses, optimizer/scheduler
policy, `pdf`, and `discrete_mass` remain eager. Compilation of `prob` includes
the associated autograd work, but does not compile the whole optimization loop.

Training generation uses `no_grad`; completed runs use `eval` and
`inference_mode`. A fresh stage explicitly returns to training mode.
Execution settings are saved as metadata and may be overridden on restore;
compiled callables are rebuilt outside the serialized module. Saving a checkpoint
does not compile or warm up anything. See [recovery semantics](../implementation/checkpoints-and-stages.md)
and the [native six-dimensional report](../implementation/compilation-native-6d.md)
for checkpoint/recovery costs. The earlier [tensor-method report](../implementation/compilation-6d.md)
also covers the observed float32 gradient discrepancy.

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

Use `scheduler=false` in TOML or Python; Python also accepts `scheduler=None`.
Losses and enabled schedulers use their component tables. Configuration aliases
from previous schemas are not accepted.

For which settings can change during restore or stage transfer, see
[checkpoints and stages](../implementation/checkpoints-and-stages.md).
