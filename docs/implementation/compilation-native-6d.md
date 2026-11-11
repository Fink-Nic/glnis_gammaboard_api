# Native compilation, six-dimensional proposals, and checkpoint recovery

Native compilation improves steady training by 2.3–2.7×, inference at batch
1,000 by 2.9–4.0×, and bulk inference by 1.8–2.0× in these measurements.
Checkpoint creation takes 0.36–0.55 seconds and does not compile anything.
Recovered workers regain steady performance after lazy compilation; empty-cache
CUDA graph runs show floating-point roundoff rather than bitwise continuation.

This repeats the [6D tensor-method experiment](compilation-6d.md) through
`GammaBoardSampler` and its orchestrator-owned execution policy. It includes
the public generation/feedback path, training-buffer copies, finite-value checks,
the configured KL loss, eager Adam, and the cosine scheduler. It also measures
checkpoint creation and continuation in fresh Python workers.

The [benchmark](../../benchmarks/orchestrator_compile.py) does not call
`torch.compile` itself. It selects `execution.kind="torch_compile"`; the native
runtime owns compilation, CUDA graph boundaries, and retained tensor storage.

## Workload and protocol

| Setting | Continuous-only | With discrete sampling |
| --- | --- | --- |
| Public domain | `RectangularDomain((), 6)` | `RectangularDomain((3,), 6)` |
| Continuous coupling blocks | 6 | 6 |
| MLP configuration | `layers=4, units=512` | `layers=4, units=512` for both coupling and discrete head |
| Parameters | 3,450,414 | 4,045,233 |

`layers=4` means four linear layers, including the output layer: three hidden
layers have 512 units. The mixed case has one discrete axis with three choices.
It retains the 64-feature Transformer embedding/feedforward width, four heads,
and one encoder layer. Both proposals use 10-bin rational-quadratic splines and
float64. The seed is 42, and parameters are perturbed away from the initially
uniform proposal before training.

Measured on 2026-11-11 with Python 3.13.12, PyTorch 2.15.0+cu132, NVIDIA A30 GPUs,
and one CPU thread per process. Each domain uses a separate GPU. Timed GPU blocks
are synchronized. These are shared-host measurements with ordinary timing
variation; no other system/Triton caches were purged.

Each profile starts with its own empty Inductor cache directory. The three
policies are eager, native `torch_compile` defaults, and `reduce-overhead` with
static sampling shapes. Both compiled policies use `fullgraph=True` and a static
training `prob`; default sampling uses `dynamic=True`.

The training batch is 1,000. Each training measurement has one separately timed
first cycle, 100 warmup cycles, and three timed repeats of 30 cycles. After the
first 191 steps, the benchmark generates 500 points, ingests feedback for 250,
and saves a checkpoint. It then generates the remaining 500 and completes the
pending feedback window. Another 191 steps finish the configured 383-step run.
Inference then uses `eval()` and `inference_mode`, with 100 warmups and three
repeats of 100 calls. Bulk inference uses batches of 100,000, three warmups, and
three repeats of 10 calls. Tables show the median repeat.

Each saved checkpoint is also recovered in two fresh workers: one reuses the
parent's Inductor directory, and the other starts with an empty directory. Both
workers repeat the continuation, remaining training, and inference protocol.
Checkpoint/reference files live in temporary directories. Full repeat timings,
arguments, and correctness diagnostics are in the [raw measurements](../../benchmarks/results/orchestrator_compile_6d_4x512_a30_2026-11-11.json).

## Native steady performance

<!-- results:steady:start -->
All values are **milliseconds per call**; compiled rows include speedup over eager.

| Domain | Policy | Training cycle, 1,000 | Inference, 1,000 | Bulk inference, 100,000 |
| --- | --- | ---: | ---: | ---: |
| 6D | eager | 31.763 | 7.153 | 211.047 |
| 6D | default | 13.131 (2.42×) | 1.947 (3.67×) | 106.808 (1.98×) |
| 6D | reduce-overhead | 13.473 (2.36×) | 1.794 (3.99×) | 106.222 (1.99×) |
| 6D + three-way | eager | 40.975 | 7.865 | 238.951 |
| 6D + three-way | default | 17.905 (2.29×) | 2.725 (2.89×) | 131.844 (1.81×) |
| 6D + three-way | reduce-overhead | 15.449 (2.65×) | 2.520 (3.12×) | 131.382 (1.82×) |
<!-- results:steady:end -->

Training includes a cheap positive synthetic evaluator:
`exp(-sum((x - 0.4)**2)) + 0.05`, multiplied by `1 + 0.25 * sum(discrete)`.
The benchmark excludes expensive integrands, coordinate mappings, worker IPC,
and overlap with concurrent evaluation. Application throughput still depends on
the evaluator's share of the total runtime.

The native path performs more work than the earlier tensor-method cycle,
particularly per-parameter finite-gradient checks and public buffer/array
handling. The two reports measure different scopes; their absolute times and
speedups should not be interchanged.

CUDA graphs offer a modest gain for the mixed proposal; continuous training is
similar to default compilation within the observed variation. Dynamic sampling
avoids the additional compilation for 500-point and 100,000-point requests in
this run. Static CUDA graphs spend another 30–40 seconds on the partial shape
and 35–44 seconds on bulk inference. This matters when deciding between fixed
and variable sampling batches.

## Startup and checkpoint creation

<!-- results:startup:start -->
| Domain | Policy | First training s | First partial draw s | First inference s | First bulk s | Save ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 6D | eager | 0.52 | 0.008 | 0.01 | 0.22 | 364 |
| 6D | default | 93.93 | 0.002 | 2.96 | 0.10 | 357 |
| 6D | reduce-overhead | 83.11 | 29.946 | 1.58 | 35.17 | 361 |
| 6D + three-way | eager | 1.49 | 0.009 | 0.02 | 0.24 | 418 |
| 6D + three-way | default | 123.81 | 0.003 | 45.38 | 0.13 | 431 |
| 6D + three-way | reduce-overhead | 110.59 | 40.417 | 32.20 | 43.62 | 548 |

Checkpoints are approximately 93.31 MiB for 6D and 109.45 MiB for the mixed proposal. Compiled policies change only a few KiB of metadata.
<!-- results:startup:end -->

First-call times include lazy compilation and CUDA graph setup where applicable.
The first inference call can specialize again after the training-to-evaluation
transition. Static sampling also specializes for the partial 500-point request
and the 100,000-point bulk request. Dynamic sampling can reuse more shapes but
does not eliminate mode/context specialization.

Saving serializes the raw module, optimizer, scheduler, training status/buffers,
and RNG states. It neither compiles nor warms a graph, and the benchmark verifies
that saving leaves CPU/CUDA RNG states unchanged. Compiler artifacts are not
included in checkpoint sizes. Compilation is repeated lazily after reconstruction,
with compatible external caches reducing the work.

For default compilation, the initial training cost alone takes roughly 5,000
full cycles to amortize against the measured eager baseline. Extra shapes/modes,
checkpoint recovery, evaluator cost, and warmup can change that estimate. The
383-step benchmark run does not recover its fresh-compilation cost. These results
support keeping compilation optional, with eager execution as the default.

## Fresh-worker recovery

<!-- results:restore:start -->
All times are **seconds**.

| Domain | Policy | Cache | Ready | Load | First draw, 500 | Finish feedback | Next cycle, 1,000 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 6D | eager | reused | 3.588 | 1.733 | 0.323 | 0.285 | 0.043 |
| 6D | eager | empty | 3.622 | 1.746 | 0.308 | 0.227 | 0.032 |
| 6D | default | reused | 4.153 | 2.210 | 12.834 | 2.151 | 0.020 |
| 6D | default | empty | 3.828 | 1.978 | 26.294 | 21.765 | 0.017 |
| 6D | reduce-overhead | reused | 4.320 | 2.241 | 4.727 | 2.195 | 2.068 |
| 6D | reduce-overhead | empty | 4.107 | 2.120 | 16.766 | 21.893 | 14.522 |
| 6D + three-way | eager | reused | 3.723 | 1.722 | 1.085 | 0.379 | 0.043 |
| 6D + three-way | eager | empty | 3.949 | 1.912 | 1.106 | 0.364 | 0.041 |
| 6D + three-way | default | reused | 3.919 | 2.040 | 15.197 | 2.583 | 0.021 |
| 6D + three-way | default | empty | 4.461 | 2.461 | 34.438 | 34.289 | 0.020 |
| 6D + three-way | reduce-overhead | reused | 4.035 | 2.243 | 5.809 | 2.647 | 2.189 |
| 6D + three-way | reduce-overhead | empty | 4.324 | 2.340 | 20.251 | 31.575 | 16.544 |
<!-- results:restore:end -->

“Ready” starts at script entry before NumPy/PyTorch imports and includes context
creation and checkpoint loading; interpreter/bootstrap startup is excluded.
“Load” measures only the restore factory after CUDA context discovery.
It overlaps Ready and must not be added to it. The first 500-point draw and the
feedback completing the interrupted window are separate lazy compilation
boundaries. The next full 1,000-point cycle can need another static shape.

<!-- results:recovered-steady:start -->
Each cell is **uninterrupted / reused-cache recovery / empty-cache recovery**, in ms/call.

| Domain | Policy | Training | Inference, 1,000 | Bulk, 100,000 |
| --- | --- | ---: | ---: | ---: |
| 6D | eager | 32.350 / 30.054 / 39.540 | 7.153 / 6.567 / 8.087 | 211.047 / 211.447 / 212.021 |
| 6D | default | 13.117 / 12.484 / 11.572 | 1.947 / 2.023 / 1.945 | 106.808 / 106.697 / 106.951 |
| 6D | reduce-overhead | 11.667 / 11.277 / 11.456 | 1.794 / 1.784 / 1.724 | 106.222 / 106.399 / 106.535 |
| 6D + three-way | eager | 38.475 / 42.482 / 44.646 | 7.865 / 8.768 / 8.326 | 238.951 / 238.641 / 238.967 |
| 6D + three-way | default | 20.013 / 16.327 / 18.019 | 2.725 / 2.545 / 2.900 | 131.844 / 131.917 / 132.022 |
| 6D + three-way | reduce-overhead | 16.444 / 14.799 / 16.160 | 2.520 / 2.205 / 2.439 | 131.382 / 131.333 / 130.991 |
<!-- results:recovered-steady:end -->

Recovery restores steady performance after startup. Reused artifacts still
require graph reconstruction, guards, loading, and CUDA graph capture; a
checkpoint does not contain an executable compiled worker. An empty compiler
cache does not prevent restoration, but increases the first-use cost.

## Numerical continuation

Before generating anything, workers compare restored parameters, optimizer,
scheduler, training buffers/statistics, pending mapping weights, and CPU/CUDA RNG
states exactly against the saved parent state. After continued execution they
compare the next partial draw, first inference/bulk draws, final parameters,
optimizer, scheduler, and complete loss history against uninterrupted execution.
Discrete values, scheduler, and optimizer parameter groups remain exact. Floating
outputs record bitwise equality separately from agreement at
`rtol=1e-8, atol=1e-9`, matching the earlier f64 benchmark tolerance.
Validation runs outside the timed blocks.

<!-- results:correctness:start -->
All twelve workers restored saved state exactly before execution and passed the
numerical continuation and public PDF/weight checks. Bitwise continuation:

| Domain | Policy | Reused cache | Empty cache |
| --- | --- | --- | --- |
| 6D | eager | passed | passed |
| 6D | default | passed | passed |
| 6D | reduce-overhead | passed | failed |
| 6D + three-way | eager | passed | passed |
| 6D + three-way | default | passed | passed |
| 6D + three-way | reduce-overhead | passed | failed |

Maximum absolute differences from uninterrupted execution:

| Domain / policy / cache | Coordinates | Weights | Parameters | Optimizer | Loss history |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6D / reduce-overhead / empty | 1.11e-15 | 3.29e-14 | 1.15e-15 | 4.68e-17 | 5.13e-16 |
| 6D + three-way / reduce-overhead / empty | 1.33e-15 | 1.49e-13 | 1.29e-15 | 4.46e-17 | 5.77e-16 |
<!-- results:correctness:end -->

The raw results preserve the initial strict CUDA graph failures as well as the
follow-up diagnostics. The benchmark did not change sampler arithmetic or relax
the execution tests. Exactly restored state does not guarantee bitwise arithmetic
after fresh compiler code generation, even on the same device and precision.

Sampling weights are also checked against public eager `pdf` queries at the
updated parameters: `pdf * weights == 1` within the same f64 tolerance. This
checks compiled sampling against the independent eager scoring path. The earlier
report retains the separate eager/compiled gradient comparisons and its f32
limitation; this follow-up measures f64 only.

## Reproduce

```bash
# Continuous-only; omit the trailing --discrete for the default three-way axis.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
TRITON_DEFAULT_FP_FUSION=1 TORCHINDUCTOR_CACHE_DIR=/tmp/glnis-native-fresh \
  .venv/bin/python benchmarks/orchestrator_compile.py --device cuda:0 \
  --policy default --discrete --output /tmp/native-continuous.json
```

Repeat with a distinct empty cache directory for each initial profile, using
`--policy eager` and `--policy reduce-overhead`, and `--discrete 3` for the mixed
domain. Each invocation launches both recovery workers automatically. Benchmark
defaults already select six continuous dimensions and `layers=4, units=512`.

Equivalent local configuration uses `[sampler.module.flow.subnet]` and
`[sampler.module.discrete.subnet]` for MLP settings, plus
`[sampler.orchestrator.execution]` for compilation. See the
[module reference](../configuration/modules.md) and
[execution reference](../configuration/orchestrators.md#execution-and-compilation).
