# Compilation with six-dimensional, 4×512 proposals

For these f64 models, complete-method compilation improves sampling at batch
1,000 by 3.8–4.6× and training cycles by 3.5–3.6×. Bulk sampling improves by
1.8–2×. MLP-only compilation gives little consistent benefit, and fresh inference
or training compilation takes 31–96 seconds.

This repeats the [earlier compilation experiment](compilation.md) with six
continuous dimensions and larger MLPs, comparing a continuous-only proposal with
one that also samples a three-way discrete choice. It measures proposal execution
and training costs. Production defaults to eager, with compilation available
through [orchestrator execution settings](../configuration/orchestrators.md#execution-and-compilation).

## Models and measurement

| Setting | Continuous-only | With discrete sampling |
| --- | --- | --- |
| Domain | `RectangularDomain((), 6)` | `RectangularDomain((3,), 6)` |
| Continuous coupling blocks | 6 | 6 |
| Coupling MLP configuration | `layers=4, units=512` | `layers=4, units=512` |
| Discrete-head MLP | Absent | `layers=4, units=512` |
| Trainable parameters | 3,450,414 | 4,045,233 |

Here `layers=4` means four linear layers, with three 512-unit hidden layers, as
defined by the existing MLP implementation. The discrete case has **one axis with
three choices**, not three discrete axes. Its Transformer retains the earlier
settings: 64 embedding features, 64 feedforward features, four heads, and one
encoder layer. Both models use 10-bin rational-quadratic splines, logarithmic
coupling masks, and no external conditioning. Parameters are perturbed away from
the initially uniform proposal with the same seed before comparison.

The environment is Python 3.13.12 and PyTorch 2.15.0+cu132. GPU runs use two NVIDIA
A30s on the same host, one domain per GPU; CPU runs use one thread on an AMD EPYC
9754. Each invocation compares fresh copies of identical initial parameters.
CUDA synchronization surrounds timed blocks, so host/launch overhead is included.
Precision is f64 unless explicitly marked f32.

The primary inference and cycle comparisons use 100 warmup calls. Inference uses
three repeats of 100 calls; training uses three repeats of 30 calls. Initial
fresh-cache runs, the other diagnostic variants, and CPU runs use three warmup
calls. Bulk inference uses three repeats of 10 calls with batches of 100,000;
CPU sampling uses three repeats of 10 calls with batches of 1,000. Tables show
the median repeat. All repeats, arguments, cache groups, and correctness results
are preserved in the [recorded measurements](../../benchmarks/results/torch_compile_6d_4x512_a30_2026-11-11.json).

The longer warmup confirms the main mode comparisons after short-warmup CUDA-graph
timings showed variation. These are shared-host observations, not isolated
hardware guarantees. Compare each candidate to its paired eager baseline rather
than subtracting rows measured in separate processes or on different GPUs.

## Steady-state GPU results

`default` and `reduce-overhead` compile the complete `sample()`/`prob()` tensor
methods with Inductor, `fullgraph=True`, and static batch shapes. Protected
CUDA-graph training includes explicit cycle boundaries and copies of generated
samples/probabilities outside the compiled function, as required by the earlier
lifetime experiment.

<!-- results:core:start -->
All times below are **milliseconds per call**, shown as eager → compiled (speedup).

| Workload | Batch | Continuous-only | With three-way discrete sampling |
| --- | ---: | ---: | ---: |
| Sampling, default | 1,000 | 6.766 → 1.467 (4.61×) | 7.954 → 2.077 (3.83×) |
| Sampling, reduce-overhead | 1,000 | 6.325 → 1.391 (4.55×) | 7.556 → 1.758 (4.30×) |
| Probability queries, default | 1,000 | 5.611 → 1.483 (3.78×) | 6.508 → 1.837 (3.54×) |
| Fixed-batch training, default | 1,000 | 19.572 → 5.556 (3.52×) | 25.722 → 6.567 (3.92×) |
| Training cycle, default | 1,000 | 28.611 → 7.919 (3.61×) | 33.192 → 9.511 (3.49×) |
| Training cycle, protected reduce-overhead | 1,000 | 28.649 → 7.149 (4.01×) | 39.041 → 8.527 (4.58×) |
| Bulk sampling, default | 100,000 | 209.859 → 106.291 (1.97×) | 239.672 → 131.191 (1.83×) |
| Bulk sampling, reduce-overhead | 100,000 | 210.975 → 106.386 (1.98×) | 238.643 → 131.194 (1.82×) |

Probability-only and fixed-batch training rows use the initial three-call warmup;
the primary 1,000-sample sampling/cycle rows use 100 warmup calls.
<!-- results:core:end -->

The `train` workload measures probability evaluation, a weighted
negative-log-probability loss, backward, and eager Adam on a fixed batch. The
`cycle` workload also generates samples, transfers coordinates/proposal
probabilities to NumPy, evaluates a cheap positive target, and transfers feedback
weights back to the device. No native integrand, coordinate mapping, public
adapter, scheduler, or IPC is included. The larger model represents the requested
neural workload; integrand evaluation remains a synthetic surrogate.

<!-- results:diagnostics:start -->
All times below are **milliseconds per call**, shown as eager → compiled (speedup).

| Workload | Batch | Continuous-only | With three-way discrete sampling |
| --- | ---: | ---: | ---: |
| MLP-only sampling | 1,000 | 6.476 → 6.969 (0.93×) | 8.567 → 8.233 (1.04×) |
| MLP-only training cycle, variant limit 32 | 1,000 | 35.508 → 32.011 (1.11×) | 48.139 → 60.669 (0.79×) |
| Sampling, training mode + no_grad | 1,000 | 9.069 → 2.066 (4.39×) | 9.371 → 1.839 (5.10×) |
| Sampling, eval + no_grad | 1,000 | 6.907 → 2.245 (3.08×) | 8.350 → 2.862 (2.92×) |
| Sampling, dynamic shapes | 1,000 | 7.766 → 1.509 (5.14×) | 9.684 → 1.947 (4.97×) |
| Sampling, f32 | 1,000 | 7.722 → 1.428 (5.41×) | 10.407 → 1.923 (5.41×) |
| Training cycle, f32 | 1,000 | Gradient check failed | 46.446 → 14.808 (3.14×) |
| CPU sampling, f64 | 1,000 | 262.864 → 189.767 (1.39×) | 304.043 → 219.278 (1.39×) |
<!-- results:diagnostics:end -->

MLP-only mixed training initially exceeded Dynamo's default eight-variant limit
for the shared MLP `forward()` function. Different coupling feature widths and
training/generation contexts consume separate variants. The repeat uses
`--recompile-limit 32` for both domains; the original failure remains in the raw
results. This flag changes only the benchmark process. See the
[PyTorch compile API](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)
for code-object cache sharing and recompilation limits.

## Startup, shapes, and memory

Fresh-cache rows use a separate empty Inductor directory for each domain's first
sampling and first training-cycle run. Other system/Triton caches were not
purged. Runtime setup and the first call are recorded separately from steady
state. Break-even estimates subtract eager startup and divide the remaining
startup cost by the measured per-call saving; they exclude optional warmup,
validation, and checkpoint checks.

<!-- results:startup:start -->
| Domain | Workload | Inductor cache | Runtime setup s | First call s | Approx. break-even calls |
| --- | --- | --- | ---: | ---: | ---: |
| Continuous | Sampling, 1,000 | Fresh | 1.07 | 31.08 | 6,562 |
| Continuous | Training cycle, 1,000 | Fresh | 0.50 | 78.95 | 3,985 |
| Continuous | Sampling, 1,000 | Same workload cached | 1.03 | 3.30 | 816 |
| Continuous | Training cycle, 1,000 | Same workload cached | 0.42 | 5.42 | 223 |
| Continuous | Bulk sampling, 100,000 | Existing cache, new shape | 0.95 | 44.53 | 437 |
| Mixed | Sampling, 1,000 | Fresh | 0.83 | 32.61 | 5,765 |
| Mixed | Training cycle, 1,000 | Fresh | 0.43 | 95.51 | 3,717 |
| Mixed | Sampling, 1,000 | Same workload cached | 0.91 | 3.40 | 729 |
| Mixed | Training cycle, 1,000 | Same workload cached | 0.39 | 5.60 | 199 |
| Mixed | Bulk sampling, 100,000 | Existing cache, new shape | 1.15 | 52.49 | 492 |

For the fresh-cache 1,000-sample workloads, a 50–100-step training run still does
not amortize compilation. Reusing artifacts changes the calculation substantially,
but exact break-even estimates vary with launch overhead, cache contents, and the
workload. The fully cached rows are independent repeat measurements, not a promise
that an arbitrary worker can start that quickly.
<!-- results:startup:end -->

<!-- results:shapes:start -->
`dynamic=True` successfully handled the additional batch shapes:

| Domain | Steady-state batch 1,000 ms | First batch 7 ms | First batch 1 s |
| --- | ---: | ---: | ---: |
| Continuous | 1.509 | 2.93 | 23.32 |
| Mixed | 1.947 | 2.74 | 17.17 |

Batch seven reused the dynamically compiled graph; batch one still required
specialization. The first dynamic call at batch 1,000 took 53.4 seconds for the
continuous model and 55.0 seconds for the mixed model, using existing caches.
Dynamic shapes help fragmented requests without eliminating all startup costs.

| Domain | Sampling 1,000 peak MiB | Cycle 1,000 peak MiB | Bulk default peak MiB | Bulk reduce-overhead peak MiB |
| --- | ---: | ---: | ---: | ---: |
| Continuous | 69.1 | 200.0 | 956.2 | 955.5 |
| Mixed | 78.3 | 238.6 | 1003.3 | 1003.3 |
<!-- results:shapes:end -->

GPU memory counters cover PyTorch allocations in the benchmark process, including
the base model and its measurement copy; they exclude driver/library allocations
outside PyTorch's allocator. Peak allocated memory is reset before each measured
candidate and recorded after timed repeats, before gradient/restore validation.
Reserved memory also reflects the caching allocator's earlier allocations; it is
not an isolated estimate of CUDA-graph overhead.

## Interpretation

<!-- results:interpretation:start -->
Complete-method compilation remains useful with the larger models: default
sampling at batch 1,000 improves by 4.61× without discrete sampling and 3.83× with
it. Default training cycles improve by 3.61× and 3.49× respectively. These are
smaller gains than the best small-model CUDA-graph result in the earlier report.

CUDA graphs bring a modest additional reduction in compiled time at batch 1,000:
sampling falls from 1.467 to 1.391 ms for the continuous model and from 2.077 to
1.758 ms for the mixed model. Training cycles fall from 7.919 to 7.149 ms and
from 9.511 to 8.527 ms. These cross-run differences are approximate; the reported
speedup ratios use each run's own eager baseline. At batch 100,000, default and
CUDA-graph sampling are effectively tied, with approximately 2× / 1.8× gains over
eager. The earlier 10–15× inference result should not be generalized to this
larger workload.

Compiling MLPs alone still gives little consistent benefit. Warm sampling is
0.93× / 1.04× eager, and training cycles with the raised variant limit are
1.11× / 0.79×. Compiling across the complete proposal methods is the stronger
candidate here. CPU sampling improves by about 1.39× for both domains.

All f64 complete-method probability, gradient, shape, and requested replay checks
passed. Supplemental continuous-only f32 training reproducibly failed the strict
gradient check: one element of a 512-element bias gradient differed by
`2.79e-5`, above the `2e-5` absolute tolerance. The failing parameter is
`continuous_flow.subnets.2.layers.4.bias`; its relative gradient L2 error is
`0.00141018`. No tolerance was relaxed and no validated training speedup is
reported for that case. f32 sampling passed for both models, and the mixed f32
training check passed. This experiment supports the f64 results more broadly
than the f32 training results.
<!-- results:interpretation:end -->

These speedups apply to the sampler and training paths measured here. With an
expensive integrand, application throughput depends on the evaluator's share of
the run and how evaluation overlaps sampling/training. For a serial workload,
the corresponding speedup is
`(sampler_time + evaluator_time) / (compiled_sampler_time + evaluator_time)`.
The benchmark does not assign an invented cost to a real integrand.

## Reproduce

The [benchmark](../../benchmarks/torch_compile.py) accepts explicit MLP depth and
width overrides. A width override applies to both the coupling MLPs and the
discrete head; omitting it retains the original Gaussian-profile defaults.

```bash
# Six continuous coordinates, no discrete sampling.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
TRITON_DEFAULT_FP_FUSION=1 .venv/bin/python benchmarks/torch_compile.py \
  --device cuda:0 --continuous 6 --discrete --mlp-layers 4 --mlp-units 512 \
  --workload sample --warmup 100 --iterations 100 --check-restore \
  --output /tmp/compile-6d-continuous.json

# Six continuous coordinates plus one three-way discrete choice.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
TRITON_DEFAULT_FP_FUSION=1 .venv/bin/python benchmarks/torch_compile.py \
  --device cuda:0 --continuous 6 --discrete 3 --mlp-layers 4 --mlp-units 512 \
  --workload cycle --mode reduce-overhead --protect-training-samples \
  --warmup 100 --output /tmp/compile-6d-mixed-training.json
```

For the fresh-cache measurements, use a new `TORCHINDUCTOR_CACHE_DIR`, three
warmup calls, and 30 iterations. Reuse that directory to measure subsequent
processes with cached artifacts. Use `--target subnets --recompile-limit 32`
for MLP-only training; `--batch-size 100000 --warmup 3 --iterations 10` for bulk
inference; `--dynamic --probe-batch-sizes 1 7 1000` for shape checks; and
`--device cpu --iterations 10` for CPU sampling. CPU compilation also requires
working compiler tools and `openssl` on `PATH` in this environment.

## Correctness and scope

Sampling probabilities are compared with eager probability queries, and training
probabilities/parameter gradients are compared against eager at the candidate's
updated parameters. Tolerances are unchanged: f64 uses `rtol=1e-8, atol=1e-9`;
f32 uses `rtol=2e-4, atol=2e-5`. The benchmark does not require matching random
streams between eager and compiled execution.

The initial 1,000-sample static inference cases also check exact module/RNG
continuation after reconstructing compiled callables from a raw module snapshot.
This is same-process
replay, not fresh-worker or interrupted-feedback/optimizer recovery. Those
application integration checks are covered separately by the current
[execution tests](../../tests/test_execution.py), including real Inductor
fresh-process recovery on CPU and CUDA. The historical benchmark's results and
limits above are unchanged.
