# Sampler compilation experiments

Compiling complete tensor methods can substantially accelerate this proposal on
CUDA. Compilation takes long enough that short training runs lose time overall.
Keep it opt-in and decide from the expected number of batches, including startup.
This branch provides a benchmark and removes graph-capture obstacles; the
production orchestrator still runs eagerly and has no compilation configuration.

## What is compiled

The orchestrator calls `SamplerModule.sample()` and `prob()`, not `forward()`.
`module.compile()` compiles `__call__`, so applying it to the proposal itself
does not compile these entry points. Compile their bound methods explicitly.
`module.compile()` does work for the MLP subnets, which are called normally.
See the [PyTorch module API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html#torch.nn.Module.compile).

Three changes allow full-graph capture and Inductor lowering of the mixed flow:

- The discrete Transformer receives `is_causal=True` with its causal mask,
  avoiding automatic inspection of mask values.
- Coupling blocks use integer coordinate indices derived from their fixed masks;
  discrete conditioning concatenates one-hot blocks with known cardinalities.
- The spline retains its batch shape and selects identity outputs with `where`.
  Inactive rows are evaluated at an interior point to avoid invalid arithmetic.
  Its existing rule that an outside coordinate makes the entire row inactive is
  preserved.

Coordinate indices are rebuilt when loading older module snapshots or replacing
mask buffers through `load_state_dict()`. Parameters, mapping Jacobians, feedback
corrections, checkpoint contents, and configuration defaults retain their roles.

The benchmark keeps compiled callables outside the serialized proposal. Saving a
proposal does not save a runnable compiler artifact: recreate the callables after
loading. NumPy mappings, feedback windows, optimizer policy, scheduling, and
checkpoint I/O remain outside the compiled region.

## Measurements

Measured on 2026-11-11 with Python 3.13.12, PyTorch 2.15.0+cu132, an NVIDIA A30,
and one CPU thread on an AMD EPYC 9754 host. These are shared-host measurements,
not an isolated hardware performance guarantee. The network sizes match
[the Gaussian example](../../tests/gaussian_test.py): three continuous dimensions,
one discrete dimension of cardinality three, 3-layer/32-unit coupling MLPs,
10 spline bins, and one 64-feature Transformer layer with four heads. Parameters
are perturbed away from the initially uniform proposal before comparison.

Each result uses three warmup calls and the median of three repeats of 30 calls;
the 100,000-sample case uses 10 calls per repeat. CUDA synchronization surrounds
each repeat, so timings include Python/launch overhead. Eager and compiled runs
start from separate copies of the same proposal. Default compilation uses
`fullgraph=True`, `dynamic=False`, and the Inductor backend.

| Workload | Batch | Precision | Eager ms/call | Compiled ms/call | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| CUDA sampling, eval + inference mode | 1,000 | f64 | 6.408 | 1.047 | 6.1× |
| CUDA sampling, `reduce-overhead` | 1,000 | f64 | 4.971 | 0.321 | 15.5× |
| CUDA sampling, eval + inference mode | 100,000 | f64 | 45.028 | 11.538 | 3.9× |
| CUDA sampling, training mode + no_grad | 1,000 | f64 | 8.359 | 1.297 | 6.4× |
| CUDA sampling, eval + no_grad | 1,000 | f64 | 5.932 | 1.989 | 3.0× |
| CUDA probability, eval + inference mode | 1,000 | f64 | 5.467 | 0.910 | 6.0× |
| CUDA probability + loss/backward + Adam | 1,000 | f64 | 17.126 | 3.765 | 4.5× |
| CUDA generation/evaluation/training cycle | 1,000 | f64 | 31.495 | 7.749 | 4.1× |
| CUDA cycle, protected `reduce-overhead` outputs | 1,000 | f64 | 28.525 | 3.732 | 7.6× |
| CUDA sampling, MLP-only compilation | 1,000 | f64 | 7.001 | 6.721 | 1.04× |
| CUDA training cycle, MLP-only compilation | 1,000 | f64 | 22.344 | 27.616 | 0.81× |
| CUDA sampling, eval + inference mode | 1,000 | f32 | 7.181 | 1.003 | 7.2× |
| CPU sampling, eval + inference mode | 1,000 | f64 | 10.417 | 8.205 | 1.27× |

Only the proposal methods are compiled in the training rows. `train` measures a
fixed batch with a weighted negative-log-probability loss, backward, and eager
Adam. `cycle` additionally generates samples, transfers coordinates and proposal
probabilities to NumPy, evaluates a cheap positive synthetic target, and transfers
feedback weights back to the device. It does not run the public GammaBoard
adapter, mapping layers, scheduler, orchestration checks, IPC, or a native
integrand. It is a training-path approximation, not an application benchmark.

The repeated cycle measurement varied from 2.1× to 4.1×; the earlier
`reduce-overhead` measurement was 10.9×. Cross-row timings should not be used to
infer a precise advantage for dtype or evaluation context. Complete method
compilation consistently produced larger gains than MLP-only compilation here.
The [recorded measurements](../../benchmarks/results/torch_compile_a30_2026-11-11.json)
include repeat timings and arguments. A separate CUDA training check with two
discrete axes, external conditioning, and continuous-first factorization also
passed probability/gradient comparisons (64 samples, two repeats of five cycles).

### CUDA graphs during training

The unprotected `reduce-overhead` training cycle failed with an overwritten
CUDA-graph output. Dropping the benchmark's first result did not resolve it.
Marking cycle boundaries with `torch.compiler.cudagraph_mark_step_begin()` and
cloning generated samples/probabilities outside the compiled function resolved
the failure and passed probability/gradient checks. The protected cycle measured
7.6× faster with copies included; it took 14.1 seconds on first use with existing
caches and about 540 cycles to amortize startup.

Run this variant with `--workload cycle --mode reduce-overhead
--protect-training-samples`. The raw results retain both the failure and the
successful protected run. This is a real integration requirement:
[`RectangularTrainingStatus.add_generation_data`](../../src/glnis/utils/types.py)
currently retains full-window tensors directly, while partial windows copy into
buffers. A future compiled orchestrator must own retained tensors before another
compiled call can reuse their storage, and set iteration boundaries deliberately.
This prototype does not establish correctness for delayed or fragmented feedback
under CUDA graphs.

## Startup and batch shapes

With a fresh Inductor cache, the first `reduce-overhead` sampling call took
25.4 seconds, plus 0.9 seconds preparing the runtime. The first complete training
cycle took 71.5 seconds, plus 0.5 seconds preparing the runtime. These amortize
after approximately 5,650 sampling batches and 2,980 training cycles respectively,
using the savings measured in those runs. Other system/compiler caches were not
purged. The JSON results distinguish these runs from ones using existing caches.

Cached compilation can change the decision substantially. A probability/backward
training run using existing kernel caches took 3.3 seconds for its first call,
and approximately 190 steps to amortize startup. CPU sampling gained only 1.27×;
its initial first call took 37 seconds, while a subsequent cached run took
3.4 seconds. Neither a warm cache nor the measured break-even point is portable
across changed models, devices, shapes, versions, and modes.

For comparison, the Gaussian example trains for only 50 steps. Fresh compilation
costs much more than it saves in that run. Even the 100,000-sample inference case
needed roughly 800 batches to amortize its observed first-call cost. Count batches
in the intended deployment before enabling compilation.

Static compilation generated additional variants for batches of one and seven,
taking another 10.1 and 12.5 seconds respectively. With `dynamic=True`, batch seven
reused the graph compiled for 1,000, taking about 2 ms on first use; batch one
still specialized and took 11.7 seconds. Dynamic sampling averaged 1.45 ms versus
6.12 ms eager in that run. It is a useful candidate for fragmented requests, but
does not remove all specialization. The
[compile API](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)
describes dynamic-shape specialization and the CUDA-graph tradeoffs of
`reduce-overhead`, including extra memory use. GPU memory overhead was not profiled
in these experiments.

## Reproduce or extend the experiment

Use [the editable development environment](../development.md). The benchmark
requires working compiler tools as well as the Python packages. This host's CPU
compiler initially failed because `openssl` was absent from `PATH`; exposing the
installed executable resolved it. That failure was environmental, not a sampler
graph failure.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
TRITON_DEFAULT_FP_FUSION=1 .venv/bin/python benchmarks/torch_compile.py \
  --device cuda:0 --workload sample --mode reduce-overhead \
  --check-restore --probe-batch-sizes 1 7 1000 \
  --output /tmp/compile-sampling.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
TRITON_DEFAULT_FP_FUSION=1 .venv/bin/python benchmarks/torch_compile.py \
  --device cuda:0 --workload cycle --output /tmp/compile-training.json
```

Set `TORCHINDUCTOR_CACHE_DIR` to a new directory for a fresh Inductor-cache test.
JSON output includes runtime setup, first-call time, all steady-state repeats,
break-even calls, arguments, and correctness results. Compilation failures return
a nonzero exit code with the error recorded in JSON.

Useful variations are `--device cpu`, `--dtype float32`, `--batch-size 100000`,
`--target subnets`, `--context no-grad`, `--context eval-no-grad`, `--dynamic`,
and `--workload prob` or `train`. Use `--discrete 2 3 --condition 2 --continuous-first`
for multiple conditional discrete axes in the opposite
factorization order. `--discrete` with no values selects a pure continuous flow.
`--backend eager` checks graph capture without measuring Inductor performance.
For sampling/probability timings, `no-grad` uses training mode, `eval-no-grad`
uses evaluation mode, and `inference` uses evaluation plus inference mode.
Training workloads always use training mode and generate under `no_grad`.

## Correctness and adoption

The benchmark checks generated probabilities against eager probability queries.
For probability/training workloads it also compares probabilities and parameter
gradients at identical, updated parameters. f64 comparisons use `rtol=1e-8` and
`atol=1e-9`; f32 comparisons use `rtol=2e-4` and `atol=2e-5`. It does not demand
identical eager/compiled samples: their random streams and arithmetic can differ.

`--check-restore` checks bitwise module/RNG continuation within the same execution
mode, reconstructing the compiled callable after loading the raw module. This
passed for CPU Inductor and CUDA Inductor with both static CUDA-graph sampling
and dynamic sampling within the same process. Fresh-worker replay was not tested.
It is not a test of a compiled orchestrator's interrupted
feedback/optimizer recovery; no such orchestrator is implemented here.

[Compilation tests](../../tests/test_torch_compile.py) exercise graph capture,
conditional probabilities and gradients after parameter updates, both
factorization orders, pure domains, support changes without new graph shapes,
outside-spline gradients, and legacy module loading. The default tests use the
eager compiler backend; actual Inductor correctness is checked by the benchmark.
The existing checkpoint and feedback tests continue to run through the production
eager orchestrator.

For a future production option, keep compiled callables on the orchestrator,
outside its raw proposal and checkpoint data. Compile `prob()` for long training
runs and `sample()` for repeated generation; do not recompile explicitly after
each parameter update. Benchmark separate training and evaluation contexts. Keep
training generation under `no_grad`, since its tensors are later used by autograd;
reserve `eval()` plus `inference_mode()` for the completed training phase. See
[PyTorch's inference-mode restrictions](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.grad_mode.inference_mode.html).
Any explicit warmup must preserve RNG and training buffers. Application-level
timing, retained-output ownership, and interrupted-window recovery remain
necessary before adopting that runtime option.
