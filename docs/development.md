# Development and verification

Build/install commands are in the [project README](../README.md). Repository
editing conventions are in [AGENTS.md](../AGENTS.md). Start with the
[architecture](implementation/architecture.md) before changing component ownership
or numerical contracts.

## Editable environments

For core Python development:

```bash
nix develop
glnis-dev
source .venv/bin/activate
```

For the optional GammaLoop backend:

```bash
nix develop .#gammaloop
glnis-dev
source .venv/bin/activate
```

`glnis-dev [venv-directory]` creates/updates an editable environment using the
selected shell's backend. Its Python launcher supplies the Nix shared-library
and compiler paths before starting Python, including for workers launched outside
the shell. A `.nix-runtime` GC root retains those paths and the interpreter.
It does not install a Python startup hook or runtime JSON configuration.
Recreate environments produced by the previous helper once to discard their
startup hooks. Ordinary editable pip/uv installation also works inside the shells.

Python edits take effect immediately. Rerun `glnis-dev` from the GammaLoop shell after Rust
changes; Cargo reuses `native/gammaloop/target`. Editable native builds use the
development profile; packaged Nix builds use release. Rerun `glnis-dev` after
updating the Nix pin so the environment uses the matching Python/native runtime.

Use separate environment directories if both core and full profiles are needed.
Installing the core into an existing full environment does not uninstall its
optional dependencies.

## Tests

With the existing `.venv`, a deterministic CPU-oriented invocation is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
```

The command includes the deployment submission/publication checks in
`tests/test_deployment.py`. Run focused tests during development;
use the full suite for changes spanning sampler contracts.

| Change area | Useful tests |
| --- | --- |
| Public generation, domains, feedback, masses | [test_gammaboard_sampler.py](../tests/test_gammaboard_sampler.py) |
| Conditioning and sample/probability identities | [test_flow_modules.py](../tests/test_flow_modules.py) |
| Mapping geometry, partitions, raw feedback, PDFs | [test_mapped_sampler.py](../tests/test_mapped_sampler.py), [test_layered_mapping.py](../tests/test_layered_mapping.py) |
| Loss gradients and kind-selected components | [test_losses_and_schedulers.py](../tests/test_losses_and_schedulers.py) |
| Interrupted restore and fresh stages | [test_sampler_checkpoints.py](../tests/test_sampler_checkpoints.py) |
| Local configuration and execution | [test_aggregator_config.py](../tests/test_aggregator_config.py), [test_sampler_aggregator.py](../tests/test_sampler_aggregator.py) |
| Comparison/slice behavior | [test_sampler_comparison.py](../tests/test_sampler_comparison.py), [test_slice_plots.py](../tests/test_slice_plots.py) |
| Optional-backend isolation and packaging | [test_optional_backends.py](../tests/test_optional_backends.py), [test_optional_packaging.py](../tests/test_optional_packaging.py) |

The manual [Gaussian example](../tests/gaussian_test.py) and
[mapped Gaussian example](../tests/gaussian_test_glnis_mapping.py) remain useful
smoke runs. They write `saved_state.pkl` and `snapshot.json` in their current
working directory and may reuse `prepared_state.pkl`. Run them in temporary
working directories with the checkout environment's Python to avoid touching
existing checkpoint files.

Tests should exercise probability identities after parameters have changed,
not only the initially uniform proposal. Keep test artifacts in temporary paths.
Report optional skips and the device/precision actually exercised; a CPU run
is not CUDA validation.

For performance experiments, [benchmarks/torch_compile.py](../benchmarks/torch_compile.py)
compares eager and compiled sampling, probabilities, and training paths. Read
[the compilation report](implementation/compilation.md) for measured results,
reproduction commands, correctness checks, and the distinction between compiler
startup and steady-state timing. The orchestrator defaults to eager; optional
[execution settings](configuration/orchestrators.md#execution-and-compilation)
enable compilation.

The [six-dimensional follow-up](implementation/compilation-6d.md) uses
`--continuous 6 --mlp-layers 4 --mlp-units 512`, with either `--discrete` or
`--discrete 3`. Benchmark JSON includes parameter counts, coupling-block counts,
and peak CUDA allocation during measurement.

[benchmarks/orchestrator_compile.py](../benchmarks/orchestrator_compile.py)
repeats that workload through the native execution policy and public adapter,
including interrupted training-window recovery in separate workers. See the
[native report](implementation/compilation-native-6d.md) for complete timings
and bitwise versus numerical recovery checks.

[Execution tests](../tests/test_execution.py) use the eager compiler backend by
default. To exercise real Inductor compilation and restore pending feedback in a
fresh process on CPU, set `GLNIS_TEST_COMPILE_BACKEND=inductor`. For CUDA graphs:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
GLNIS_TEST_COMPILE_BACKEND=inductor GLNIS_TEST_COMPILE_DEVICE=cuda:0 \
GLNIS_TEST_COMPILE_MODE=reduce-overhead .venv/bin/python -m pytest -q \
  tests/test_execution.py::test_compiled_checkpoint_in_fresh_process
```

The check compares subsequent draws, trained parameters, optimizer and scheduler
states exactly within the same execution/device setup. Eager/compiled switching
is a runtime override with different numerical/RNG behavior. Fresh code generation
can also change roundoff; the native benchmark records exact and numerical
agreement separately instead of weakening the execution test. The test fixture
clears compiler caches between tests so unrelated architectures do not exhaust
the per-function variant limit.

## Optional backend checks

Core checks must work without compiling the companion or importing Symbolica:

```bash
.venv/bin/python -m pytest -q tests/test_optional_backends.py tests/test_optional_packaging.py
```

With the full environment installed, require the backend explicitly:

```bash
GLNIS_REQUIRE_GAMMALOOP=1 .venv/bin/python -m pytest -q tests/test_rust_extension.py tests/test_rust_state_parser.py
```

State integration additionally needs real external inputs, supplied through
`GLNIS_STATE_FOLDER`, `GLNIS_PROCESS_ID`, and `GLNIS_INTEGRAND_NAME`; see
[test_rust_state_parser.py](../tests/test_rust_state_parser.py). Missing optional
backends/fixtures can skip tests. Failures inside an installed backend, such as
binary loading or license errors, should propagate rather than be treated as
an absent feature.

The native namespace is `glnis_gammaloop_native._rust`; state parsing imports
the companion directly. Explicit graph mappings
remain core-only and do not need state parsing.

## Updating dependencies

Dependency declarations are authoritative in [pyproject.toml](../pyproject.toml),
[the companion project](../native/gammaloop/pyproject.toml),
[Cargo.toml](../native/gammaloop/Cargo.toml), [Cargo.lock](../native/gammaloop/Cargo.lock),
and [flake.nix](../flake.nix)/[flake.lock](../flake.lock). Avoid copying revision
numbers into prose that will become stale.

For Rust changes:

```bash
nix develop .#gammaloop
cargo update --manifest-path native/gammaloop/Cargo.toml -p gammalooprs
glnis-dev
```

Update the manifest and companion lockfile together. Maturin and Nix build with
`--locked`; the relevant lockfile is the companion's, not a Git dependency's
lockfile. The flake's single nixpkgs pin supplies the native Rust toolchain too.

Use `nix flake update <input-name>` to update Nix inputs, then refresh the editable
environment. Coordinate the Nix Python/PyTorch/CUDA pin with GammaBoard. Python
editable installs follow `pyproject.toml`; Nix builds use the sources and versions
selected by the flake. For pinned scientific wheels, update the wheel version,
Python ABI tag, and checksum together when changing the pin.

The fully Nix-packaged GPU runtime may build CUDA dependencies such as NCCL from
source. NVSHMEM's GPU test and example programs are disabled in this runtime;
its library features remain enabled. Optional binary caches are described in the
[NixOS CUDA guide](https://wiki.nixos.org/wiki/CUDA#Setting_up_CUDA_Binary_Cache);
the host administrator controls which caches are trusted. The editable venv and
Apptainer profiles instead install prebuilt PyTorch wheels.

Keep the core version, companion version, and companion requirement in the
`gammaloop` extra aligned when releasing. The core remains a pure Python package;
do not include native build outputs or global Python startup hooks.
Neural-network namespace-package discovery must continue to
include the files under `glnis/core/nn`.

## Runtime and HPC notes

Nix launchers provide `libstdc++`/zlib for pip wheels, adding GMP/MPFR/MPC only for
GammaLoop. They put the C/C++ compiler and OpenSSL executable on `PATH` for
Inductor, preserving caller `CC`/`CXX` choices. On NixOS they also point Triton at
`/run/opengl-driver/lib`, avoiding its `/sbin/ldconfig` probe on
that non-FHS host. Ordinary Python installations and Apptainer images use their
platform's library loader and tools. Application imports do not rewrite paths,
restart Python, or patch `ctypes`/`subprocess` globally. Device and precision
choices belong to the orchestrator.

The Nix package pin supplies PyTorch 2.13 with CUDA 13.0, which requires a
compatible host NVIDIA driver. The Apptainer profile uses its own wheel choice;
it does not inherit Nix's CUDA libraries.

Use [ops/check_runtime.py](../ops/check_runtime.py) with `.venv/bin/python` or
`result/bin/python` to check training, densities and checkpoint recovery. It runs
on CPU by default; `--device cuda:0 --compile` checks an allocated GPU with
Inductor. The packaged Nix runtime passed both CPU and compiled GPU checks on a
local NVIDIA A30. The optional native backend and UBELIX validation remain separate.

The Inductor execution policy sets `TRITON_DEFAULT_FP_FUSION=1` when unset,
preserving the measured arithmetic and avoiding Triton's changing-default warning.
An explicit caller value is retained. No thread count is imposed at import time.

A cold driver build with the tested pip stack (Triton 3.9 and glibc 2.44) also
emits a `_POSIX_C_SOURCE` redefinition warning from the CUDA/Python header order.
The driver build and compiled GPU checks pass. This upstream header warning is
left visible; the application does not patch installed Triton sources.

GPU selection tests cover visible-device ordering, occupied devices, free-memory
fallbacks, explicit indices, and CPU selection without requiring actual GPUs.

Apptainer installs prebuilt Python/CUDA packages directly, with an optional
Rust build for GammaLoop state parsing. The [UBELIX guide](../ops/ubelix/README.md)
covers Slurm submission, scratch/storage, GPU bindings, and the first-cluster
validation checklist. There is no Nix installation inside the image.
See [GammaBoard usage](usage/gammaboard.md) for worker commands and license setup.

## Documentation changes

The [documentation index](README.md) separates workflows, option reference, and
implementation contracts. Keep the root README limited to building and entry
points. Update references alongside changed options or public contracts, and
link to canonical TOML examples instead of duplicating them.

For documentation-only edits, check local links/anchors, parse TOML/JSON snippets,
compile Python examples, and exercise relevant runnable examples where practical.
A full numerical suite is not required for prose or link changes alone.
