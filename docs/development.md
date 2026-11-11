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
glnis-dev --gammaloop
source .venv/bin/activate
```

`glnis-dev [--gammaloop] [venv-directory]` creates/updates the editable environment
and records the selected native-library/compiler paths. From the core shell,
`--gammaloop` enters the full shell automatically. This supports worker launches
outside `nix develop`, where native library lookup would otherwise differ.
Ordinary editable pip/uv installation also works within the shells.

Python edits take effect immediately. Rerun `glnis-dev --gammaloop` after Rust
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

The repository-wide command includes `ops/ubelix/tests`. Use `pytest -q tests`
for the Python sampler/analysis suite alone. Run focused tests during development;
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
startup and steady-state timing. Compilation is not enabled in the default
orchestrator.

The [six-dimensional follow-up](implementation/compilation-6d.md) uses
`--continuous 6 --mlp-layers 4 --mlp-units 512`, with either `--discrete` or
`--discrete 3`. Benchmark JSON includes parameter counts, coupling-block counts,
and peak CUDA allocation during measurement.

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

The native namespace is `glnis_gammaloop_native._rust`. Older checkouts may contain
ignored `src/glnis/_rust*.so` binaries that shadow the compatibility facade;
current state parsing imports the companion directly. Explicit graph mappings
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
glnis-dev --gammaloop
```

Update the manifest and companion lockfile together. Maturin and Nix build with
`--locked`; the relevant lockfile is the companion's, not a Git dependency's
lockfile. Native dependencies use a separately pinned Rust toolchain in the flake.

Use `nix flake update <input-name>` to update Nix inputs, then refresh the editable
environment. Coordinate the Nix Python/PyTorch/CUDA pin with GammaBoard. Python
editable installs follow `pyproject.toml`; Nix builds use the sources and versions
selected by the flake. For pinned scientific wheels, update the wheel version,
Python ABI tag, and checksum together when changing the pin.

Keep the core version, companion version, and companion requirement in the
`gammaloop` extra aligned when releasing. The core remains a pure Python package;
do not include native build outputs or globally install the source checkout's
`sitecustomize.py`. Neural-network namespace-package discovery must continue to
include the files under `glnis/core/nn`.

## Runtime and HPC notes

[Runtime bootstrap](../src/glnis_runtime_bootstrap.py) configures native libraries,
compiler tools, and Triton compatibility for worker processes. Keep those concerns
in the bootstrap/runtime layer rather than adding environment mutations to
sampler math. Device and floating-point choices belong to the orchestrator.

Apptainer uses Nix inside its image build and exposes normal executables in the
resulting SIF. The [UBELIX guide](../ops/ubelix/README.md) is the canonical source
for Slurm resources, build caches/logging, GPFS handling, and image deployment.
See [GammaBoard usage](usage/gammaboard.md) for worker commands and license setup.

## Documentation changes

The [documentation index](README.md) separates workflows, option reference, and
implementation contracts. Keep the root README limited to building and entry
points. Update references alongside changed options or public contracts, and
link to canonical TOML examples instead of duplicating them.

For documentation-only edits, check local links/anchors, parse TOML/JSON snippets,
compile Python examples, and exercise relevant runnable examples where practical.
A full numerical suite is not required for prose or link changes alone.
