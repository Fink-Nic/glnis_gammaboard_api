# glNIS GammaBoard API

Neural importance sampling for GammaBoard.
See [the documentation](docs/README.md) for usage, configuration, implementation,
and development guidance.

## Python installation

Use Python 3.13. The core Python package requires no Rust toolchain, Symbolica
license, or GammaLoop installation:

```bash
uv venv --python python3.13 .venv
uv pip install --python .venv/bin/python -e '.[test]'
source .venv/bin/activate
```

Omit the `test` extra when pytest is not needed. GammaLoop state parsing requires
both the native companion and the `gammaloop` extra, plus a Rust toolchain and
native libraries. The full Nix development shell supplies those build dependencies:

```bash
uv pip install --python .venv/bin/python -e ./native/gammaloop -e '.[gammaloop,test]'
```

The companion is built separately with Maturin; the core wheel remains pure Python.
Optional-backend runtime setup is covered in the
[GammaBoard guide](docs/usage/gammaboard.md#gammaloop-inputs).

## Nix builds and editable development

The flake currently targets Linux x86_64. It pins Python and the scientific
runtime through [flake.lock](flake.lock).

```bash
nix build .#runtime            # core runtime; also the default package
nix build .#runtime-gammaloop  # runtime with the optional GammaLoop backend
nix build .#wheel              # pure Python wheel
nix build .#wheel-gammaloop     # native companion wheel
```

For an editable checkout:

```bash
nix develop
# Use `nix develop .#gammaloop` for the optional native backend.
glnis-dev
source .venv/bin/activate
```

Rerun `glnis-dev` from the selected shell after changing the Nix pin or Rust
sources. See [development](docs/development.md) for environment details,
dependency updates, and verification commands.

## Apptainer builds

Build from the repository root so the definition's local source paths resolve:

```bash
apptainer build --force glnis.sif apptainer.def
```

The definition installs Python packages directly, including prebuilt CUDA-enabled
PyTorch wheels. It requires neither Nix nor Docker on the host. The optional
GammaLoop backend is enabled with `--build-arg WITH_GAMMALOOP=1`.
On UBELIX, submit the build to a CPU compute node with:

```bash
python3 ubelix.py
# Optional native backend: python3 ubelix.py --with-gammaloop --time 06:00:00
```

See the [UBELIX build guide](ops/ubelix/README.md) for prerequisites, submission,
monitoring, and image deployment.

## Entry points

Activate the environment before invoking installed commands.

| Installed command | Python module equivalent | Purpose |
| --- | --- | --- |
| `glnis-gammaboard-sampler` | `python -u -m run_sampler` | GammaBoard sampler worker |
| `glnis-integrate` | `python -m glnis.scripts.run` | Local integration from TOML |
| `glnis-compare-samplers` | `python -m glnis.scripts.sampler_comparison` | Compare named sampler configurations |
| `glnis-slice-plot` | `python -m glnis.scripts.slice_plots` | Evaluate or replot saved PDF slices |

The Nix runtime's Python executable is `result/bin/python` after building the
selected runtime. `nix run` launches the core sampler; `nix run .#gammaloop`
launches the worker with the optional backend. Inside an Apptainer image, invoke
`glnis-gammaboard-sampler` through `apptainer exec`.

For arguments and workflows, see [GammaBoard usage](docs/usage/gammaboard.md),
[local runs](docs/usage/local-runs.md), and [analysis](docs/usage/analysis.md).
