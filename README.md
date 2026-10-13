# glNIS GammaBoard API

MadNIS sampler implementation for GammaBoard using the
`gammaboard_process.run_sampler(...)` Python wrapper. Includes support for additional coordinate mappings schemes that allow the direct sampling of momentum space.

## Runtime Options

Supply your own Symbolica license in the environment used to launch GammaBoard
or the sampler, including for editable development and Apptainer:

```bash
export SYMBOLICA_LICENSE='your-license-key'
```

### Nix

Build the project environment:

```bash
nix build .#runtime
```

This builds a locked runtime with the Rust dependency artifacts cached separately
from the Python package. `nix run` also
launches the sampler, and `nix build .#wheel` builds the Nix-native Python wheel.

Use this GammaBoard process command:

```toml
command = ["$resources/../../glnis_gammaboard_api/result/bin/python", "-u", "-m", "run_sampler"]
cwd = "$resources/.."
```

### Apptainer

Apptainer/SIF is the portable runtime path for HPC systems that do not provide
Nix. The definition uses Nix only inside the image build container to reproduce
`nix build .#runtime`; the final SIF exposes normal executables and does not
require Nix on the host.

Build from this directory so the `%files` entries in `apptainer.def` resolve to
the local checkout:

```bash
apptainer build --force glnis.sif apptainer.def
```

Use this GammaBoard process command:

```toml
command = ["apptainer", "exec", "--no-mount", "/etc/localtime", "--nv", "--bind", "$resources/..:$resources/..", "$resources/../../glnis_gammaboard_api/glnis.sif", "glnis_gammaboard_sampler"]
cwd = "$resources/.."
```

### Editable development

Enter the development shell and install the checkout in editable mode:

```bash
nix develop
glnis-dev
source .venv/bin/activate
```

The shell provides Python 3.13, Rust, and native dependencies without building
the runtime. `glnis-dev [venv-directory]` creates or updates the venv and records
the selected native library/compiler paths so its sampler can also run outside
the shell. Python edits take effect immediately. Rerun `glnis-dev` after Rust or
dependency changes; Cargo reuses the local `target` directory. Editable builds
use the development profile; packaged Nix builds use the release profile.

After updating `flake.lock`, rerun `glnis-dev`. It also refreshes the venv's base
interpreter in place when the Nix Python changes, keeping Python, glibc, and the
Rust extension compatible while preserving installed packages.

An ordinary editable install is also supported inside the shell, or on a Linux
host with Python 3.13, Rust, a C/C++ compiler, and the required native libraries:

```bash
uv venv --python python3.13 .venv
uv pip install --python .venv/bin/python -e .
```

On Nix hosts, use `glnis-dev` when the venv must run outside `nix develop`.

Use this GammaBoard process command:

```toml
command = ["$resources/../../glnis_gammaboard_api/.venv/bin/glnis-gammaboard-sampler"]
cwd = "$resources/.."
```

With `cwd = "$resources/.."`, sampler `save_path` values should be relative to
the GammaBoard workspace, for example:

```toml
save_path = "integrations/glnis_gammaboard_api/checkpoints/ghost_bump_madnis"
```

### Updating dependencies

Change a Rust dependency version or Git revision in `Cargo.toml`, then update the
lockfile using Cargo, for example:

```bash
nix develop
cargo update -p gammalooprs
glnis-dev                 # incremental editable build
# Or build the deployable runtime:
nix build .#runtime
```

Commit `Cargo.toml` and `Cargo.lock` together. Maturin and Nix both build with
`--locked`. The GammaLoop revision is also used for its embedded assets, with no
separate source hash to maintain. Cargo uses this project's lockfile, not the
lockfile inside a Git dependency. For GammaLoop `40616e5`, keep Rust Symbolica
and its transitive Numerica dependency locked to 3.0.0: Numerica 3.0.1 adds a
required method that this GammaLoop revision does not implement. If a broader
Cargo update changes those versions, restore them with:

```bash
cargo update -p symbolica --precise 3.0.0
cargo update -p numerica --precise 3.0.0
```

Use `nix flake update <input-name>` to update a Nix input and commit `flake.lock`.
The editable Python dependencies use the sources declared in `pyproject.toml`;
the Nix runtime uses the source revisions in `flake.lock` and Nixpkgs packages.
Both the Python Symbolica wheel and the Rust Symbolica crate are pinned to
3.0.0. When updating the Python wheel, update its version in `pyproject.toml`
and the version, Python ABI tag, and checksum in `flake.nix` using the release
files published on PyPI.

Changing the Rust dependency set can rebuild Nix's dependency derivation. Local
editable builds retain Cargo's incremental cache across dependency updates.


## Use With GammaBoard

`examples/ghost_bump_madnis.toml` is a ready-to-copy run template. It uses the
nix build command by default.

The sampler command uses `$resources/..` because GammaBoard expands
`$resources` to the default resource directory. Sampler `args` are passed
through unchanged, so `save_path` should be relative to the configured process
`cwd`.

The process entrypoint is:

```bash
python -u -m run_sampler
```
