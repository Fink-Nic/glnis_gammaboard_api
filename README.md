# glNIS GammaBoard API

MadNIS sampler implementation for GammaBoard using the
`gammaboard_process.run_sampler(...)` Python wrapper. Includes support for parameterisation schemes.

## Runtime Options

### Direct venv

For local demos or machines where Apptainer is not available, install the
sampler directly into a virtual environment under this integration directory:

```bash
uv venv --python 3.13 --seed .venv
. .venv/bin/activate
python -m pip install .
```

Use this GammaBoard process command:

```toml
command = ["$resources/../integrations/glnis_gammaboard_api/.venv/bin/glnis-gammaboard-sampler"]
cwd = "$resources/.."
```

With `cwd = "$resources/.."`, sampler `save_path` values should be relative to
the GammaBoard workspace, for example:

```toml
save_path = "integrations/glnis_gammaboard_api/checkpoints/ghost_bump_madnis"
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
command = ["apptainer", "exec", "--no-mount", "/etc/localtime", "--nv", "--bind", "$resources/..:$resources/..", "$resources/../integrations/glnis_gammaboard_api/glnis.sif", "glnis_gammaboard_sampler"]
cwd = "$resources/.."
```

Nix is still supported where available:

```bash
nix build .#runtime
```

## Use With GammaBoard

`examples/ghost_bump_madnis.toml` is a ready-to-copy run template. It uses the
direct venv command by default and keeps Apptainer and Nix alternatives
commented next to it.

The sampler command uses `$resources/..` because GammaBoard expands
`$resources` to the default resource directory. Sampler `args` are passed
through unchanged, so `save_path` should be relative to the configured process
`cwd`.

The process entrypoint is:

```bash
python -u -m run_sampler
```
