# glNIS GammaBoard API

MadNIS sampler implementation for GammaBoard using the
`gammaboard_process.run_sampler(...)` Python wrapper. Includes support for parameterisation schemes.

## Runtime Options

### Direct venv

For local demos or machines where Apptainer is not available, install the
sampler directly into a virtual environment under this integration directory:

```bash
cd ~/gammaboard/integrations/madnis

uv venv --python 3.13 --seed .venv
. .venv/bin/activate
python -m pip install .
```

Use this GammaBoard process command:

```toml
command = ["$resources/../integrations/madnis/.venv/bin/madnis-gammaboard-sampler"]
cwd = "$resources/.."
```

With `cwd = "$resources/.."`, sampler `save_path` values should be relative to
the GammaBoard workspace, for example:

```toml
save_path = "integrations/madnis/checkpoints/ghost_bump_madnis"
```

### Apptainer

Apptainer is the most portable path for UBELIX and other non-Nix systems:

```bash
apptainer build --force madnis.sif apptainer.def
```

The definition file builds from Git, not from the local checkout. Pin the exact
source when needed:

```bash
GAMMABOARD_REF=<branch-or-commit> apptainer build --force madnis.sif apptainer.def
```

On UBELIX, run the build from the GammaBoard workspace:

```bash
python ubelix.py build apptainer integrations/madnis/madnis.sif integrations/madnis/apptainer.def
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
