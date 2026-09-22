# glNIS GammaBoard API

MadNIS sampler implementation for GammaBoard using the
`gammaboard_process.run_sampler(...)` Python wrapper. Includes support for additional coordinate mappings schemes that allow the direct sampling of momentum space.

## Runtime Options

### Nix

Build the project environment:

```bash
nix build .#runtime
```

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
command = ["apptainer", "exec", "--no-mount", "/etc/localtime", "--nv", "--bind", "$resources/..:$resources/..", "$resources/../integrations/glnis_gammaboard_api/glnis.sif", "glnis_gammaboard_sampler"]
cwd = "$resources/.."
```

### Direct venv 

Install in editable mode for convenient python development without rebuilding the project:

```bash
uv venv --python 3.13 --seed .venv
uv pip install -e .
source .venv/bin/activate
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
