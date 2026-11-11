# Documentation

Start with the [project README](../README.md) to build or install the project.
These pages describe the current sampler interfaces and configuration schema.

## Run the project

- [GammaBoard](usage/gammaboard.md): worker commands, process arguments, paths,
  evaluator metadata, and deployment templates.
- [Local runs](usage/local-runs.md): TOML runs, raw feedback, the Python API,
  and saved histories.
- [Analysis](usage/analysis.md): comparisons, portable checkpoints, and PDF slices.

## Configure a run

Read the [configuration overview](configuration/README.md) first. It distinguishes
GammaBoard process arguments from the outer tables in local TOML files.

| Reference | Contents |
| --- | --- |
| [Sampler](configuration/sampler.md) | Adapter options and public-domain settings |
| [Orchestrators](configuration/orchestrators.md) | Training, devices, losses, schedulers |
| [Models](configuration/models.md) | Mixed flows, continuous/discrete flows, subnets, splines |
| [Mappings](configuration/mappings.md) | Coordinate layers, channel partitions, parser metadata, graph properties |
| [Local runs and analysis](configuration/local-runs.md) | Integrands, aggregation, comparisons, slice settings |

## Understand or extend the implementation

- [Architecture](implementation/architecture.md): responsibilities and source layout.
- [Sampling and feedback](implementation/sampling-and-feedback.md): array shapes,
  coordinate conventions, training windows, and strata.
- [Mappings](implementation/mappings.md): Jacobians, integration weights, and PDFs.
- [Checkpoints and stages](implementation/checkpoints-and-stages.md): exact recovery
  versus a fresh training run on a retained proposal.
- [Extending components](implementation/extending.md): registration and implementation contracts.

[Development](development.md) covers testing, editable environments, dependency
updates, and optional-backend troubleshooting. Project-specific agent guidance
is in [AGENTS.md](../AGENTS.md); HPC build instructions remain in the
[UBELIX guide](../ops/ubelix/README.md).

The current runnable examples are [ghostbumps](../examples/local/ghostbumps.toml),
[sine](../examples/local/sine.toml), [mapped Gaussian](../examples/local/gaussian.toml),
and [sampler comparison](../examples/local/sampler_comparison.toml). Older sign-strategy
examples and deployment templates need migration; their presence does not imply
that the retired sampler implementations are available.
