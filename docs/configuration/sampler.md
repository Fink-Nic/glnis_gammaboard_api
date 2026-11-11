# Sampler adapter

These options belong to the local `[sampler]` table, or directly to a GammaBoard
process `args` object. Source:
[GammaBoardSamplerConfig and GammaBoardSampler](../../src/gammaboard_sampler.py).

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `kind` | string | `default` | Adapter implementation; currently only `default` is accepted |
| `save_path` | path string or Python `None` | `None` | Destination of the binary checkpoint; required when saving |
| `module` | table | `{}` | Proposal-module configuration; defaults to a mixed flow |
| `orchestrator` | table | `{}` | Generation/training configuration; defaults to rectangular orchestration |
| `mapping` | table or Python `None` | `None` | Optional layered coordinate mapping |
| `parser` | table or Python `None` | `None` | Metadata-parser settings |
| `graph_properties` | table or Python `None` | `None` | Explicit graph data used by mapping layers |
| `evaluator_metadata` | table or Python `None` | `None` | Evaluator context, including optional GammaLoop state identifiers |

`kind` is accepted by the constructor, rather than stored as a field of
`GammaBoardSamplerConfig`. Python callers may pass a `cfg` instance instead of
individual settings. `snapshot` and `fresh_training` are lifecycle construction
arguments used by the restore/stage factories, not normal run-file options.

## Domain and coordinate conventions

GammaBoard provides `domain`. A local runner derives it from its integrand;
there is no independent sampler-dimension option in local TOML.

The implementation requires a rectangular shape. It retains singleton discrete
axes in public samples, removes them internally, and appends any hidden mapping
channels to the internal proposal domain. The mapping output dimension must
match the public continuous dimension.

`save_path` and the top-level proposal `module` table are passed to the
orchestrator as `save_path` and `module_kwargs`. Configure them here rather than
duplicating those adapter-owned values inside `[sampler.orchestrator]`.

Use [module configuration](modules.md), [orchestrator configuration](orchestrators.md),
and [mapping configuration](mappings.md) for their nested options. For array
shapes and feedback ownership, read [sampling and feedback](../implementation/sampling-and-feedback.md).

## Compatibility

Current run files use `sampler.module`; direct Python/GammaBoard arguments use
`module`. The old proposal `sampler` key and orchestrator `sampler_kwargs` key
are migrated when reading checkpoint metadata. New snapshots write only `module`
and `module_kwargs`. Set proposal options in the adapter's `module` table;
duplicating them in `orchestrator.module_kwargs` is rejected.

The current adapter is `GammaBoardSampler`. Older `MadnisSampler` classes,
`madnis` adapter spelling, and `sgn_strat_naive`/`sgn_strat_full` kinds are not
available. A new stratum-specific implementation belongs behind an orchestrator
`kind`; see [extending components](../implementation/extending.md#orchestrators-and-strata).
