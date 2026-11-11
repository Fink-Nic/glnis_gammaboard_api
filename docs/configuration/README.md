# Configuration overview

The sampler accepts dictionaries, and local runners read equivalent TOML tables.
Defaults and validation come from the configuration classes linked on each
reference page. Omit optional values in TOML where the Python default is `None`;
TOML has no null literal.

## Two nesting conventions

GammaBoard's sampler process `args` object is passed directly to
`GammaBoardSampler(domain=..., evaluator_metadata=..., **args)`. Its `sampler`
key selects the proposal module and its `orchestrator` key selects orchestration.
GammaBoard supplies the domain separately.

A local run adds an outer `[sampler]` table containing those same adapter
arguments, alongside `[integrand]` and `[aggregator]`:

```text
sampler                         GammaBoardSampler arguments
  sampler                       proposal module
    flow                        continuous flow
      subnet, spline            coupling-network and spline configuration
    discrete                    discrete proposal
      subnet                    transformer output network
  orchestrator                  generation and training
    loss, scheduler             training components
  mapping, parser,
  graph_properties,
  evaluator_metadata            optional coordinate mapping inputs
integrand                       local numerical integrand
aggregator                      local execution and evaluation settings
```

The reference pages use local TOML paths such as `sampler.sampler.flow`.
Remove only the outer `sampler` level when constructing a process `args` object
or passing keyword arguments directly to the Python sampler.

## Selecting implementations

A component table's `kind` selects a registered implementation. These are the
current defaults for the sampler stack:

| Table | Default `kind` | Reference |
| --- | --- | --- |
| `sampler.orchestrator` | `rectangular` | [Orchestrators](orchestrators.md) |
| `sampler.sampler` | `mixed flow` | [Models](models.md) |
| `sampler.sampler.flow` | `coupling` | [Models](models.md#continuous-flow) |
| `sampler.sampler.flow.spline` | `rqs` | [Models](models.md#spline) |
| Flow/transformer `subnet` | `mlp` | [Models](models.md#subnets) |
| `sampler.sampler.discrete` | `transformer` | [Models](models.md#discrete-flow) |
| `sampler.orchestrator.loss` | `kl_divergence` | [Losses](orchestrators.md#losses) |
| `sampler.orchestrator.scheduler` | `cosineannealing` | [Schedulers](orchestrators.md#schedulers) |

Mapping layers require an explicit `kind`. The adapter itself accepts only
`kind="default"`; this is distinct from the proposal module's `kind`.
The local integrand factory requires `kind` and uses its own explicit registry.
New sampler components must be imported before `BuildFromKind` can discover them.
See [extending components](../implementation/extending.md).

## Paths, overrides, and validation

Local TOML `sampler.save_path`, `sampler.evaluator_metadata.state_folder`, and
`sampler.parser.gammaloop_directory` resolve relative to the TOML file. Comparison
and slice output/input paths also resolve relative to their settings file.
Direct Python construction and GammaBoard process arguments do not receive this
local-file path normalization.

Comparison defaults merge nested tables recursively; other values, including
lists of mapping layers, replace their defaults. Changing a `kind` in comparison
settings does not automatically discard options inherited from that table, so
ensure the merged options belong to the selected implementation.

Checkpoint/stage overrides use the sampler's merge policy: changing an explicit
component `kind` replaces that component table. Exact recovery permits only
[runtime/output overrides](../implementation/checkpoints-and-stages.md#exact-recovery).

The local configuration loader rejects unknown outer settings and validates
known component tables. Some dimension-dependent or constructor-specific checks
occur when building the model. Parsing a table successfully does not establish
statistical validity for an experimental component.

Use the [checked-in local examples](../../examples/local/) for complete current
configurations. Retired flat options such as `sampler.flow_config` and
`sampler.training_steps` are not the current schema.
