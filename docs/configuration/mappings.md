# Coordinate mapping configuration

Source: [mappings.py](../../src/glnis/core/mappings.py),
[MetaDataParser](../../src/glnis/core/parser.py), and
[GraphProperties](../../src/glnis/utils/types.py).
Mappings belong under `sampler.mapping` in local TOML.

## Layer stack

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `input_space` | `HCUBE` or `MOMENTUM` | `HCUBE` | Coordinate space entering the first layer |
| `layer_cfgs` | nonempty list of tables | required | Ordered layers, each with a `kind` |
| `use_f128` | boolean | `false` | Use NumPy float128 for mapping calculations |

Each layer receives the previous layer's output space. The ordinary `spherical`,
`kaapo`, and `reduced kaapo` mappings switch spaces; `identity` and `s` preserve
space. Not every mapping implements both directions. Dimension-changing layers
must produce the adapter's public continuous dimension.

The [Gaussian example](../../examples/local/gaussian.toml) is a complete spherical
mapping configuration with explicit graph data. Layers that sample a channel add
hidden discrete coordinates to the proposal; the evaluator never sees them.

## Layer options

The following are concrete registered mapping kinds. Constructor defaults are
listed here; registration alone does not imply every option combination has a
valid inverse or requires only core dependencies.

| `kind` | Options and defaults | Notes |
| --- | --- | --- |
| `spherical` | `conformal_scale=1.0` | Spherical/rational radial coordinates; scale is additionally multiplied by positive `e_cm` |
| `s` | `exponent=2.0` | Hypercube smoothing; exponent is clamped to at least 1 |
| `identity` | `seed=42`, `uniform_continuous=false`, `force_dim=None` | Preserves coordinates and has unit Jacobians in the ordinary mode |
| `kaapo` | `mu=pi`, `a=0.5`, `b=1.0`, `vary_a=false`, `a_min=0.2`, `angle_shift=0.0` | Forward momentum parameterization; no analytic inverse is implemented |
| `reduced kaapo` | Same options as `kaapo` | Reduced angular coordinates; verify the target's symmetry assumptions |
| `ose` | `ose_exponent=1.0`, `mapping_kwargs={kind="spherical"}` | Multichannel wrapper with an on-shell-energy partition |
| `fermi` | `ose_exponent=4.0`, `fermi_exponent=1.0`, `set_bosonic_edge_to_one=true`, `mapping_kwargs={kind="spherical"}` | Multichannel wrapper with energy/Fermi-surface factors |
| `jac` | `jac_exponent=1.0`, `mapping_kwargs={kind="spherical"}` | Multichannel partition based on the wrapped transform's inverse Jacobian |
| `madspace` | `diagram="epem_a_ttxh_LO"`, `return_full_phase_space=false` | Experimental external phase-space mapping; requires a separately available `madspace` package |

For `identity`, `force_dim=None` inherits the current width. The
`uniform_continuous=true` option redraws coordinates using a seeded NumPy RNG;
it is a stochastic option, not an invertible identity transform.

For the Kaapo mappings, `mu` can be a scalar or a list of per-edge values.
`vary_a` consumes an extra continuous coordinate. Use a positive `b`: the
current `b=0` fallback is not supported. The reduced mapping uses one continuous
coordinate for one loop, or `3 * (n_loops - 1)` for more loops, plus the optional
coordinate for `vary_a`.

The multichannel `mapping_kwargs` table configures the wrapped transform and
requires its own `kind` when supplied. Each wrapper's channel cardinality follows
`GraphProperties.n_channels`. Partition factors are separate from coordinate
Jacobians; see [the mapping contract](../implementation/mappings.md).

`madspace` also recognizes `diagram="epem_a_ddx_LO"`. Its current implementation
forces `return_full_phase_space=false` even if true is supplied; it has no inverse
implementation. Neither the core package nor the GammaLoop extra installs
Madspace. The commented Momtrop adapter is not registered or enabled.

## Explicit graph properties

Supply a `sampler.graph_properties` table when no GammaLoop state parser is used.
It takes precedence over evaluator metadata when constructing mappings.

| Field | Default | Meaning |
| --- | --- | --- |
| `edge_masses` | required | One mass per internal edge |
| `edge_momentum_shifts` | required | One three-vector per internal edge |
| `graph_signature` | `[]` | Edge-by-loop momentum-basis coefficients; needed for usable mapping geometry |
| `generation_lmb_edges` | `[]` | Edge indices defining the generation loop-momentum basis |
| `lmb_edges` | `[]` | Edge-index matrix defining the available channel bases |
| `e_cm` | `0.0` | Center-of-mass energy; positive values rescale spherical coordinates |
| `graph_external_vertices` | `[]` | External-vertex identifiers |
| `edge_src_dst_vertices` | `[]` | Graph connectivity |
| `edge_external_sigs` | `[]` | External-momentum coefficients |
| `external_momenta` | `[]` | External momenta |
| `orientation_ids` | `[]` | Orientation identifiers |
| `orientation_signatures` | `[]` | Signatures corresponding to the orientation IDs |

Shapes, channel bases, and generation basis must describe the same graph.
Orientation IDs and signatures must have equal lengths. Internal channel
transforms and inverses are derived, not input settings. Empty defaults allow
staged parser construction; they are not a complete graph for numerical mapping.

## Evaluator metadata and parser

For GammaLoop-derived geometry, use evaluator metadata with these fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `kind` | required | `gammaloop` |
| `state_folder` | required | GammaLoop state path |
| `process_id` | required | Nonnegative process identifier |
| `integrand_name` | required | Integrand identifier |
| `momentum_space` | `false` | Descriptive metadata retained by the adapter |
| `coordinate_space` | `x_space` | Descriptive coordinate-space string |
| `domain_axes` | `[]` | Descriptive axis names |

The parser uses the state path, process ID, and integrand name to resolve graph
properties. The descriptive fields do not automatically choose coordinate layers.
The optional native companion, pydot, and symbolic backend are required for state
parsing; explicit graph data avoid these imports.

`sampler.parser` accepts `override_lmb_heuristics=false` and
`gammaloop_directory=""`. These fields are retained in the parser configuration,
but the current metadata-to-graph-properties path does not consult them. The
state source is `evaluator_metadata.state_folder`.

The parser can return multiple graph groups, while the current numerical layer
stack expects a single `GraphProperties` object. Automatic selection or
composition of multiple graph groups is not implemented.
