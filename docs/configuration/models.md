# Proposal models

These paths assume local TOML nesting. Direct sampler construction uses the
contents of `sampler.sampler` as its `sampler` argument. Dimensions and external
conditioning widths are supplied by the caller/domain, not repeated in these tables.

## Mixed flow

Source: [MixedFlowConfig](../../src/glnis/core/nn/sampler_module.py).

| Option under `sampler.sampler` | Type | Default | Meaning |
| --- | --- | --- | --- |
| `kind` | string | `mixed flow` | Registered rectangular sampler module |
| `discrete_dims_first` | boolean | `true` | Factorize as discrete proposal followed by a conditionally continuous proposal |
| `flow` | table | `{}` | Continuous flow |
| `discrete` | table | `{}` | Discrete flow, omitted internally if no discrete axes are sampled |

With `discrete_dims_first=false`, continuous samples condition the discrete
proposal. Exact discrete marginal masses are then unavailable when continuous
dimensions are present. Purely continuous and purely discrete domains are supported.

## Continuous flow

Source: [FlowConfig](../../src/glnis/core/nn/flow/continuous_flow.py).

| Option under `sampler.sampler.flow` | Type | Default | Meaning / constraint |
| --- | --- | --- | --- |
| `kind` | string | `coupling` | Coupling-block flow |
| `permutations` | `log`, `random`, or `exchange` | `log` | Strategy for selecting conditioning coordinates |
| `condition_masks` | boolean matrix or Python `None` | `None` | Explicit `(blocks, continuous_dims)` masks; true coordinates condition transformed false coordinates |
| `blocks` | integer or Python `None` | `None` | Required when `permutations` is `random` or `exchange` |
| `spline` | table | `{}` | Coupling transformation |
| `subnet` | table | `{}` | Coupling network |

Explicit masks override mask generation. One-dimensional flows use a dummy
regular input for their subnets; zero-dimensional continuous flows are identities.
The latent distribution is uniform on the unit cube. The retired
`uniform_latent` Gaussian-latent option is not part of this configuration.

## Discrete flow

Source: [TransformerConfig](../../src/glnis/core/nn/flow/discrete_flow.py).

| Option under `sampler.sampler.discrete` | Type | Default | Meaning |
| --- | --- | --- | --- |
| `kind` | string | `transformer` | Autoregressive discrete transformer |
| `embedding_dim` | integer | `64` | Embedding width; must be compatible with `heads` |
| `feedforward_dim` | integer | `256` | Transformer feed-forward width |
| `heads` | integer | `8` | Attention heads |
| `layers` | integer | `3` | Transformer encoder layers |
| `subnet` | table | `{kind="mlp", layers=3, units=256}` | Output network |

The model's cardinality mask limits each discrete coordinate to its own allowed
values. External conditions are embedded into the transformer and appended as
conditioning features to the output subnet. Its sampling and probability paths
use the same autoregressive prefix convention.

`prob(x, c)` requires complete discrete vectors of shape
`(batch, num_discrete_dims)`. Prefixes and holes marked `-1` belong to
`discrete_mass`, which evaluates their exact masses eagerly. Full-point scoring,
prefix scoring, and sampling share the cardinality mask and log-probability
normalization. Mixed-flow `prob` likewise requires all packed internal coordinates.

## Subnets

The active source is [flow/subnet_constructor.py](../../src/glnis/core/nn/flow/subnet_constructor.py).
Both continuous and discrete subnet tables use this factory.

| `kind` | Options | Behavior |
| --- | --- | --- |
| `mlp` | `layers=3`, `units=128` | Fully connected network; regular and conditional features are combined at the input |
| `mlp fully conditioned` | `layers=3`, `units=128` | Appends the original condition to each hidden layer |
| `stacked mlp` | required `channels`, `layers=3`, `units=128` | Independent channel networks; requires compatible caller-side batching/channel routing |

The transform supplies `features_in`, `features_c`, and `features_out`.
Conditioning features occupy the final input axis positions. The regular MLP
uses `features_in + features_c` as its input width. Callable activation/layer
constructors are Python extension points, not TOML values.

Selecting `stacked mlp` does not create a stratum orchestrator or provide automatic
channel routing. Its caller must honor its supported channel/batch layout.

## Spline

Source: [RQSConfig](../../src/glnis/core/nn/flow/spline_transform.py).

| Option under `sampler.sampler.flow.spline` | Default | Meaning |
| --- | --- | --- |
| `kind` | `rqs` | Rational quadratic spline |
| `bins` | `10` | Number of bins |
| `left`, `right` | `0.0`, `1.0` | Forward-coordinate interval |
| `bottom`, `top` | `0.0`, `1.0` | Transformed-coordinate interval |
| `min_bin_width` | `0.001` | Minimum normalized bin width |
| `min_bin_height` | `0.001` | Minimum normalized bin height |
| `min_bin_derivative` | `0.001` | Derivative regularization parameter |

Choose positive bin sizes with `bins * min_bin_width < 1` and
`bins * min_bin_height < 1`. Bounds and spline settings must remain compatible
with the flow's unit-cube latent distribution. Shape-dependent checks occur at
model construction or evaluation; not every mathematical constraint is validated
by the configuration dataclass.

See [extending components](../implementation/extending.md) for adding implementations.
