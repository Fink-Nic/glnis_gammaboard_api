# Extending sampler components

Start with the [architecture](architecture.md) and the contract for the component
you are changing. Configuration kinds are discovered through
[BuildFromKind](../../src/glnis/utils/types.py).

## Registration

Subclass the appropriate base, assign a unique `KIND`, implement its interface,
and import the module before construction. The registry walks imported classes;
it does not scan source files or load external entry points. Only explicitly
defined `KIND` attributes register a new implementation, and duplicates raise
an error. `DEFAULT_KIND` on the factory base controls an omitted kind.

For example, a loss with a different default clip can reuse the current KL loss:

```python
from glnis.core.nn.losses import KLDivergence

class ClippedKL(KLDivergence):
    KIND = "clipped kl"

    def __init__(self, threshold=10.0, normalize_across_training_steps=False):
        super().__init__(threshold=threshold,
                         normalize_across_training_steps=normalize_across_training_steps)
```

Once imported, select it using `loss={"kind": "clipped kl"}` in the orchestrator
configuration. Keep constructor options in a configuration dataclass where
appropriate. Update the option reference and behavior tests with the implementation.

## Interfaces

| Component | Base / required behavior |
| --- | --- |
| Orchestrator | `Orchestrator`; generation, feedback, snapshot, `_from_snapshot`; `_from_stage` for stage transfer |
| Proposal module | `RectangularSamplerModule`; `sample`, `prob`, `pdf`, `discrete_mass`, `to_disc_cont` |
| Execution policy | `SamplerExecution`; runtime `sample`/`prob` callables and a `cuda_graphs` flag for iteration boundaries |
| Continuous flow | `Flow` factory family; conditional `sample` and probability interface compatible with the mixed module |
| Discrete proposal | `DiscreteFlow`; conditional `sample`, `prob`, and supported marginal masses |
| Subnet | `SubnetConstructor`; PyTorch `forward` on the supplied regular/conditional features |
| Spline | `SplineTransform`; condition width and coordinate/log-Jacobian transformation |
| Mapping | `Mapping`; coordinate transforms, dimensions/cardinalities, and separate integration correction |
| Loss | `Loss`; callable on training status and differentiable proposal probabilities |
| Scheduler | `Scheduler`; PyTorch-compatible step and state interface |

Abstract methods must be implemented even if a capability explicitly returns
`None`, such as an unavailable analytic marginal/PDF. Do not hide implementation
errors behind a blanket exception that returns `None`.

## Neural modules and conditions

Register parameters, child modules, and buffers through PyTorch so device moves,
optimizer construction, and persistence see them. Honor supplied dtype/device
rather than relying on global defaults.

Use the active [flow subnet interface](../../src/glnis/core/nn/flow/subnet_constructor.py).
Its final `features_c` input features are the condition; `mlp` uses the combined
input width, while `mlp fully conditioned` reuses the condition at hidden layers.
Transformer subnets can receive tensors with a sequence axis. The proposal's
sampling and differentiable probability paths must use identical factorization
and conditions after training, not only at zero-initialized weights.

`prob` scores complete internal points. Discrete marginal/prefix queries belong
to `discrete_mass` and stay eager. The orchestrator may compile `sample`/`prob`
through a separate execution policy; module implementations should not install
compiler wrappers into their serialized state. A stratum orchestrator can own
one execution policy per child proposal.

The current spline base is a factory-selected callable, not a registered
`nn.Module` by itself. If adding learned spline parameters, ensure the owning
PyTorch model actually registers and optimizes them; selection through `kind`
alone does not provide that ownership.

## Coordinate mappings

Keep the [mapping contract](mappings.md): individual transforms return geometric
Jacobians and coordinates, and `integration_weight` supplies any separate forward
correction. Report input/output widths and discrete cardinalities accurately.
Use an actual inverse when implementing `backward`; preserving coordinate space
does not mean applying the forward transform twice inverts it.

Add tests for round trips where an inverse exists, Jacobians, public PDF behavior,
and sample/feedback correction ownership. Keep optional backend imports inside
the operations that need them. A forward-only map should report its inverse as
unimplemented rather than fabricate a density.

## Orchestrators and strata

An orchestrator receives the internal rectangular domain, shared proposal
`module_kwargs`, and checkpoint `save_path`. Expose a configuration with
`to_dict()` for the adapter's snapshot metadata. For local runners, expose the
training-window/status information they use and a generation limit.

A module-per-stratum design can own child modules, optimizers, schedulers, and
training states. It must also own pending draw provenance so feedback can be
routed in order, and persist that routing information with every child state.
Define how `training_remaining` covers the chosen training window and how the
parent reports `WAITING` while feedback is outstanding.

The adapter preserves GammaBoard stratum assignments and masks. Independent
stratum proposals, budget allocation, and cross-kind stage conversion are not
implemented by registering a new name alone. Implement lifecycle behavior and
statistical semantics explicitly, then add interrupted-recovery and fresh-stage
tests for that orchestrator.

## Integrands and tests

Local integrands use the existing explicit `Integrand.from_config` registry,
not `BuildFromKind`. Extend that factory deliberately if adding a new local
integrand; its batch evaluation must be read-only and safe for concurrent calls.

Start from [factory tests](../../tests/test_losses_and_schedulers.py),
[flow identities](../../tests/test_flow_modules.py), and
[checkpoint tests](../../tests/test_sampler_checkpoints.py). For a complete
verification workflow, see [development](../development.md).
