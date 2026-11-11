# Sampling and feedback contract

Sources: [GammaBoardSampler](../../src/gammaboard_sampler.py),
[RectangularOrchestrator](../../src/glnis/core/nn/orchestrator.py), and
[RectangularTrainingStatus](../../src/glnis/utils/types.py).

## Public arrays and internal coordinates

For `N` samples, the public `SampleBatch` contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `xs_discrete` | `(N, public_discrete_dims)` | Public discrete coordinates, including singleton axes |
| `xs_continuous` | `(N, public_continuous_dims)` | Coordinates after optional mapping |
| `weights` | `(N,)` | Integration weights, already including proposal and mapping corrections |
| `training_remaining` | positive integer or `None` | Ungenerated training-window size before this draw, or no feedback request |

All discrete cardinalities are positive. Continuous dimensions can be zero in
the sampler API. Internal discrete cardinalities consist of the public axes
followed by hidden mapping axes; singleton axes are removed before model
construction. Generation restores them as zero-valued columns before mapping.

A module's sampled tensor packs discrete and continuous coordinates according to
its configuration. `to_disc_cont()` splits this tensor into NumPy arrays. The
orchestrator retains internal samples and their original probabilities for the
loss; it does not reconstruct them from mapped evaluator coordinates.

## Training-window lifecycle

1. `generate(max_samples)` receives a positive integer bound. During training it
   returns at most the ungenerated window remainder; internally it may combine
   several model forward passes.
2. Generated internal samples/probabilities are appended to training buffers.
3. When a full window has been generated but its feedback is incomplete,
   generation returns `GenerationStatus.WAITING`.
4. `feedback(Feedback(...))` supplies raw values in generation order. Feedback
   cannot exceed the number of generated but unconsumed training samples.
5. A complete window triggers optimization, records the loss, increments the step,
   and resets the active buffer counts.
6. After `training_steps`, generation returns inference draws with
   `training_remaining=None`; the default orchestrator ignores extra feedback.

The GammaBoard process contract supplies complete draws. The buffers also support
fragmented draws/windows and persist their unfinished contents for recovery.
The default orchestrator does not report `FINISHED` when training ends; inference
is available until the caller's external budget is exhausted.

All-zero windows advance without changing the proposal or stepping the scheduler.
Nonfinite losses/gradients skip optimization. Such skips still complete a window
and are visible in loss history.

## Raw feedback and correction ownership

`Feedback.unmasked_values` contains raw evaluator values. `Feedback.values`
applies the requested/actual stratum mask when assignments are present. The
adapter creates corrected feedback without mutating the caller's array or
losing its assignments.

If the internal proposal probability is `q`, the mapping Jacobian is `J`, and
the separate integration correction is `r`, then:

```text
returned batch weight       = J * r / q
value supplied to the loss  = raw masked evaluator value * J * r
```

The loss already accounts for `q`. Multiplying feedback by the batch's full
importance weight would apply the proposal correction twice. Pending `J * r`
values follow feedback order and are stored as `pending_mapping_weights` in
snapshot metadata. See [mapping semantics](mappings.md).

## Strata

Rectangular domains may declare leading stratum axes. The adapter preserves
those public axes and passes a domain with sampled singleton-free stratum axes
to the orchestrator. GammaBoard can supply requested and actual stratum labels;
the feedback mask removes mismatched samples from the training target.

The built-in orchestrator uses the masked values; it does not duplicate models
or allocate stratum-specific budgets. A future orchestrator with separate child
models must retain draw provenance, route feedback, and checkpoint all pending
routing state. See [extension guidance](extending.md#orchestrators-and-strata).

## Local execution

`SamplerAggregator` serializes sampler operations and evaluates independent chunks
concurrently. It restores training order before calling the adapter with raw
projected values. It retains weighted values separately for integration statistics.
Inference chunks can update the accumulator in completion order.

Relevant checks are in [sampler tests](../../tests/test_gammaboard_sampler.py),
[flow tests](../../tests/test_flow_modules.py),
[mapped feedback tests](../../tests/test_mapped_sampler.py), and
[ordered local aggregation tests](../../tests/test_sampler_aggregator.py).
