# Local integration

A local TOML run combines `[sampler]`, `[integrand]`, and an optional `[aggregator]`.
The public domain is derived from the integrand. Refer to the
[configuration overview](../configuration/README.md) for table nesting.

## Run an existing example

From an activated installation:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 glnis-integrate examples/local/ghostbumps.toml --history /tmp/ghostbumps-history.json
```

The equivalent module is `python -m glnis.scripts.run`. In a source checkout,
`python tests/run_from_toml.py --history /tmp/history.json` defaults to the
[ghostbumps example](../../examples/local/ghostbumps.toml).

Other current examples are [sine](../../examples/local/sine.toml) on the unit cube
and [Gaussian](../../examples/local/gaussian.toml) over physical momentum space.
The Gaussian's spherical mapping supplies its change-of-variables correction;
a Gaussian sampled on the unit cube alone generally does not integrate to one.
Use these checked-in files as complete configurations instead of combining
fragments from different schema versions.

`--history` writes training and inference observables as JSON. Training has one
record per completed step; inference records accumulator updates. Nonfinite
JSON diagnostic values are represented as `null`.

## Python aggregation API

```python
from glnis.core.sampler_aggregator import SamplerAggregator

run = SamplerAggregator.from_toml("examples/local/ghostbumps.toml")
accumulator = run.run()
training_history = run.training_history
inference_history = run.inference_history
```

Inside an event loop, use `await run.run_async()`. Repeat inference with
`run.run(mode="infer", inference_samples=1000)` to extend the accumulator.
Inference requires completed training or a proposal configured with zero
training steps. Histories start empty when constructing a restored aggregator;
pass an existing compatible accumulator explicitly if continuing its statistics.

The aggregator serializes generation and feedback on one thread and evaluates
chunks concurrently. It restores draw and sample order before training feedback.
Integrands must support concurrent read-only evaluation. A fresh local training
run starts at a training-step boundary; use the sampler's recovery API when
resuming a partially filled window.

## Direct sampler API

For full control, evaluate the returned public coordinates yourself:

```python
import numpy as np
from gammaboard_process import Feedback, RectangularDomain
from gammaboard_sampler import GammaBoardSampler

sampler = GammaBoardSampler(
    domain=RectangularDomain(discrete_cardinalities=(), continuous_dims=2),
    orchestrator={"training_steps": 1, "training_batch_size": 128, "use_gpu": False},
)
batch = sampler.generate(128)
values = np.ones(len(batch.weights))
estimate = (values * batch.weights).mean()
sampler.feedback(Feedback(values))
```

Feedback is raw and unweighted. Sample weights already include the inverse
proposal probability, coordinate Jacobian, and any integration correction.
The sampler applies the mapping part of that correction to training feedback.
See [the numerical contract](../implementation/sampling-and-feedback.md).

## Read-only services

Use `run.pdf(discrete, continuous, batch_size=...)` while the aggregator is idle.
Inputs use public coordinates and exclude hidden mapping channels. A PDF without
an analytic implementation raises `NotImplementedError` at the aggregator layer.

`metadata = run.snapshot("checkpoints/sampler.pt")` saves the proposal and training
state. `SamplerAggregator.from_snapshot(metadata, run_config)` restores the
integrand/aggregator settings around that sampler. For exact sampler recovery,
runtime overrides, and fresh stages, see [checkpoints and stages](../implementation/checkpoints-and-stages.md).
