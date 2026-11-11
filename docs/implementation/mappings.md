# Coordinate mappings and integration weights

The mapping contract distinguishes coordinate geometry, integration corrections,
and probability densities. Source: [mappings.py](../../src/glnis/core/mappings.py)
and the adapter's [generation/PDF paths](../../src/gammaboard_sampler.py).

## Individual mapping interface

For a configured forward map `y = T(x)`, the methods return:

```python
jacobian, y = mapping.forward(x, discrete)
inverse_jacobian, x = mapping.backward(y, discrete)
correction = mapping.integration_weight(x, y, discrete)
```

The Jacobians are absolute coordinate determinants for the corresponding
direction. Existing individual mappings return column Jacobians, typically
shape `(N, 1)`, and continuous arrays of shape `(N, dims_out)`.
`integration_weight` returns a separate `(N,)` correction using the endpoints
in the configured forward direction. Its default is one.

An identity map has unit forward and backward Jacobians and preserves the
proposal density. A unit Jacobian does not make that density uniform.
Mappings that preserve coordinate space can still need distinct inverse logic;
for example, hypercube smoothing has its own `backward()` implementation.

## Layered result

`LayeredMapping` consumes the hidden discrete axes assigned to successive layers,
passes through unused continuous coordinates, and accumulates the two factors
independently:

```python
result = mapping_stack.forward(discrete, continuous)
y = result.continuous
J = result.jacobian
r = result.integration_weight
```

`LayeredMappingResult` is a named tuple with those three fields. Continuous arrays
have shape `(N, output_dims)` and both factor arrays have shape `(N,)`.
Backward traversal reverses layer/discrete-group order and returns inverse
coordinate Jacobians with reciprocal integration corrections where defined.
It uses the same public result type; there is no separate density transform API.

The pipeline allocates enough coordinate capacity for dimension-expanding layers.
Nonfinite coordinate/Jacobian results are tracked through `LayerData`; failed
rows return zero output coordinates and zero factors. Integration factors remain
separate from the accumulated coordinate Jacobian.

## Multichannel layers

`MCLayer` delegates coordinate transformations to its wrapped mapping. Its
`forward()`/`backward()` Jacobians contain only the wrapped coordinate geometry.
The `ose`, `fermi`, and `jac` implementations compute channel partitions in
momentum space through `_mc_weight()`.

For an HCUBE-to-MOMENTUM layer, the forward integration correction is the chosen
channel partition. For the reverse configured orientation, it is the reciprocal.
This convention preserves the stack's integration-weight behavior while keeping
its coordinate Jacobians independent. A layered stack multiplies the corrections
in traversal order.

If `c` denotes hidden channels and `q(d, c, x)` is the joint internal proposal,
the returned integration weight is:

```text
W(d, c, x) = J(d, c, x) * r(d, c, x) / q(d, c, x)
```

Raw evaluator feedback receives `J * r`, while the loss applies the sampling
probability correction. The adapter checkpoints pending combined factors so an
interrupted feedback window can be resumed.

## Public density

PDF queries use public coordinates and marginalize hidden mapping channels.
For invertible channel branches, the sampler computes:

```text
p(d, y) = sum_c q(d, c, T_c^-1(y)) * |det D T_c^-1(y)|
```

Only the coordinate Jacobian enters this expression. Channel integration
corrections do not change the actual proposal's pushforward distribution.
Consequently, `1 / batch.weights` need not equal the public marginal density
for a multichannel estimator.

When a mapping's inverse is not implemented, the sampler returns `None` for the
PDF query. Reduced or stochastic parameterizations should not be assumed to
provide an invertible density transform merely because they report a Jacobian.
The `uniform_continuous` identity option is a redraw mode, not the deterministic
identity used by the unit-Jacobian contract.

Tests verify [identity Jacobians, partition separation, stacked transforms,
feedback, and PDF invariance](../../tests/test_mapped_sampler.py), plus
[configuration-driven stack round trips](../../tests/test_layered_mapping.py).
