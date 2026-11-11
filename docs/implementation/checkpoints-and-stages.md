# Checkpoints and fresh stages

Sources: [adapter snapshots](../../src/gammaboard_sampler.py),
[orchestrator persistence](../../src/glnis/core/nn/orchestrator.py), and
[proposal serialization](../../src/glnis/core/nn/sampler_module.py).

## Checkpoint contents

`sampler.snapshot()` requires `save_path`. It returns JSON-safe metadata and
atomically writes a binary PyTorch checkpoint to that path.

| Location | Contents |
| --- | --- |
| Adapter metadata | Adapter kind/configuration, public domain, orchestrator snapshot reference, pending mapping weights |
| Orchestrator metadata | Orchestrator kind, binary source path, normalized configuration |
| Binary checkpoint | Serialized module, training buffers/status, optimizer state, scheduler state, CPU and used CUDA RNG states |

The module is serialized with its Python class. Moving or renaming implementation
classes may require a checkpoint migration. Loading uses PyTorch/pickle and is
appropriate only for trusted files. A path is a mutable checkpoint destination:
saving again there replaces the previous binary state.

Mappings are reconstructed from adapter configuration. If their graph data come
from GammaLoop metadata rather than explicit `graph_properties`, the state and
optional parsing dependencies must remain available at reconstruction time.
Custom mapping-local RNGs/state are not automatically part of the default
orchestrator's PyTorch RNG persistence.

## Exact recovery

Given a sampler configured with a checkpoint path:

```python
snapshot = sampler.snapshot()
restored = GammaBoardSampler.from_snapshot(
    snapshot=snapshot,
    domain=sampler.domain,
    init_args={"save_path": "checkpoints/resumed.pt", "orchestrator": {"use_gpu": False}},
)
```

`from_snapshot()` keeps the proposal, optimizer/scheduler states, counters,
unfinished generation/feedback windows, and pending mapping factors. Outstanding
feedback must still arrive in its original order. This supports recovery during
a partially completed training window.

The public domain must remain compatible. These runtime/output settings may be
overridden through `init_args`:

| Setting | Location | Effect |
| --- | --- | --- |
| `save_path` | adapter | Future checkpoint destination |
| `use_gpu` | orchestrator | Device preference |
| `cuda_id` | orchestrator | Device index |
| `use_f64` | orchestrator | Model and training-buffer precision |
| `max_batch_size` | orchestrator | Generation/forward-pass chunk limit |
| `execution` | orchestrator | Eager/compiled policy and per-method compiler settings |

The binary state is read from `snapshot["orchestrator_snapshot"]["save_path"]`,
even when the future output path changes. To relocate a standalone checkpoint,
update that source reference. Comparison restoration resolves its relative
checkpoint references automatically.

Changing proposal architecture, mapping, parser/graph configuration, or training
settings is rejected during exact recovery. Configuration-less metadata require
`init_args` to supply the missing configuration. The old `pending_mapping_jac`
metadata key is read as the combined pending mapping weights; new snapshots write
`pending_mapping_weights`.

Restoring retains the saved numerical state and RNG states. Subsequent bitwise
continuation also depends on the execution environment, including compiler
artifacts. Device or precision overrides can change numerical results.

Execution callables are runtime objects, outside the serialized module. Saving a
checkpoint retains the raw module and training state without compilation. Restore
recreates the execution policy after device/precision placement and restores RNG
last; compilation happens lazily on first use and may reuse compatible PyTorch
caches. Switching eager/compiled policies or compiler settings can change random
streams and arithmetic and is not a promise of bitwise-identical continuation.
No compiler cache is required to read the checkpoint.
The [native six-dimensional benchmark](compilation-native-6d.md) recovered
bitwise with reused compiler artifacts. With an empty Inductor cache, some CUDA
graph runs differed at floating-point roundoff despite exactly restored state;
fresh compilation is not a guarantee of bitwise continuation.
The checkpoint is initially read on CPU; module/buffer placement and optimizer
loading then move the appropriate tensors, preserving Adam's CPU step counters.

## Fresh stages

```python
stage = GammaBoardSampler.from_stage(
    source_state=snapshot,
    source_args={},
    domain=sampler.domain,
    init_args={
        "save_path": "checkpoints/stage.pt",
        "orchestrator": {
            "training_steps": 20,
            "training_batch_size": 256,
            "learning_rate": 0.0005,
            "loss": {"kind": "variance"},
        },
    },
)
```

A stage keeps compatible proposal parameters and inherits source settings, with
`init_args` overrides. It creates fresh training buffers, statistics, counters,
optimizer, scheduler, and a new draw sequence seeded by the effective stage seed.
Source pending feedback is discarded. It works with a fresh or previously
trained source proposal. Set `training_steps=0` for an inference stage.

The orchestrator explicitly selects training/evaluation mode from the new stage's
training status. Starting fresh training on a saved evaluation-mode module returns
it to training mode and creates new execution callables.

Proposal architecture, mapping, parser/graph configuration, public domain, and
orchestrator kind remain compatible. In particular, this factory does not
convert a rectangular orchestrator into a different orchestrator implementation.
`source_args` supply fallback configuration when the source metadata lack it.

## Verification and local services

[Checkpoint tests](../../tests/test_sampler_checkpoints.py) cover pending feedback,
JSON metadata, precise continuation, runtime/precision overrides, atomic writes,
legacy field reads, and resetting a new stage. [Factory tests](../../tests/test_losses_and_schedulers.py)
cover registered custom components through both lifecycle factories.

The [local aggregator](../usage/local-runs.md#read-only-services) can save and
restore completed proposals with an integrand/configuration. It starts fresh
histories and does not reconstruct a partially accumulated local training-step
statistical record from a sampler checkpoint.
