"""Read-only PDF queries and completed-run sampler checkpoints."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from glnis.core.config import RunConfig
from glnis.core.integrand import SineIntegrand
from glnis.core.sampler_aggregator import SamplerAggregator


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_config(kind="default"):
    return {
        "sampler": {
            "kind": kind, "use_gpu": False,
            "training_steps": 1, "training_batch_size": 32, "max_batch_size": 8,
            "flow_config": {"layers": 2, "units": 8, "bins": 4},
            "transformer_config": {
                "embedding_dim": 8, "feedforward_dim": 8, "heads": 2,
                "mlp_units": 8, "transformer_layers": 1,
            },
        },
        "integrand": {"kind": "sine", "continuous_dims": 2, "offset": 0.1},
        "aggregator": {"inference_samples": 16, "log_every": 0},
    }


@pytest.mark.parametrize("kind", ["default", "sgn_strat_naive", "sgn_strat_full"])
@pytest.mark.parametrize("typed_config", [False, True])
def test_completed_run_pdf_and_checkpoint_recovery(tmp_path, kind, typed_config):
    config = run_config(kind)
    aggregator = SamplerAggregator.from_config(config)
    aggregator.run()
    assert aggregator.sampler.training_samples_remaining() is None
    assert aggregator.pdf_continuous_dims == 2
    assert aggregator.pdf_discrete_cardinalities == ([] if kind == "default" else [2])
    continuous = np.linspace(0.1, 0.9, 26).reshape(13, 2)
    discrete = (np.empty((13, 0), dtype=np.int64) if kind == "default"
                else (np.arange(13, dtype=np.int64) % 2)[:, None])
    original_continuous, original_discrete = continuous.copy(), discrete.copy()
    histories = deepcopy((aggregator.training_history, aggregator.inference_history))
    expected = aggregator.sampler.pdf(discrete.copy(), continuous.copy())
    np.testing.assert_allclose(aggregator.pdf(discrete, continuous, batch_size=3), expected, rtol=1e-13)
    np.testing.assert_array_equal(continuous, original_continuous)
    np.testing.assert_array_equal(discrete, original_discrete)
    checkpoint = aggregator.snapshot(tmp_path / "checkpoints" / "sampler.pt")
    assert aggregator.sampler.cfg.save_path is None
    assert checkpoint["save_path"] == str(tmp_path / "checkpoints" / "sampler.pt")
    np.testing.assert_equal(histories, (aggregator.training_history, aggregator.inference_history))

    # Recovery constructs only the saved model; no freshly initialized flow is
    # necessary. Both accepted configuration representations restore the run.
    recovered = SamplerAggregator.from_snapshot(
        checkpoint, RunConfig.from_dict(config) if typed_config else config,
        accumulator=aggregator.accumulator,
    )
    assert type(recovered.sampler) is type(aggregator.sampler)
    assert recovered.sampler.training_samples_remaining() is None
    assert recovered.sampler.cfg.save_path == checkpoint["save_path"]
    assert recovered.accumulator is aggregator.accumulator
    assert recovered.training_history == recovered.inference_history == []
    np.testing.assert_allclose(recovered.pdf(discrete, continuous), expected, rtol=1e-13)
    recovered.run(mode="infer", inference_samples=8)
    assert recovered.accumulator.result.n_points == 24


class ServiceSampler:
    def __init__(self):
        self.cfg = SimpleNamespace(max_batch_size=3, save_path=None)
        self._all_discrete_cardinalities = [2]
        self.calls = []

    def pdf(self, discrete, continuous):
        self.calls.append(len(continuous))
        result = continuous.sum(axis=1) + discrete[:, 0]
        discrete[:] = 0
        continuous[:] = 0
        return result

    def snapshot(self):
        self.calls.append(self.cfg.save_path)
        return {"save_path": self.cfg.save_path}


def service_aggregator():
    return SamplerAggregator(ServiceSampler(), SineIntegrand(continuous_dims=2))


def test_pdf_batches_preserve_order_and_protect_input_arrays():
    aggregator = service_aggregator()
    discrete = (np.arange(7) % 2)[:, None]
    continuous = np.arange(14, dtype=float).reshape(7, 2)
    before = (discrete.copy(), continuous.copy())
    expected = continuous.sum(axis=1) + discrete[:, 0]
    np.testing.assert_array_equal(aggregator.pdf(discrete, continuous, batch_size=20), expected)
    assert aggregator.sampler.calls == [3, 3, 1]
    np.testing.assert_array_equal(discrete, before[0])
    np.testing.assert_array_equal(continuous, before[1])
    assert aggregator.training_history == aggregator.inference_history == []


@pytest.mark.parametrize("discrete,continuous,message", [
    ([[0]], [[0.5]], "continuous coordinates must have shape"),
    ([[0], [1]], [[0.5, 0.5]], "discrete coordinates must have shape"),
    ([[0.0]], [[0.5, 0.5]], "discrete coordinates must be integers"),
    ([[2]], [[0.5, 0.5]], "discrete column 0"),
    ([[-1]], [[0.5, 0.5]], "discrete column 0"),
    ([[0]], [[float("nan"), 0.5]], "finite real numbers"),
    ([[0]], [[0.5j, 0.5]], "finite real numbers"),
])
def test_pdf_validates_inputs_before_evaluating(discrete, continuous, message):
    aggregator = service_aggregator()
    with pytest.raises(ValueError, match=message):
        aggregator.pdf(discrete, continuous)
    assert aggregator.sampler.calls == []


def test_empty_pdf_does_not_call_sampler_and_unsupported_pdf_is_explicit():
    aggregator = service_aggregator()
    assert aggregator.pdf(np.empty((0, 1), dtype=np.int64), np.empty((0, 2))).shape == (0,)
    assert aggregator.sampler.calls == []
    aggregator.sampler.pdf = lambda *args: None
    with pytest.raises(NotImplementedError, match="does not provide PDF"):
        aggregator.pdf([[0]], [[0.2, 0.3]])


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_pdf_rejects_invalid_batch_sizes(batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        service_aggregator().pdf([[0]], [[0.2, 0.3]], batch_size=batch_size)


def test_services_cannot_race_with_active_generation(tmp_path):
    aggregator = service_aggregator()
    aggregator._running = True
    with pytest.raises(RuntimeError, match="while the aggregator is running"):
        aggregator.pdf([[0]], [[0.2, 0.3]])
    with pytest.raises(RuntimeError, match="while the aggregator is running"):
        aggregator.snapshot(tmp_path / "checkpoint.pt")
    assert aggregator.sampler.calls == []


def test_snapshot_requires_a_path_and_restores_path_after_failure(tmp_path):
    aggregator = service_aggregator()
    with pytest.raises(ValueError, match="checkpoint path or sampler.save_path"):
        aggregator.snapshot()
    original = str(tmp_path / "configured.pt")
    aggregator.sampler.cfg.save_path = original
    assert aggregator.snapshot() == {"save_path": original}

    def fail():
        raise OSError("checkpoint write failed")

    aggregator.sampler.snapshot = fail
    with pytest.raises(OSError, match="checkpoint write failed"):
        aggregator.snapshot(tmp_path / "override.pt")
    assert aggregator.sampler.cfg.save_path == original


@pytest.mark.parametrize("kind", ["default", "sgn_strat_naive"])
def test_mapping_pdf_metadata_and_checkpoint_recovery(tmp_path, kind):
    config = run_config(kind)
    config["sampler"].update({
        "training_steps": 0,
        "mapping": {"layer_cfgs": [{"kind": "spherical", "conformal_scale": 2.0}]},
        "graph_properties": {
            "edge_masses": [0.0], "edge_momentum_shifts": [[0.0, 0.0, 0.0]],
            "graph_signature": [[1]], "lmb_edges": [[0]],
            "generation_lmb_edges": [0], "e_cm": 0.0,
        },
    })
    config["integrand"] = {"kind": "gaussian", "continuous_dims": 3, "sigma": [0.1, 0.2]}
    aggregator = SamplerAggregator.from_config(config)
    assert aggregator.pdf_continuous_dims == 3
    assert aggregator.pdf_discrete_cardinalities == ([2] if kind == "default" else [2, 2])
    generated = aggregator.sampler.generate(8)
    if isinstance(generated, tuple):
        batch, strata = generated
        discrete = np.column_stack((strata, batch.xs_discrete))
    else:
        batch = generated
        discrete = batch.xs_discrete
    continuous = batch.xs_continuous.copy()
    expected = aggregator.sampler.pdf(discrete.copy(), continuous.copy())
    assert np.all(expected > 0)
    np.testing.assert_allclose(aggregator.pdf(discrete, continuous), expected, rtol=1e-13)
    np.testing.assert_array_equal(continuous, batch.xs_continuous)
    recovered = SamplerAggregator.from_snapshot(aggregator.snapshot(tmp_path / "mapped.pt"), config)
    np.testing.assert_allclose(recovered.pdf(discrete, continuous), expected, rtol=1e-13)
