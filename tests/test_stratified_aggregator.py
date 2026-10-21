"""Strata travel with sample positions through asynchronous evaluation."""

from threading import Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest
from gammaboard_process import GenerationStatus, SampleBatch

from glnis.core.accumulator import DefaultAccumulator, TrainingAccumulator
from glnis.core.config import AggregatorConfig
from glnis.core.sampler_aggregator import SamplerAggregator
from glnis.utils.types import LayerData


VALUES = np.array([1 + 3j, -2 - 4j, -3 + 5j, 4 - 6j, 0 + 7j, 8 + 0j])


class ComplexIntegrand:
    continuous_dims = 1
    discrete_cardinalities = [3]
    target = None

    def evaluate_batch(self, continuous, discrete):
        ids = continuous[:, 0].astype(int)
        # Ordinary integrand channels must remain separate from strata.
        np.testing.assert_array_equal(discrete[:, 0], ids % 3)
        return VALUES[ids % len(VALUES)]


class StratifiedSampler:
    def __init__(self, *, steps=2, window=7, maximum=3, multi=False):
        self.cfg = SimpleNamespace(max_batch_size=maximum)
        self.training_status = SimpleNamespace(step=0, num_generated=0, num_ingested=0, last_loss=0.0)
        self.steps, self.window, self.multi = steps, window, multi
        self.next_id = 0
        self.draws = []
        self.feedback_values = []
        self.transforms = []
        self.lock = Lock()

    def get_strat_cardinalities(self):
        return [2, 3] if self.multi else [2]

    def training_samples_remaining(self):
        if self.training_status.step >= self.steps:
            return None
        return self.window - self.training_status.num_generated

    def generate(self, limit):
        remaining = self.training_samples_remaining()
        if remaining == 0:
            return GenerationStatus.WAITING
        n = limit if remaining is None else min(limit, remaining)
        ids = np.arange(self.next_id, self.next_id + n, dtype=np.int64)
        self.next_id += n
        strata = ids % 2
        if self.multi:
            strata = np.column_stack([strata, ids % 3])
        # Negative weights distinguish selecting the raw phase from selecting
        # the weighted contribution. Weighting must happen after the transform.
        weights = np.where(ids % 2 == 0, 2.0, -3.0)
        batch = SampleBatch((ids % 3)[:, None], ids[:, None].astype(float), weights, remaining)
        self.draws.append((ids.copy(), batch, strata.copy()))
        if remaining is not None:
            self.training_status.num_generated += n
        return batch, strata

    def feedback(self, values):
        self.feedback_values.append(np.array(values, copy=True))
        self.training_status.num_ingested += len(values)
        if self.training_status.num_ingested == self.window:
            assert self.training_status.num_generated == self.window
            self.training_status.step += 1
            self.training_status.num_generated = self.training_status.num_ingested = 0

    def strat_channel_transform(self, values, strata):
        assert values.ndim == 1
        assert strata.dtype == np.int64
        assert strata.ndim == (2 if self.multi else 1)
        with self.lock:
            self.transforms.append((values.copy(), strata.copy()))
        sign = strata[:, 0] if self.multi else strata
        values[(sign == 0) & (values < 0)] = 0
        values[(sign == 1) & (values > 0)] = 0
        return values


def configuration(phase="real", mode="train_then_infer", inference_samples=9):
    return AggregatorConfig(mode=mode, training_phase=phase, inference_samples=inference_samples,
                            evaluation_batch_size=2, evaluation_workers=3,
                            max_in_flight_batches=3, log_every=0)


def filtered_values(ids, phase):
    values = VALUES[ids % len(VALUES)].copy()
    selected = np.abs(values) if phase == "abs" else getattr(values, phase)
    keep = ((ids % 2 == 0) & (selected > 0)) | ((ids % 2 == 1) & (selected < 0))
    values[~keep] = 0
    return values


@pytest.mark.parametrize("phase", ["real", "imag", "abs"])
def test_phase_filtering_and_strata_preserve_exact_feedback_order(monkeypatch, phase):
    later_finished = Event()

    class ReorderedIntegrand(ComplexIntegrand):
        def evaluate_batch(self, continuous, discrete):
            first = int(continuous[0, 0])
            if first == 0:
                assert later_finished.wait(5)
            values = super().evaluate_batch(continuous, discrete)
            if first == 3:
                later_finished.set()
            return values

    training_chunks = []
    original_init = TrainingAccumulator.__init__

    def capture(self, data, **kwargs):
        training_chunks.append((data.continuous[:, 0].astype(int), data.strata.copy(),
                                data.func_val.ravel().copy(), kwargs["strat_channels"]))
        original_init(self, data, **kwargs)

    monkeypatch.setattr(TrainingAccumulator, "__init__", capture)
    sampler = StratifiedSampler()
    aggregator = SamplerAggregator(sampler, ReorderedIntegrand(), configuration(phase))
    layer_data = []
    init_layer_data = aggregator._init_layer_data

    def capture_data(n):
        data = init_layer_data(n)
        layer_data.append(data)
        return data

    monkeypatch.setattr(aggregator, "_init_layer_data", capture_data)
    aggregator.run()
    assert later_finished.is_set()
    assert [record["n_samples"] for record in aggregator.training_history] == [7, 7]
    assert len(sampler.feedback_values) == 6  # Three generation draws per step.
    training_draws = sampler.draws[:len(sampler.feedback_values)]
    for (ids, batch, _), feedback in zip(training_draws, sampler.feedback_values):
        weighted = filtered_values(ids, phase) * batch.weights
        expected = np.abs(weighted) if phase == "abs" else getattr(weighted, phase)
        np.testing.assert_array_equal(feedback, expected)
    for ids, strata, values, channels in training_chunks:
        np.testing.assert_array_equal(strata[:, 0], ids % 2)
        np.testing.assert_array_equal(values, filtered_values(ids, phase))
        assert channels == [2]
    # Inference uses the same configured phase and records strata, without
    # sending any of its samples back as training feedback.
    inference_draws = sampler.draws[len(sampler.feedback_values):]
    ids = np.concatenate([draw[0] for draw in inference_draws])
    weights = np.concatenate([draw[1].weights for draw in inference_draws])
    expected = filtered_values(ids, phase) * weights
    assert aggregator.accumulator.result.n_points == 9
    for channel in (0, 1):
        result = aggregator.accumulator.statistics._strat_dict[(channel,)]
        mask = ids % 2 == channel
        assert result.n_points == int(mask.sum())
        assert result.real.mean == pytest.approx(expected[mask].real.mean())
        assert result.imag.mean == pytest.approx(expected[mask].imag.mean())
    assert aggregator.accumulator.get_observables()["n_failures"] == 0
    # Sign rejection only changes func_val. Every other LayerData field is
    # preserved, including for rejected samples in both training and inference.
    for data in layer_data:
        ids = data.continuous[:, 0].astype(int)
        np.testing.assert_array_equal(data.wgt.ravel(), np.where(ids % 2 == 0, 2.0, -3.0))
        np.testing.assert_array_equal(data.jac.ravel(), np.ones(len(ids)))
        np.testing.assert_array_equal(data.discrete[:, 0], ids % 3)
        np.testing.assert_array_equal(data.strata[:, 0], ids % 2)
        np.testing.assert_array_equal(data.func_val.ravel(), filtered_values(ids, phase))
        assert data.success.all()


def test_multiple_stratum_dimensions_remain_separate_from_integrand_channels():
    sampler = StratifiedSampler(steps=0, multi=True)
    aggregator = SamplerAggregator(sampler, ComplexIntegrand(), configuration(mode="infer", inference_samples=12))
    aggregator.run()
    assert aggregator.strat_cardinalities == [2, 3]
    strata_results = aggregator.accumulator.statistics._strat_dict
    assert set(strata_results) == {(a, b) for a in range(2) for b in range(3)}
    assert all(result.n_points == 2 for result in strata_results.values())
    for phase_values, strata in sampler.transforms:
        assert len(phase_values) == len(strata)
        assert strata.shape[1] == 2


def test_compatible_existing_stratified_accumulator_can_be_extended():
    data = LayerData(n_points=1, n_cont=1, n_disc=1, n_strata=1)
    data.continuous = [[0]]
    data.discrete = [[0]]
    data.strata = [1]
    data.func_val = np.array([-5 - 7j])
    data.update()
    accumulator = DefaultAccumulator(data, strat_channels=[2])
    aggregator = SamplerAggregator(StratifiedSampler(steps=0), ComplexIntegrand(),
                                  configuration(mode="infer", inference_samples=2), accumulator=accumulator)
    assert aggregator.run() is accumulator
    assert accumulator.result.n_points == 3
    assert accumulator.statistics._strat_dict[(1,)].n_points == 2


@pytest.mark.parametrize("channels", [None, [3], [2, 2]])
def test_existing_accumulator_must_match_sampler_strata(channels):
    with pytest.raises(ValueError, match="stratification channels"):
        SamplerAggregator(StratifiedSampler(), ComplexIntegrand(), configuration(),
                          accumulator=DefaultAccumulator(strat_channels=channels))


@pytest.mark.parametrize("strata, message", [
    (np.array([0, 1]), "Strata must have shape"),
    (np.array([0.0, 1.0, 0.0]), "integer channel indices"),
    (np.array([0, -1, 0]), "indices in"),
    (np.array([0, 2, 0]), "indices in"),
    (np.zeros((3, 2), dtype=np.int64), "Strata must have shape"),
    (None, "Strata must have shape"),
])
def test_invalid_strata_are_rejected_before_evaluation(strata, message):
    sampler = StratifiedSampler(steps=0)
    generate = sampler.generate
    sampler.generate = lambda limit: (generate(limit)[0], strata)
    aggregator = SamplerAggregator(sampler, ComplexIntegrand(), configuration(mode="infer", inference_samples=3))
    with pytest.raises(ValueError, match=message):
        aggregator.run()
    assert not sampler.transforms


def test_stratified_sampler_must_return_the_tuple_contract():
    sampler = StratifiedSampler(steps=0)
    generate = sampler.generate
    sampler.generate = lambda limit: generate(limit)[0]
    aggregator = SamplerAggregator(sampler, ComplexIntegrand(), configuration(mode="infer", inference_samples=3))
    with pytest.raises(TypeError, match="stratified sampler must return"):
        aggregator.run()


def test_generation_status_is_not_unpacked_as_strata():
    sampler = StratifiedSampler()
    sampler.generate = lambda limit: GenerationStatus.WAITING
    aggregator = SamplerAggregator(sampler, ComplexIntegrand(), configuration(mode="train"))
    with pytest.raises(RuntimeError, match="WAITING with no outstanding"):
        aggregator.run()


def test_unstratified_sampler_does_not_call_the_transform():
    sampler = StratifiedSampler(steps=0)
    generate = sampler.generate
    sampler.generate = lambda limit: generate(limit)[0]
    sampler.get_strat_cardinalities = lambda: None

    def forbidden(*args):
        raise AssertionError("No transform is needed for an unstratified sampler")

    sampler.strat_channel_transform = forbidden
    aggregator = SamplerAggregator(sampler, ComplexIntegrand(), configuration(mode="infer", inference_samples=3))
    aggregator.run()
    expected = VALUES[:3] * np.array([2, -3, 2])
    assert aggregator.accumulator.result.real.mean == pytest.approx(expected.real.mean())
    assert aggregator.accumulator.result.imag.mean == pytest.approx(expected.imag.mean())
