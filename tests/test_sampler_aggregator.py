"""Ordering and phase boundaries with deterministic samples and delayed workers."""

import asyncio
from dataclasses import replace
from threading import Event, Lock, get_ident
from types import SimpleNamespace

import numpy as np
import pytest
from gammaboard_process import GenerationStatus, SampleBatch

from glnis.core.accumulator import DefaultAccumulator
from glnis.core.config import AggregatorConfig
from glnis.core.sampler_aggregator import SamplerAggregator
from glnis.utils.types import LayerData


class OrderedSampler:
    def __init__(self, steps=2, window=7, maximum=3):
        self.cfg = SimpleNamespace(max_batch_size=maximum)
        self.training_status = SimpleNamespace(step=0, num_generated=0, num_ingested=0, last_loss=None)
        self.steps, self.window = steps, window
        self.next_sample = 0
        self.generated = []
        self.feedback_values = []
        self.threads = set()
        self.outstanding = []

    def training_samples_remaining(self):
        self.threads.add(get_ident())
        return None if self.training_status.step == self.steps else self.window - self.training_status.num_generated

    def generate(self, limit):
        remaining = self.training_samples_remaining()
        if remaining == 0:
            return GenerationStatus.WAITING
        n = min(limit, remaining) if remaining is not None else limit
        ids = np.arange(self.next_sample, self.next_sample + n, dtype=float)
        self.next_sample += n
        batch = SampleBatch(np.empty((n, 0), dtype=np.uint64), ids[:, None], 1 + ids / 10, remaining)
        self.generated.append((self.training_status.step, batch))
        if remaining is not None:
            self.training_status.num_generated += n
            self.outstanding.append(batch)
        return batch

    def feedback(self, values):
        self.threads.add(get_ident())
        batch = self.outstanding.pop(0)
        assert len(values) == len(batch.weights), "feedback must contain exactly one complete draw"
        self.feedback_values.append(np.array(values))
        self.training_status.num_ingested += len(values)
        if self.training_status.num_ingested == self.window:
            assert self.training_status.num_generated == self.window
            self.training_status.step += 1
            self.training_status.last_loss = 1 / self.training_status.step
            self.training_status.num_generated = self.training_status.num_ingested = 0


class IdentityIntegrand:
    continuous_dims = 1
    discrete_cardinalities = []
    target = None

    def evaluate_batch(self, continuous, discrete):
        return continuous[:, 0] + 1


class ReorderedIntegrand(IdentityIntegrand):
    """Force sample 0's chunk to complete after chunks from later draws."""

    def __init__(self, unblock_at=3):
        self.unblock_at = unblock_at
        self.later_finished = Event()
        self.completed = []
        self.lock = Lock()

    def evaluate_batch(self, continuous, discrete):
        first = int(continuous[0, 0])
        if first == 0:
            assert self.later_finished.wait(5), "later chunks must evaluate while the first waits"
        with self.lock:
            self.completed.append(first)
        if first == self.unblock_at:
            self.later_finished.set()
        return super().evaluate_batch(continuous, discrete)


def config(**overrides):
    return replace(AggregatorConfig(log_every=0, evaluation_workers=3, evaluation_batch_size=2,
                                    max_in_flight_batches=3), **overrides)


def test_training_restores_exact_sample_order_and_resets_each_step():
    sampler = OrderedSampler()
    integrand = ReorderedIntegrand()
    aggregator = SamplerAggregator(sampler, integrand, config(mode="train"))
    aggregator.run()

    assert integrand.completed.index(0) > integrand.completed.index(3)
    assert len(sampler.threads) == 1
    assert [record["n_samples"] for record in aggregator.training_history] == [7, 7]
    assert [record["step"] for record in aggregator.training_history] == [1, 2]
    assert [len(values) for values in sampler.feedback_values] == [3, 3, 1, 3, 3, 1]
    for (_, batch), feedback in zip(sampler.generated, sampler.feedback_values):
        np.testing.assert_array_equal(feedback, (batch.xs_continuous[:, 0] + 1) * batch.weights)
    for step, record in enumerate(aggregator.training_history):
        values = np.concatenate(sampler.feedback_values[3 * step:3 * (step + 1)])
        assert record["observables"]["real_mean"] == pytest.approx(values.mean())
    assert not aggregator.inference_history
    assert aggregator.accumulator.result.n_points == 0


def test_inference_merges_completed_chunks_without_waiting_for_sample_order():
    release_first = Event()

    class BlockedFirst(IdentityIntegrand):
        def evaluate_batch(self, continuous, discrete):
            if continuous[0, 0] == 0:
                assert release_first.wait(5)
            return super().evaluate_batch(continuous, discrete)

    sampler = OrderedSampler(steps=0, maximum=4)
    aggregator = SamplerAggregator(sampler, BlockedFirst(), config(mode="infer", inference_samples=4,
                                                            max_in_flight_batches=1))

    async def run():
        task = asyncio.create_task(aggregator.run_async())
        try:
            async with asyncio.timeout(3):
                while not aggregator.inference_history:
                    await asyncio.sleep(0.001)
        finally:
            release_first.set()
            await task

    asyncio.run(run())
    assert aggregator.inference_history[0]["sample_start"] == 2
    assert [record["n_samples"] for record in aggregator.inference_history] == [2, 4]
    assert not sampler.feedback_values
    values = (np.arange(4) + 1) * (1 + np.arange(4) / 10)
    assert aggregator.accumulator.result.real.mean == pytest.approx(values.mean())


def test_train_then_infer_exact_budgets_and_repeated_inference_snapshots():
    sampler = OrderedSampler(steps=1, window=5)
    aggregator = SamplerAggregator(sampler, IdentityIntegrand(), config(inference_samples=8))
    result = aggregator.run()
    assert result is aggregator.accumulator
    assert result.result.n_points == 8
    assert len(aggregator.training_history) == 1
    assert sum(record["update_samples"] for record in aggregator.inference_history) == 8
    first = aggregator.inference_history[0]
    first_times = dict(first["observables"]["processing_times"])
    aggregator.run(mode="infer", inference_samples=2)
    assert result.result.n_points == 10
    assert first["observables"]["processing_times"] == first_times
    assert sum(len(values) for values in sampler.feedback_values) == 5
    assert all(len(batch.weights) <= 3 for _, batch in sampler.generated)


def test_starts_from_existing_inference_accumulator():
    data = LayerData(n_points=2, n_cont=1)
    data.continuous = [[0], [0]]
    data.func_val = [10, 20]
    data.update()
    initial = DefaultAccumulator(data)
    aggregator = SamplerAggregator(OrderedSampler(steps=0), IdentityIntegrand(),
                                  config(mode="infer", inference_samples=1), accumulator=initial)
    assert aggregator.run() is initial
    assert initial.result.n_points == 3
    assert initial.result.real.mean == pytest.approx(31 / 3)


@pytest.mark.parametrize("phase", ["real", "imag", "abs"])
def test_complex_weighted_feedback(phase):
    class ComplexIntegrand(IdentityIntegrand):
        def evaluate_batch(self, continuous, discrete):
            x = continuous[:, 0]
            return (x + 1) * (2 - 3j) + 1j * x**2

    sampler = OrderedSampler(steps=2, window=5)
    aggregator = SamplerAggregator(sampler, ComplexIntegrand(), config(mode="train", training_phase=phase))
    aggregator.run()
    weights = np.concatenate([ComplexIntegrand().evaluate_batch(batch.xs_continuous, batch.xs_discrete) * batch.weights
                              for _, batch in sampler.generated])
    expected = np.abs(weights) if phase == "abs" else getattr(weights, phase)
    np.testing.assert_allclose(np.concatenate(sampler.feedback_values), expected)
    assert len(aggregator.training_history) == sampler.steps
    for step, record in enumerate(aggregator.training_history):
        window = slice(step * sampler.window, (step + 1) * sampler.window)
        values = expected[window]
        mean = values.mean()
        std = values.std(ddof=1)
        rsd = std / abs(mean)
        summary = record["training_result"]
        assert record["training_phase"] == phase
        assert set(summary) == {"mean", "error", "rsd", "tvar"}
        assert all(type(value) is float for value in summary.values())
        assert "training_data" not in record
        assert summary["mean"] == pytest.approx(mean)
        assert summary["error"] == pytest.approx(std / np.sqrt(sampler.window))
        assert summary["rsd"] == pytest.approx(rsd)
        assert summary["tvar"] == pytest.approx(std**2 * record["observables"]["time_total"] / sampler.window)
        assert record["observables"]["imag_mean"] == pytest.approx(weights.imag[window].mean())
        if phase == "abs":
            assert summary["mean"] != pytest.approx(record["observables"]["abs_real_mean"])
            assert summary["rsd"] != pytest.approx(record["observables"]["abs_real_rsd"])


def test_nonfinite_evaluations_are_counted_and_flagged():
    class Nonfinite(IdentityIntegrand):
        def evaluate_batch(self, continuous, discrete):
            values = super().evaluate_batch(continuous, discrete)
            values[continuous[:, 0] == 1] = np.nan
            return values

    aggregator = SamplerAggregator(OrderedSampler(steps=0), Nonfinite(), config(mode="infer", inference_samples=3))
    aggregator.run()
    obs = aggregator.accumulator.get_observables()
    assert obs["n_points"] == 3
    assert obs["n_failures"] == 1
    assert obs["real_mean"] == pytest.approx((1 + 0 + 3 * 1.2) / 3)


def test_worker_failure_propagates_and_prevents_incomplete_resume():
    class Broken(IdentityIntegrand):
        def evaluate_batch(self, continuous, discrete):
            raise ValueError("evaluation failed")

    aggregator = SamplerAggregator(OrderedSampler(), Broken(), config(mode="train"))
    with pytest.raises(ValueError, match="evaluation failed"):
        aggregator.run()
    assert not aggregator.training_history
    assert not aggregator._tasks
    assert not aggregator._running
    with pytest.raises(RuntimeError, match="failed previously"):
        aggregator.run()


def test_abs_phase_overflow_is_flagged_before_feedback():
    class Overflow(IdentityIntegrand):
        def evaluate_batch(self, continuous, discrete):
            return np.full(len(continuous), complex(1.4e308, 1.4e308))

    sampler = OrderedSampler(steps=1, window=1)
    aggregator = SamplerAggregator(sampler, Overflow(), config(mode="train", training_phase="abs"))
    aggregator.run()
    np.testing.assert_array_equal(sampler.feedback_values[0], [0])
    assert aggregator.training_history[0]["observables"]["n_failures"] == 1


def test_generation_batch_size_can_override_sampler_forward_pass_size():
    sampler = OrderedSampler(steps=1, window=7, maximum=3)
    aggregator = SamplerAggregator(sampler, IdentityIntegrand(), config(mode="train", generation_batch_size=5))
    aggregator.run()
    assert [len(batch.weights) for _, batch in sampler.generated] == [5, 2]


def test_waiting_without_pending_work_fails_instead_of_hanging():
    sampler = OrderedSampler()
    sampler.generate = lambda _: GenerationStatus.WAITING
    aggregator = SamplerAggregator(sampler, IdentityIntegrand(), config(mode="train"))
    with pytest.raises(RuntimeError, match="no outstanding"):
        aggregator.run()


@pytest.mark.parametrize("field, value, message", [
    ("xs_continuous", np.array([[1 + 1j]]), "Continuous coordinates must be real"),
    ("weights", np.array([1 + 1j]), "Sample weights must be real"),
    ("xs_discrete", np.array([[0.5]]), "Discrete coordinates must be integers"),
])
def test_sampler_values_are_not_silently_cast(field, value, message):
    sampler = OrderedSampler(steps=0)
    integrand = IdentityIntegrand()
    if field == "xs_discrete":
        integrand.discrete_cardinalities = [2]
    batch = SampleBatch(np.zeros((1, len(integrand.discrete_cardinalities)), dtype=np.uint64),
                        np.zeros((1, 1)), np.ones(1))
    setattr(batch, field, value)
    sampler.generate = lambda _: batch
    aggregator = SamplerAggregator(sampler, integrand, config(mode="infer", inference_samples=1))
    with pytest.raises(ValueError, match=message):
        aggregator.run()


def test_inference_rejects_unfinished_training():
    aggregator = SamplerAggregator(OrderedSampler(), IdentityIntegrand(), config(mode="infer"))
    with pytest.raises(ValueError, match="completed training"):
        aggregator.run()


def test_async_entrypoint_and_sync_guard():
    aggregator = SamplerAggregator(OrderedSampler(steps=0), IdentityIntegrand(), config(mode="infer", inference_samples=1))

    async def run():
        with pytest.raises(RuntimeError, match="event loop"):
            aggregator.run()
        return await aggregator.run_async()

    assert asyncio.run(run()).result.n_points == 1


def test_default_feedback_every_ten_steps_does_not_throttle_history(capsys):
    aggregator = SamplerAggregator(OrderedSampler(steps=11, window=2), IdentityIntegrand(),
                                  config(mode="train", log_every=10))
    aggregator.run()
    assert len(aggregator.training_history) == 11
    output = capsys.readouterr().out
    assert output.count("Training step") == 1
    assert "Training step 10" in output


def test_outstanding_draws_are_bounded_and_cancellation_cleans_up():
    release = Event()
    entered = Event()

    class Blocked(IdentityIntegrand):
        def evaluate_batch(self, continuous, discrete):
            entered.set()
            assert release.wait(5)
            return super().evaluate_batch(continuous, discrete)

    sampler = OrderedSampler(steps=0, maximum=3)
    aggregator = SamplerAggregator(sampler, Blocked(), config(mode="infer", inference_samples=30,
                                                            max_in_flight_batches=2))

    async def run():
        task = asyncio.create_task(aggregator.run_async())
        try:
            async with asyncio.timeout(3):
                while len(sampler.generated) < 2 or not entered.is_set():
                    await asyncio.sleep(0.001)
            await asyncio.sleep(0.01)
            assert len(sampler.generated) == 2
        finally:
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert not aggregator._running
    assert not aggregator._tasks
    assert not aggregator.inference_history
