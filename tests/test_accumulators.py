"""Accumulator lifecycle checks for streamed training and inference batches."""

import numpy as np
import pytest

from glnis.core.accumulator import DefaultAccumulator, ProcessingTimes, TrainingAccumulator
from glnis.utils.types import LayerData


def make_data(values, weights=None, elapsed=0.0):
    values = np.asarray(values)
    data = LayerData(n_points=values.size, n_cont=1)
    data.continuous = np.arange(values.size, dtype=float).reshape(-1, 1)
    data.wgt = np.ones(values.size) if weights is None else np.asarray(weights)
    data.func_val = values
    data.update("integrand")
    # Deliberately deterministic timings, independent of test machine speed.
    data.get_processing_times = lambda: {"integrand": elapsed, "total_time": elapsed}
    return data


@pytest.mark.parametrize("values", [[-7.0, 0.0, 2.5, 9.0, -1.5], [1 + 3j, -2 - 4j, 5 + 2j, -1j, -3 + 7j]])
def test_combined_statistics_match_whole_sample(values):
    values = np.asarray(values)
    weights = np.array([0.2, 1.0, 3.0, 0.5, 2.0])
    weighted = values * weights
    accumulator = DefaultAccumulator()
    for batch in (slice(0, 2), slice(2, 3), slice(3, 5)):
        accumulator.combine_with(DefaultAccumulator(make_data(values[batch], weights[batch], elapsed=2.0)))
        accumulator.get_observables()

    result = accumulator.result
    assert result.n_points == len(values)
    assert result.total_time == 6.0
    for phase, expected in (
        (result.real, weighted.real),
        (result.imag, weighted.imag),
        (result.abs_real, np.abs(weighted.real)),
        (result.abs_imag, np.abs(weighted.imag)),
    ):
        assert phase.mean == pytest.approx(np.mean(expected))
        assert phase.error == pytest.approx(np.std(expected, ddof=1) / np.sqrt(len(expected)))
    assert accumulator.get_observables()["time_integrand"] == 6.0


@pytest.mark.parametrize("data", [None, LayerData(), make_data([])])
def test_empty_accumulator_can_report_and_accept_data(data):
    accumulator = DefaultAccumulator(data)
    assert accumulator.result.n_points == 0
    assert accumulator.get_observables()["n_failures"] == 0
    assert "0" in accumulator.str_report()
    accumulator.combine_with(DefaultAccumulator(make_data([3.0, 7.0])))
    assert accumulator.result.real.mean == 5.0
    assert accumulator.max_weight.real_neg_max_wgt == 3.0
    assert accumulator.max_weight.real_pos_max_wgt == 7.0


def test_zero_count_time_and_zero_mean_report():
    assert "0" in ProcessingTimes().str_report()
    assert "Integration Result" in DefaultAccumulator(make_data([-1.0, 1.0])).str_report()


@pytest.mark.parametrize("phase", ["real", "imag", "abs"])
def test_training_preserves_order_and_finalisation_is_idempotent(phase):
    values = np.array([1 + 2j, -3 - 4j, 5 + 6j, -7 + 8j])
    accumulator = TrainingAccumulator(make_data(values[:2]), training_phase=phase)
    accumulator.combine_with(TrainingAccumulator(make_data(values[2:]), training_phase=phase))
    expected = {"real": values.real, "imag": values.imag, "abs": np.abs(values)}[phase]
    np.testing.assert_array_equal(accumulator.training_data, expected)
    combined = accumulator.training_data
    accumulator.finalise()
    accumulator.get_observables()
    assert accumulator.training_data is combined
    assert accumulator.training_result.mean == pytest.approx(np.mean(expected))
    assert accumulator.training_result.error == pytest.approx(np.std(expected, ddof=1) / 2)


def test_observables_are_independent_of_later_updates_and_caller_edits():
    data = make_data([1.0, 4.0], elapsed=2.0)
    accumulator = DefaultAccumulator(data)
    history = accumulator.get_observables()
    old_point = history["real_pos_max_wgt_point"][1].copy()
    accumulator.combine_with(DefaultAccumulator(make_data([8.0], elapsed=3.0)))
    assert history["real_mean"] == 2.5
    assert history["time_total"] == 2.0
    assert history["processing_times"]["integrand"] == 2.0
    np.testing.assert_array_equal(history["real_pos_max_wgt_point"][1], old_point)

    current = accumulator.get_observables()
    current["processing_times"]["integrand"] = -1
    current["real_pos_max_wgt_point"][1][:] = -1
    assert accumulator.get_observables()["processing_times"]["integrand"] == 5.0
    assert accumulator.max_weight.real_pos_max_wgt_point[1][0] == 0


def test_failure_mask_and_history_survive_subsequent_stages():
    data = make_data([np.nan, 2.0, 3.0, 4.0])
    first = DefaultAccumulator(data)
    history = first.get_observables()
    data.func_val = np.array([1.0, np.inf, 3.0, 4.0])
    data.update("integrand")
    data.func_val = np.array([1.0, 2.0, np.nan, 4.0])
    data.update("integrand")
    data.wgt = np.ones(4)
    data.update("later_stage")
    np.testing.assert_array_equal(data.success, [False, False, False, True])
    np.testing.assert_array_equal(data.func_val.ravel(), [0, 0, 0, 4])
    assert len(data.failures["integrand"]) == 3
    assert history["n_failures"] == 1
    assert len(history["failures"]["integrand"]) == 1

    chunks = list(data.as_chunks(2))
    np.testing.assert_array_equal(np.concatenate([chunk.success for chunk in chunks]), data.success)
    combined = DefaultAccumulator.cat([DefaultAccumulator(chunk) for chunk in chunks])
    assert combined.get_observables()["n_failures"] == 3
    assert len(combined.get_observables()["failures"]["integrand"]) == 3


def test_replacing_complex_function_with_real_clears_imaginary_values():
    data = make_data([1 + 2j, 3 + 4j])
    data.func_val = np.array([5.0, 6.0])
    np.testing.assert_array_equal(data.func_val.ravel(), [5.0, 6.0])
