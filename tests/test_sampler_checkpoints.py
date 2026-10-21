from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from gammaboard_process import GenerationStatus

from madnis_sampler import (
    MadnisConfig,
    MadnisSampler,
    MadnisSamplerFullSgnStrat,
    MadnisSamplerNaiveSgnStrat,
)


STRATEGIES = [
    ("sgn_strat_naive", MadnisSamplerNaiveSgnStrat),
    ("sgn_strat_full", MadnisSamplerFullSgnStrat),
]


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def make_sampler(tmp_path, kind, *, training_steps=2, sampler_type=None, continuous_dims=2,
                 use_gpu=False, cuda_id=1, **initialization):
    cfg = MadnisConfig.from_dict({
        "kind": kind,
        "training_steps": training_steps,
        "training_batch_size": 8,
        "max_batch_size": 4,
        "use_gpu": use_gpu,
        "cuda_id": cuda_id,
        "save_path": str(tmp_path / "sampler.pt"),
        "flow_config": {"layers": 2, "units": 16, "bins": 4},
        "transformer_config": {
            "embedding_dim": 8, "feedforward_dim": 8, "heads": 2,
            "mlp_units": 8, "transformer_layers": 1,
        },
    })
    if sampler_type is not None:
        return sampler_type(continuous_dims=continuous_dims, discrete_cardinalities=[3], cfg=cfg, **initialization)
    return MadnisSampler.from_config(
        continuous_dims=continuous_dims, discrete_cardinalities=[3], cfg=cfg, **initialization,
    )


def leaves(sampler):
    if isinstance(sampler, MadnisSamplerFullSgnStrat):
        return [sampler.madnis_1, sampler.madnis_2]
    return [sampler]


def weighted_values(batch, strata):
    # Both sign channels receive a nonconstant, nonzero training target.
    # The checkpoint tests isolate recovery from sign-rejection statistics.
    magnitude = 0.3 + batch.xs_continuous[:, 0] + batch.xs_continuous[:, 1] ** 2
    return np.where(strata == 0, 1.0, -1.0) * magnitude * batch.weights


def train_one_step(sampler):
    batch, strata = unpack_generation(sampler.generate(8))
    sampler.feedback(weighted_values(batch, strata))


def unpack_generation(result):
    if isinstance(result, tuple):
        return result
    return result, np.zeros(len(result.weights), dtype=np.int64)


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def sampler_state(sampler):
    samplers = [sampler] if leaves(sampler) == [sampler] else [sampler, *leaves(sampler)]
    statuses = []
    for item in samplers:
        status = item.training_status
        state = {
            key: deepcopy(value) for key, value in vars(status).items()
            if not isinstance(value, torch.Tensor)
        }
        # The unused parts of partially filled buffers are uninitialized memory.
        state["generated_samples"] = status.generated_samples[:status.num_generated].clone()
        state["generated_probs"] = status.generated_probs[:status.num_generated].clone()
        state["ingested_weights"] = status.ingested_weights[:status.num_ingested].clone()
        statuses.append(state)
    models = []
    for item in leaves(sampler):
        integrator = item.madnis
        models.append({
            "cfg": asdict(item.cfg),
            "parameters": deepcopy(integrator.state_dict()),
            "optimizer": deepcopy(integrator.optimizer.state_dict()),
            "scheduler": None if integrator.scheduler is None else deepcopy(integrator.scheduler.state_dict()),
            "step": integrator.step,
        })
    return {
        "cfg": asdict(sampler.cfg), "statuses": statuses, "models": models,
        "pending_strata": getattr(sampler, "pending_strata", None).copy()
        if hasattr(sampler, "pending_strata") else None,
    }


def probe_pdf(sampler):
    # PDF inputs contain the sign channel followed by an ordinary discrete
    # integration channel. Interleave signs to check routing to both models.
    discrete = np.array([[0, 0], [1, 2], [0, 1], [1, 0]], dtype=np.int64)
    continuous = np.array([[0.1, 0.2], [0.8, 0.3], [0.4, 0.9], [0.7, 0.6]])
    return sampler.pdf(discrete, continuous)


def restore(snapshot):
    # Exercise the actual JSON-safe metadata boundary; no caller configuration
    # is supplied to hide missing persisted settings or strategy information.
    return MadnisSampler.from_snapshot(
        snapshot=json.loads(json.dumps(snapshot)),
        continuous_dims=2, discrete_cardinalities=[3],
    )


def draw(sampler, n):
    batch, strata = unpack_generation(sampler.generate(n))
    return {
        "discrete": batch.xs_discrete.copy(),
        "continuous": batch.xs_continuous.copy(),
        "weights": batch.weights.copy(),
        "strata": strata.copy(),
        "training_remaining": batch.training_remaining,
    }, weighted_values(batch, strata)


@pytest.mark.parametrize("kind,sampler_type", STRATEGIES)
@pytest.mark.parametrize("position", ["between_batches", "awaiting_feedback"])
def test_checkpoint_resumes_partial_training_exactly(tmp_path, kind, sampler_type, position):
    sampler = make_sampler(tmp_path, kind)
    train_one_step(sampler)
    assert sampler.training_status.step == 1
    assert all(item.madnis.optimizer.state for item in leaves(sampler))

    count, ingested = (4, 2) if position == "between_batches" else (8, 5)
    batch, strata = sampler.generate(count)
    values = weighted_values(batch, strata)
    sampler.feedback(values[:ingested])
    if kind == "sgn_strat_full" and position == "awaiting_feedback":
        assert sampler.madnis_1.training_status.step == 2
        assert sampler.madnis_2.training_status.step == 1
        assert sampler.madnis_2.training_status.num_ingested == 1
        np.testing.assert_array_equal(sampler.pending_strata, [1, 1, 1])

    before = sampler_state(sampler)
    before_pdf = probe_pdf(sampler)
    snapshot = sampler.snapshot()
    assert_nested_equal(sampler_state(sampler), before)

    def finish(candidate):
        generated = None
        next_values = None
        if position == "between_batches":
            generated, next_values = draw(candidate, 4)
        candidate.feedback(values[ingested:])
        if next_values is not None:
            candidate.feedback(next_values)
        assert candidate.training_samples_remaining() is None
        assert candidate.training_status.step == 2
        assert candidate.training_status.total_trained_samples == 16
        assert all(item.training_status.step == 2 for item in leaves(candidate))
        state = sampler_state(candidate)
        pdf = probe_pdf(candidate)
        inference, _ = draw(candidate, 4)
        assert inference["training_remaining"] is None
        return generated, state, pdf, inference

    expected = finish(sampler)
    recovered = restore(snapshot)
    assert type(recovered) is sampler_type
    assert recovered.discrete_cardinalities == [2, 3]
    assert recovered.get_strat_cardinalities() == [2]
    assert_nested_equal(sampler_state(recovered), before)
    np.testing.assert_array_equal(probe_pdf(recovered), before_pdf)
    if position == "awaiting_feedback":
        assert recovered.generate(1) is GenerationStatus.WAITING
    assert_nested_equal(finish(recovered), expected)


@pytest.mark.parametrize("kind,sampler_type", STRATEGIES)
def test_checkpoint_preserves_completed_training_and_inference(tmp_path, kind, sampler_type):
    sampler = make_sampler(tmp_path, kind, training_steps=1)
    train_one_step(sampler)
    before = sampler_state(sampler)
    before_pdf = probe_pdf(sampler)
    snapshot = sampler.snapshot()
    expected, _ = draw(sampler, 4)

    recovered = restore(snapshot)
    assert type(recovered) is sampler_type
    assert recovered.training_samples_remaining() is None
    assert_nested_equal(sampler_state(recovered), before)
    np.testing.assert_array_equal(probe_pdf(recovered), before_pdf)
    actual, _ = draw(recovered, 4)
    assert_nested_equal(actual, expected)
    assert actual["training_remaining"] is None
    assert recovered.training_status.step == 1


@pytest.mark.parametrize("kind,sampler_type", STRATEGIES)
@pytest.mark.parametrize("fail_save", [False, True])
def test_checkpoint_leaves_live_integrator_callbacks_intact(tmp_path, monkeypatch, kind, sampler_type, fail_save):
    sampler = make_sampler(tmp_path, kind)
    objects = []
    for item in leaves(sampler):
        for obj in [item.madnis, item.madnis.flow, item.madnis.flow.continuous_flow, item.madnis.flow.discrete_flow]:
            if obj is None:
                continue
            for name in ["integrand", "loss", "channel_remap_function", "prior_prob_function"]:
                if hasattr(obj, name):
                    objects.append((obj, name, getattr(obj, name)))
    before_rng = torch.get_rng_state().clone()
    before_pdf = probe_pdf(sampler)

    if fail_save:
        def fail(*args, **kwargs):
            raise OSError("simulated checkpoint write failure")

        with monkeypatch.context() as patch:
            patch.setattr(torch, "save", fail)
            with pytest.raises(RuntimeError, match="simulated checkpoint write failure"):
                sampler.snapshot()
    else:
        sampler.snapshot()

    for obj, name, value in objects:
        assert getattr(obj, name) is value
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    np.testing.assert_array_equal(probe_pdf(sampler), before_pdf)
    train_one_step(sampler)
    assert sampler.training_status.step == 1
    assert sampler.training_status.last_loss is not None


@pytest.mark.parametrize("kind,sampler_type", STRATEGIES)
def test_direct_subclass_checkpoint_preserves_actual_strategy(tmp_path, kind, sampler_type):
    sampler = make_sampler(tmp_path, "default", sampler_type=sampler_type)
    snapshot = sampler.snapshot()
    expected, _ = draw(sampler, 4)
    recovered = restore(snapshot)
    assert type(recovered) is sampler_type
    actual, _ = draw(recovered, 4)
    assert_nested_equal(actual, expected)


@pytest.mark.parametrize("kind,sampler_type", STRATEGIES)
@pytest.mark.parametrize("training_steps", [1, 2])
def test_resume_feedback_before_remaining_generation(tmp_path, kind, sampler_type, training_steps):
    sampler = make_sampler(tmp_path, kind, training_steps=training_steps)
    batch, strata = sampler.generate(4)
    values = weighted_values(batch, strata)
    sampler.feedback(values[:2])
    snapshot = sampler.snapshot()

    def finish(candidate):
        # Finish the first child's feedback before drawing the sibling's
        # samples. The first child must wait for the parent step, including
        # when it has already completed all of its configured training.
        candidate.feedback(values[2:])
        generated, next_values = draw(candidate, 4)
        if kind == "sgn_strat_full":
            np.testing.assert_array_equal(generated["strata"], np.ones(4))
            assert candidate.madnis_1.training_status.step == 1
        candidate.feedback(next_values)
        assert candidate.training_status.step == 1
        assert all(item.training_status.step == 1 for item in leaves(candidate))
        for _ in range(training_steps - 1):
            train_one_step(candidate)
        assert candidate.training_samples_remaining() is None
        state = sampler_state(candidate)
        inference, _ = draw(candidate, 4)
        return generated, state, inference

    expected = finish(sampler)
    recovered = restore(snapshot)
    assert type(recovered) is sampler_type
    assert_nested_equal(finish(recovered), expected)


@pytest.mark.parametrize("kind", ["default", "sgn_strat_naive"])
def test_legacy_payload_restores_training_state_with_init_args(tmp_path, kind):
    sampler = make_sampler(tmp_path, kind)
    train_one_step(sampler)
    _, values = draw(sampler, 4)
    sampler.feedback(values[:2])
    before = sampler_state(sampler)
    sampler.snapshot()
    checkpoint = torch.load(sampler.cfg.save_path, map_location="cpu", weights_only=False)
    legacy = {name: checkpoint[name] for name in (
        "training_status", "torch_cpu_rng_state", "madnis_blob",
    )}
    legacy["torch_gpu_rng_state"] = None
    torch.save(legacy, sampler.cfg.save_path)
    expected, _ = draw(sampler, 4)

    recovered = MadnisSampler.from_snapshot(
        snapshot={"save_path": sampler.cfg.save_path, "graph_properties": None},
        continuous_dims=2, discrete_cardinalities=[3], init_args=asdict(sampler.cfg),
    )
    assert type(recovered) is type(sampler)
    assert_nested_equal(sampler_state(recovered), before)
    actual, _ = draw(recovered, 4)
    assert_nested_equal(actual, expected)


def test_naive_checkpoint_restores_spherical_mapping_and_pdf(tmp_path):
    sampler = make_sampler(
        tmp_path, "sgn_strat_naive", training_steps=0, continuous_dims=3,
        mapping={"layer_cfgs": [{"kind": "spherical", "conformal_scale": 2.0}]},
        graph_properties={
            "edge_masses": [0.0], "edge_momentum_shifts": [[0.0, 0.0, 0.0]],
            "graph_signature": [[1]], "lmb_edges": [[0]],
            "generation_lmb_edges": [0], "e_cm": 0.0,
        },
        evaluator_metadata={"checkpoint_test": "spherical mapping"},
    )
    batch, strata = sampler.generate(4)
    discrete = np.column_stack((strata, batch.xs_discrete))
    before_pdf = sampler.pdf(discrete, batch.xs_continuous)
    assert np.all(before_pdf > 0)
    snapshot = sampler.snapshot()
    expected, _ = draw(sampler, 4)

    recovered = MadnisSampler.from_snapshot(
        snapshot=json.loads(json.dumps(snapshot)), continuous_dims=3,
        discrete_cardinalities=[3],
    )
    assert type(recovered) is MadnisSamplerNaiveSgnStrat
    assert recovered.mapping is not None
    assert recovered.evaluator_metadata == sampler.evaluator_metadata
    assert_nested_equal(
        recovered.mapping.graph_properties.as_init_dict(),
        sampler.mapping.graph_properties.as_init_dict(),
    )
    np.testing.assert_array_equal(recovered.pdf(discrete, batch.xs_continuous), before_pdf)
    actual, _ = draw(recovered, 4)
    assert_nested_equal(actual, expected)


@pytest.mark.parametrize("kind", ["default", "sgn_strat_naive", "sgn_strat_full"])
def test_snapshot_without_save_path_does_not_create_files(tmp_path, kind):
    sampler = make_sampler(tmp_path, kind)
    sampler.cfg.save_path = None
    assert "Warning" in sampler.snapshot()
    assert list(tmp_path.iterdir()) == []


def test_failed_full_checkpoint_replacement_preserves_previous_file(tmp_path, monkeypatch):
    sampler = make_sampler(tmp_path, "sgn_strat_full")
    assert sampler.madnis_1.cfg is not sampler.madnis_2.cfg
    assert sampler.madnis_1.cfg.seed == sampler.cfg.seed
    assert sampler.madnis_2.cfg.seed == sampler.cfg.seed + 1
    snapshot = sampler.snapshot()
    path = tmp_path / "sampler.pt"
    original = path.read_bytes()
    original_state = sampler_state(sampler)
    _, values = draw(sampler, 4)
    sampler.feedback(values[:2])
    pending_state = sampler_state(sampler)
    save = torch.save

    def fail_file_write(obj, destination, *args, **kwargs):
        if isinstance(obj, dict) and "checkpoint_version" in obj:
            destination.write(b"partial checkpoint")
            raise OSError("simulated file write failure")
        return save(obj, destination, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail_file_write)
        with pytest.raises(RuntimeError, match="simulated file write failure"):
            sampler.snapshot()
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    assert_nested_equal(sampler_state(sampler), pending_state)
    recovered = restore(snapshot)
    assert type(recovered) is MadnisSamplerFullSgnStrat
    assert recovered.madnis_1.cfg is not recovered.madnis_2.cfg
    assert_nested_equal(sampler_state(recovered), original_state)


@pytest.mark.parametrize("kind,sampler_type", STRATEGIES)
def test_cuda_checkpoint_restores_each_device_rng_and_training(tmp_path, kind, sampler_type):
    device_count = 2 if kind == "sgn_strat_full" else 1
    if not torch.cuda.is_available() or torch.cuda.device_count() < device_count:
        pytest.skip(f"requires {device_count} CUDA device(s)")
    if any(not (7, 0) <= torch.cuda.get_device_capability(index) < (12, 0)
           for index in range(device_count)):
        pytest.skip("requires CUDA devices supported by MadnisSampler")

    sampler = make_sampler(tmp_path, kind, use_gpu=True, cuda_id=0)
    assert [item.device for item in leaves(sampler)] == [
        torch.device(f"cuda:{index}") for index in range(device_count)
    ]
    train_one_step(sampler)
    batch, strata = sampler.generate(4)
    values = weighted_values(batch, strata)
    sampler.feedback(values[:2])
    before = sampler_state(sampler)
    before_pdf = probe_pdf(sampler)
    snapshot = sampler.snapshot()
    saved_rng = [torch.cuda.get_rng_state(index).clone() for index in range(device_count)]

    def finish(candidate):
        candidate.feedback(values[2:])
        generated, next_values = draw(candidate, 4)
        candidate.feedback(next_values)
        assert candidate.training_samples_remaining() is None
        assert all(item.training_status.step == 2 for item in leaves(candidate))
        state = sampler_state(candidate)
        pdf = probe_pdf(candidate)
        inference, _ = draw(candidate, 4)
        return generated, state, pdf, inference

    expected = finish(sampler)
    # Ensure recovery restores every child device, regardless of which device
    # happened to be current when the parent wrote the checkpoint.
    for index in range(device_count):
        torch.rand(11, device=f"cuda:{index}")
    recovered = restore(snapshot)
    assert type(recovered) is sampler_type
    assert [item.device for item in leaves(recovered)] == [
        torch.device(f"cuda:{index}") for index in range(device_count)
    ]
    for index, expected_rng in enumerate(saved_rng):
        torch.testing.assert_close(torch.cuda.get_rng_state(index), expected_rng, rtol=0, atol=0)
    assert_nested_equal(sampler_state(recovered), before)
    np.testing.assert_array_equal(probe_pdf(recovered), before_pdf)
    assert_nested_equal(finish(recovered), expected)
