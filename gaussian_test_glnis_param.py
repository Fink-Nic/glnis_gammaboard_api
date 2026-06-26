from dataclasses import asdict

from madnis_sampler import (
    MadnisConfig,
    FlowConfig,
    TransformerConfig,
    MadeConfig,
    MadnisSampler
)
import numpy as np
import json
from numpy.typing import NDArray
from pathlib import Path


sigma = np.array([0.2, 1.0, 5.0])

config = MadnisConfig(
    seed=42,
    training_batch_size=1000,
    training_steps=10,
    max_batch_size=10_000,
    save_path="madnis_blob.pt",
    flow_config=FlowConfig(
        uniform_latent=True,
        permutations="log",
        layers=3,
        units=32,
        bins=10,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_bin_derivative=1e-3
    ),
    transformer_config=TransformerConfig(
        embedding_dim=64,
        feedforward_dim=64,
        heads=4,
        mlp_units=64,
        transformer_layers=1
    ),
    made_config=MadeConfig(
        layers=2,
        nodes_per_feature=16
    )
)

parameterisation = dict(
    layer=dict(
        param_type="spherical"
    )
)
graph_properties = dict(
    edge_masses=[0.0, 0.0],
    edge_momentum_shifts=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    graph_signature=[[1], [1]],
    e_cm=1.0,
    lmb_array=[[0], [1]],
)


def gaussian_eval(discrete: NDArray, momentum: NDArray) -> NDArray:
    # Spherical param

    # Multi Gaussian, normalized to integrate to 1 over the whole space
    norm_factor = np.sum((2*np.pi * sigma ** 2)**(momentum.shape[1]/2))
    sig = sigma[discrete[:, 0]]

    return np.exp(-(momentum**2).sum(axis=1) / sig**2 / 2) / norm_factor


def train(n: int, batch_size: int, sampler: MadnisSampler) -> None:
    for _ in range(n):
        batch_size = sampler.training_samples_remaining() or 0
        samples = sampler.produce_latent_batch(batch_size)
        res = gaussian_eval(samples.xs_discrete, samples.xs_continuous) * samples.weights
        sampler.ingest_training_values(res)


if __name__ == "__main__":
    # Testing exposed functionality
    ddim = [len(sigma)]
    cdim = 3
    from_prepared = True and Path("prepared_state.pkl").exists()
    init_args = asdict(config)
    init_args.update(dict(parameterisation=parameterisation, graph_properties=graph_properties))
    sampler = MadnisSampler(discrete_cardinalities=ddim, continuous_dims=cdim, **init_args)
    print(
        f"sampler.continuous_dims: {sampler.continuous_dims}, sampler.discrete_cardinalities: {sampler.discrete_cardinalities}")
    snapshot = sampler.snapshot()
    save_path = Path(snapshot.get("save_path") or "")
    snapshot_path = Path("snapshot.json")
    json.dump(snapshot, snapshot_path.open("w"))

    for run in range(3):
        print(
            f"sampler.continuous_dims: {sampler.continuous_dims}, sampler.discrete_cardinalities: {sampler.discrete_cardinalities}")
        with snapshot_path.open("rb") as f:
            snapshot = json.load(f)
        print("Testing import of state...")
        if from_prepared:
            snapshot["save_path"] = "prepared_state.pkl"
        sampler = MadnisSampler.from_snapshot(
            snapshot=snapshot,
            discrete_cardinalities=ddim,
            continuous_dims=cdim,
            init_args=init_args,
        )
        print(f"============== STARTING RUN {run+1} ==============")
        print("Getting initial samples...")
        samples = sampler.produce_latent_batch(sampler.training_samples_remaining() or 0)
        res = gaussian_eval(samples.xs_discrete, samples.xs_continuous) * samples.weights
        mean, std = res.mean(), res.std()
        print(f"Result before training: {mean} +- {std / np.sqrt(1000)}, RSD={std/mean}     TARGET: 1.0")
        sampler.ingest_training_values(res)
        print("Starting training...")
        train(n=config.training_steps-1, batch_size=config.training_batch_size, sampler=sampler)
        samples = sampler.produce_latent_batch(sampler.training_samples_remaining() or 10000)
        res = gaussian_eval(samples.xs_discrete, samples.xs_continuous) * samples.weights
        mean, std = res.mean(), res.std()
        print(f"Result after training: {mean} +- {std / np.sqrt(1000)}, RSD={std/mean}     TARGET: 1.0")

    if save_path.exists():
        save_path.unlink()  # Clean up the saved state file
        print("Saved state file removed.")
    else:
        print("Warning: Saved state file not found for cleanup.")

    if snapshot_path.exists():
        snapshot_path.unlink()  # Clean up the saved snapshot file
        print("Saved snapshot file removed.")
    else:
        print("Warning: Saved snapshot file not found for cleanup.")
