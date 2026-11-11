from gammaboard_sampler import GammaBoardSampler
from gammaboard_process import RectangularDomain, Feedback
import numpy as np
import json
from numpy.typing import NDArray
from pathlib import Path


sigma = np.array([0.2, 1.0, 5.0])

training_steps = 50
training_batch_size = 1000
sampler_config = dict(
    kind="mixed flow",
    flow=dict(
        permutations="log",
        spline=dict(
            kind="rqs",
            min_bin_width=1e-3,
            min_bin_height=1e-3,
            min_bin_derivative=1e-3,
            bins=10,
        ),
        subnet=dict(
            kind="mlp",
            layers=3,
            units=32,
        ),
    ),
    discrete=dict(
        kind="transformer",
        embedding_dim=64,
        feedforward_dim=64,
        heads=4,
        layers=1,
        subnet=dict(
            kind="mlp",
            layers=3,
            units=64,
        )
    )
)


def gaussian_eval(discrete: NDArray, continuous: NDArray) -> Feedback:
    # Spherical param
    x, y, z = np.hsplit(continuous, [1, 2])

    r = x/(1-x)
    cos_az = (2*y-1)
    sin_az = np.sqrt(1 - cos_az**2)
    pol = 2*np.pi*z

    momentum = r * np.hstack(
        [sin_az * np.cos(pol), sin_az * np.sin(pol), cos_az])
    jac = 4*np.pi * x**2 / (1 - x)**4

    # Multi Gaussian, normalized to integrate to 1 over the whole space
    norm_factor = np.sum((2*np.pi * sigma ** 2)**(momentum.shape[1]/2))
    sig = sigma[discrete[:, 0]]

    return Feedback(jac.ravel()*np.exp(-(momentum**2).sum(axis=1) / sig**2 / 2) / norm_factor)


def train(n: int, batch_size: int, sampler: GammaBoardSampler) -> None:
    for _ in range(n):
        batch_size = sampler.orchestrator._training_samples_remaining() or 0
        samples = sampler.generate(batch_size)
        res = gaussian_eval(samples.xs_discrete, samples.xs_continuous)
        sampler.feedback(res)


if __name__ == "__main__":
    # Testing exposed functionality
    ddim = (len(sigma), )
    cdim = 3
    domain = RectangularDomain(discrete_cardinalities=ddim, continuous_dims=cdim)
    from_prepared = True and Path("prepared_state.pkl").exists()
    if from_prepared:
        save_path = Path("prepared_state.pkl")
    else:
        save_path = Path("saved_state.pkl")
    init_args = dict(
        save_path=str(save_path),
        sampler=sampler_config, 
        orchestrator=dict(
            training_steps=training_steps,
            training_batch_size=training_batch_size,
            loss_type="kl_divergence",
            loss_kwargs={"threshold": 30.0, "normalize_across_training_steps": True}
        )
    )
    init_args.pop("kind", None)
    sampler = GammaBoardSampler(domain=domain, **init_args)
    snapshot = sampler.snapshot()
    snapshot_path = Path("snapshot.json")
    json.dump(snapshot, snapshot_path.open("w"))

    n_runs = 2
    for run in range(n_runs):
        with snapshot_path.open("rb") as f:
            snapshot = json.load(f)
        print("Testing import of state...")
        if from_prepared:
            snapshot["save_path"] = "prepared_state.pkl"
        sampler = GammaBoardSampler.from_snapshot(
            snapshot=snapshot,
            domain=domain,
            init_args=init_args,
        )
        print(f"============== STARTING RUN {run+1} ==============")
        print("Getting initial samples...")
        samples = sampler.generate(sampler.orchestrator._training_samples_remaining() or 0)
        feedback = gaussian_eval(samples.xs_discrete, samples.xs_continuous)
        res = feedback.values * samples.weights
        mean, std = res.mean(), res.std()
        print(f"Result before training: {mean} +- {std / np.sqrt(1000)}, RSD={std/mean}     TARGET: 1.0")
        sampler.feedback(feedback)
        print("Starting training...")
        train(n=training_steps-1, batch_size=training_batch_size, sampler=sampler)
        
        if run == n_runs - 1:
            snapshot = sampler.snapshot()
            json.dump(snapshot, snapshot_path.open("w"))

        samples = sampler.generate(sampler.orchestrator._training_samples_remaining() or 10000)
        res = gaussian_eval(samples.xs_discrete, samples.xs_continuous).values * samples.weights
        mean, std = res.mean(), res.std()
        print(f"Result after training: {mean} +- {std / np.sqrt(1000)}, RSD={std/mean}     TARGET: 1.0")

    print(f"============== Testing inference Snapshot ==============")
    with snapshot_path.open("rb") as f:
        snapshot = json.load(f)
    print("Testing import of state...")
    if from_prepared:
        snapshot["save_path"] = "prepared_state.pkl"
    sampler = GammaBoardSampler.from_snapshot(
        snapshot=snapshot,
        domain=domain,
        init_args=init_args,
    )
    samples = sampler.generate(sampler.orchestrator._training_samples_remaining() or 10000)
    res = gaussian_eval(samples.xs_discrete, samples.xs_continuous).values * samples.weights
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
