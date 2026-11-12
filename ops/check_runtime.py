#!/usr/bin/env python3
"""Check installed sampler training, probabilities and checkpoint recovery."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch

from gammaboard_process import Feedback, RectangularDomain
from gammaboard_sampler import GammaBoardSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or an allocated device such as cuda:0")
    parser.add_argument("--compile", action="store_true", help="exercise the native Inductor policy")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("device must be cpu or cuda:INDEX")
    if device.type == "cuda" and (device.index is None or not torch.cuda.is_available()):
        parser.error("CUDA validation requires cuda:INDEX and a visible GPU; use apptainer exec --nv")
    torch.set_num_threads(1)
    execution = {"kind": "torch_compile" if args.compile else "eager"}
    with tempfile.TemporaryDirectory(prefix="glnis-check-") as directory:
        sampler = GammaBoardSampler(
            domain=RectangularDomain((3,), 6), save_path=str(Path(directory) / "sampler.pt"),
            module={"flow": {"subnet": {"layers": 2, "units": 16}, "spline": {"bins": 4}},
                    "discrete": {"embedding_dim": 8, "feedforward_dim": 16, "heads": 2, "layers": 1,
                                 "subnet": {"layers": 2, "units": 16}}},
            orchestrator={"training_steps": 2, "training_batch_size": 32,
                          "use_gpu": device.type == "cuda", "gpu_id": device.index or 0,
                          "execution": execution})
        assert sampler.orchestrator.device == device
        for _ in range(2):
            batch = sampler.generate(32)
            sampler.feedback(Feedback(1 + np.square(batch.xs_continuous - .4).sum(1)))
        losses = sampler.orchestrator.training_status.loss_history
        assert len(losses) == 2 and np.isfinite(losses).all()
        snapshot = sampler.snapshot()
        expected = sampler.generate(17)
        restored = GammaBoardSampler.from_snapshot(snapshot=snapshot, domain=sampler.domain)
        actual = restored.generate(17)
        np.testing.assert_array_equal(actual.xs_discrete, expected.xs_discrete)
        np.testing.assert_allclose(actual.xs_continuous, expected.xs_continuous, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(actual.weights, expected.weights, rtol=1e-8, atol=1e-9)
        probability = restored.pdf(actual.xs_discrete, actual.xs_continuous)
        np.testing.assert_allclose(probability * actual.weights, 1, rtol=1e-8, atol=1e-9)
    print(f"Sampler checks passed: torch={torch.__version__}, CUDA={torch.version.cuda}, device={device}, compiled={args.compile}")


if __name__ == "__main__":
    main()
