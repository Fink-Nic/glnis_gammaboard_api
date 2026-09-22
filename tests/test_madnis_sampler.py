# type:ignore
from __future__ import annotations

import numpy as np
import torch

from madnis_sampler import MadnisSampler


def test_pdf_masks_cardinality_one_discrete_dimensions(monkeypatch) -> None:
    sampler = MadnisSampler(
        discrete_cardinalities=[1, 3],
        continuous_dims=2,
        use_gpu=False,
        training_steps=1,
        training_batch_size=4,
        max_batch_size=16,
    )

    seen: list[torch.Tensor] = []

    def fake_prob(x: torch.Tensor) -> torch.Tensor:
        seen.append(x.detach().clone())
        return torch.ones(len(x), device=x.device, dtype=x.dtype)

    monkeypatch.setattr(sampler.madnis.flow, "prob", fake_prob)

    xs_discrete = np.array([[0, 1], [99, 2]], dtype=np.uint64)
    xs_continuous = np.full((2, 2), 0.5, dtype=np.float64)
    pdf = sampler.pdf(xs_discrete, xs_continuous)

    assert sampler.madnis.integrand.discrete_dims == [3]
    assert len(seen) == 1
    np.testing.assert_allclose(
        seen[0].numpy(),
        [[1.0, 0.5, 0.5], [2.0, 0.5, 0.5]],
    )
    np.testing.assert_allclose(pdf, np.ones(2))
