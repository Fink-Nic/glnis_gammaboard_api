"""Small, deterministic fixtures for the current sampler architecture."""
from copy import deepcopy

import pytest
import torch


@pytest.fixture(autouse=True)
def torch_runtime():
    threads = torch.get_num_threads()
    dtype = torch.get_default_dtype()
    device = torch.get_default_device()
    rng = torch.get_rng_state()
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_rng_state(rng)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


@pytest.fixture
def sampler_settings():
    return {
        "sampler": {
            "kind": "mixed flow",
            "flow": {"kind": "coupling", "spline": {"kind": "rqs", "bins": 4},
                     "subnet": {"kind": "mlp", "layers": 2, "units": 8}},
            "discrete": {"kind": "transformer", "embedding_dim": 8, "feedforward_dim": 8,
                         "heads": 2, "layers": 1,
                         "subnet": {"kind": "mlp", "layers": 2, "units": 8}},
        },
        "orchestrator": {"kind": "rectangular", "training_steps": 2, "training_batch_size": 8,
                         "max_batch_size": 4, "use_gpu": False,
                         "loss": {"kind": "kl_divergence"},
                         "scheduler": {"kind": "cosineannealing"}},
    }


@pytest.fixture
def graph_settings():
    return dict(edge_masses=[0.0, 0.0], edge_momentum_shifts=[[0., 0., 0.], [0., 0., 0.]],
                graph_signature=[[1], [1]], lmb_edges=[[0], [1]], generation_lmb_edges=[0], e_cm=1.)
