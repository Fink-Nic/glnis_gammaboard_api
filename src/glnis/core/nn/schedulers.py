"""PyTorch learning-rate schedulers selected by configuration ``kind``.

Implementations receive the optimizer and the configured training horizon.
They expose PyTorch's step/state_dict/load_state_dict interface directly.
"""

import torch

from glnis.utils.types import BuildFromKind


class Scheduler(torch.optim.lr_scheduler._LRScheduler, BuildFromKind):
    KIND = "scheduler base"
    DEFAULT_KIND = "cosineannealing"


class CosineAnnealing(torch.optim.lr_scheduler.CosineAnnealingLR, Scheduler):
    KIND = "cosineannealing"

    def __init__(self, optimizer, training_steps: int, T_max: int | None = None, eta_min: float = 0.0):
        horizon = max(training_steps, 1) if T_max is None else T_max
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("scheduler.T_max must be a positive integer")
        super().__init__(optimizer, T_max=horizon, eta_min=eta_min)


class Step(torch.optim.lr_scheduler.StepLR, Scheduler):
    KIND = "step"

    def __init__(self, optimizer, training_steps: int, step_size: int, gamma: float = 0.1):
        if type(step_size) is not int or step_size <= 0:
            raise ValueError("scheduler.step_size must be a positive integer")
        super().__init__(optimizer, step_size=step_size, gamma=gamma)


class Exponential(torch.optim.lr_scheduler.ExponentialLR, Scheduler):
    KIND = "exponential"

    def __init__(self, optimizer, training_steps: int, gamma: float):
        super().__init__(optimizer, gamma=gamma)
