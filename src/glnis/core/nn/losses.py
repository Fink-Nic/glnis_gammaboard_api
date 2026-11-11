"""Importance-sampling objectives, selected by configuration ``kind``."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch

from glnis.utils.types import BuildFromKind, Config, RectangularTrainingStatus


@dataclass
class LossConfig(Config):
    threshold: float | None = None
    normalize_across_training_steps: bool = False

    def __post_init__(self):
        if self.threshold is not None and (not math.isfinite(self.threshold) or self.threshold <= 0):
            raise ValueError("loss.threshold must be positive and finite")
        if type(self.normalize_across_training_steps) is not bool:
            raise ValueError("loss.normalize_across_training_steps must be bool")


class Loss(BuildFromKind, ABC):
    KIND = "loss base"
    DEFAULT_KIND = "kl_divergence"

    def __init__(self, **kwargs):
        self.cfg = LossConfig.from_kwargs(**kwargs)

    @abstractmethod
    def __call__(self, status: RectangularTrainingStatus, q_test: torch.Tensor) -> torch.Tensor:
        pass


class KLDivergence(Loss):
    KIND = "kl_divergence"

    def __call__(self, status, q_test):
        return kl_divergence(status, q_test, **self.cfg.to_dict())


class ReverseKLDivergence(Loss):
    KIND = "rkl_divergence"

    def __call__(self, status, q_test):
        return rkl_divergence(status, q_test, **self.cfg.to_dict())


class Variance(Loss):
    KIND = "variance"

    def __init__(self, normalize_across_training_steps: bool = False):
        super().__init__(normalize_across_training_steps=normalize_across_training_steps)

    def __call__(self, status, q_test):
        return variance(status, q_test, normalize_across_training_steps=self.cfg.normalize_across_training_steps)


def get_loss(loss_type: str, loss_kwargs: dict | None = None) -> Loss:
    """Compatibility helper; new configurations use ``Loss.from_kwargs``."""
    return Loss.from_kind(loss_type, **(loss_kwargs or {}))


def softclip(x: torch.Tensor, threshold: float = 30.0):
    return threshold * torch.arcsinh(x / threshold)

def dtype_epsilon(tensor: torch.Tensor) -> float:
    return torch.finfo(tensor.dtype).eps


def _normalized_weight(status, f_true, q_sample, normalize_across_training_steps):
    weight = f_true / q_sample
    norm = status.abs_result.mean if normalize_across_training_steps else weight.mean()
    return weight / max(float(norm), dtype_epsilon(f_true))

def kl_divergence(
    status: RectangularTrainingStatus,
    q_test: torch.Tensor,
    threshold: float | None = None,
    normalize_across_training_steps: bool = False,
) -> torch.Tensor:
    f_true = status.ingested_weights.detach().abs()
    q_sample = status.generated_probs
    weight = _normalized_weight(status, f_true, q_sample, normalize_across_training_steps)
    if threshold is not None:
        weight = softclip(weight, threshold)
    log_q = torch.log(q_test.clamp_min(torch.finfo(q_test.dtype).tiny))
    log_f = torch.log(weight * q_sample + dtype_epsilon(f_true))
    return torch.mean(weight * (log_f - log_q))

def rkl_divergence(
    status: RectangularTrainingStatus,
    q_test: torch.Tensor,
    threshold: float | None = None,
    normalize_across_training_steps: bool = False,
) -> torch.Tensor:
    f_true = status.ingested_weights.detach().abs()
    q_sample = status.generated_probs
    if not torch.any(f_true):
        return q_test.sum() * 0
    weight = _normalized_weight(status, f_true, q_sample, normalize_across_training_steps)
    if threshold is not None:
        weight = softclip(weight, threshold)
    ratio = q_test / q_sample
    log_q = torch.log(q_test.clamp_min(torch.finfo(q_test.dtype).tiny))
    log_f = torch.log((weight * q_sample).clamp_min(dtype_epsilon(weight)))
    return torch.mean(ratio * (log_q - log_f))

def variance(
    status: RectangularTrainingStatus,
    q_test: torch.Tensor,
    # threshold: torch.Tensor | None = None,
    normalize_across_training_steps: bool = False,
) -> torch.Tensor:
    f_true = status.ingested_weights.detach()
    q_sample = status.generated_probs
    if normalize_across_training_steps:
        mean = status.result.mean
        abs_integral = status.abs_result.mean + dtype_epsilon(f_true)
    else:
        weight = f_true / q_sample
        mean = weight.mean()
        abs_integral = weight.abs().mean() + dtype_epsilon(f_true)
    # if threshold is not None:
    #     weight = softclip(weight, threshold)

    ratio = q_test / q_sample
    sq = (f_true / q_test - mean) ** 2
    return (
        torch.mean(sq * ratio)
        if len(f_true) > 0
        else torch.tensor(0.0, device=f_true.device, dtype=f_true.dtype)
    ) / abs_integral**2
