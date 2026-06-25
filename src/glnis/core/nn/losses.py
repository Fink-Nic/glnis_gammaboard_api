# type:ignore
from typing import Literal

import madnis.integrator as madnis_integrator
import torch


def get_loss(loss_type: str, loss_kwargs: dict | None = None):
    loss = None
    match loss_type.lower():
        case "variance":
            loss = madnis_integrator.losses.stratified_variance
        case "variance_softclip":
            loss = madnis_integrator.losses.stratified_variance_softclip
        case "kl_divergence":
            loss = madnis_integrator.losses.kl_divergence
        case "kl_divergence_softclip":
            loss = madnis_integrator.losses.kl_divergence_softclip
        case "rkl_divergence":
            loss = madnis_integrator.losses.rkl_divergence
        case "test":
            loss = test()
        case _:
            pass

    if loss is None:
        return None

    loss_kwargs = loss_kwargs or {}

    def loss_with_kwargs(*args, **kwargs):
        return loss(*args, **kwargs, **loss_kwargs)

    return loss_with_kwargs


def softclip(x: torch.Tensor, threshold: torch.Tensor = 30.0):
    return threshold * torch.arcsinh(x / threshold)


class test:
    def __init__(
        self,
    ):
        self.step = 0

    def __call__(self,
                 f_true: torch.Tensor,
                 q_test: torch.Tensor,
                 q_sample: torch.Tensor | None = None,
                 channels: torch.Tensor | None = None,
                 type: Literal["variance", "kl_divergence"] = "kl_divergence",
                 threshold: torch.Tensor = 0.0,
                 exponent: torch.Tensor = 1.0,
                 T_max: torch.Tensor = 1000.0,) -> torch.Tensor:
        t = max(1.0 - self.step / T_max, 0.0)
        exp = (exponent - 1.0) * t + 1.0
        self.step += 1
        match type:
            case "variance":
                if threshold > 0.0:
                    return stratified_variance_softclip(
                        f_true.abs()**exp, q_test, q_sample, channels, threshold)
                return madnis_integrator.losses.stratified_variance(
                    f_true.abs()**exp, q_test, q_sample, channels)
            case "kl_divergence":
                if threshold > 0.0:
                    return kl_divergence_softclip(
                        f_true.abs()**exp, q_test, q_sample, channels, threshold)
                return madnis_integrator.losses.kl_divergence(
                    f_true.abs()**exp, q_test, q_sample, channels)
            case _:
                raise ValueError(f"Unknown type: {type}")


def stratified_variance_softclip(
    f_true: torch.Tensor,
    q_test: torch.Tensor,
    q_sample: torch.Tensor | None = None,
    channels: torch.Tensor | None = None,
    threshold: torch.Tensor = 30.0,
):
    """
    Computes the stratified variance as introduced in [2311.01548] for two given sets of
    probabilities, ``f_true`` and ``q_test``. It uses importance sampling with a sampling
    probability specified by ``q_sample``. A soft clipping function is applied to the
    sample weights.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
        channels: channel indices or None in the single-channel case
        threshold: approximate point of transition between linear and logarithmic behavior
    Returns:
        computed stratified variance
    """
    if q_sample is None:
        q_sample = q_test
    if channels is None:
        norm = torch.mean(f_true.detach().abs() / q_sample)
        f_true = softclip(
            f_true / q_sample / norm, threshold) * q_sample * norm
        abs_integral = torch.mean(f_true.detach().abs() / q_sample)
        return madnis_integrator.losses._variance(f_true, q_test, q_sample) / abs_integral.square()

    stddev_sum = 0
    abs_integral = 0
    for i in channels.unique():
        mask = channels == i
        fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
        norm = torch.mean(fi.detach().abs() / qsi)
        fi = softclip(
            fi / qsi / norm, threshold) * qsi * norm
        stddev_sum += torch.sqrt(madnis_integrator.losses._variance(fi, qti,
                                                                    qsi) + madnis_integrator.losses.dtype_epsilon(f_true))
        abs_integral += torch.mean(fi.detach().abs() / qsi)
    return (stddev_sum / abs_integral) ** 2


@madnis_integrator.losses.multi_channel_loss
def kl_divergence_softclip(
    f_true: torch.Tensor,
    q_test: torch.Tensor,
    q_sample: torch.Tensor,
    threshold: torch.Tensor = 30.0,
) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence for two given sets of probabilities, ``f_true`` and
    ``q_test``. It uses importance sampling, i.e. the estimator is divided by an additional factor
    of ``q_sample``. A soft clipping function is applied to the sample weights.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
        channels: channel indices or None in the single-channel case
        threshold: approximate point of transition between linear and logarithmic behavior
    Returns:
        computed KL divergence
    """
    f_true = f_true.detach().abs()
    weight = f_true / q_sample
    weight /= weight.abs().mean()
    clipped_weight = softclip(weight, threshold)
    log_q = torch.log(q_test)
    log_f = torch.log(clipped_weight * q_sample +
                      madnis_integrator.losses.dtype_epsilon(f_true))
    return torch.mean(clipped_weight * (log_f - log_q))
