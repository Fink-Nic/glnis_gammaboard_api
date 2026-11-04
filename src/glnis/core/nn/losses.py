import torch
from glnis.utils.types import RectangularTrainingStatus

def get_loss(loss_type: str, loss_kwargs: dict | None = None):
    loss = None
    match loss_type.lower():
        case "variance":
            loss = variance
        case "rkl_divergence":
            loss = rkl_divergence
        case "kl_divergence":
            loss = kl_divergence
        case _:
            pass

    if loss is None:
        return None

    loss_kwargs = loss_kwargs or {}

    def loss_with_kwargs(*args, **kwargs):
        return loss(*args, **kwargs, **loss_kwargs)

    return loss_with_kwargs


def softclip(x: torch.Tensor, threshold: float = 30.0):
    return threshold * torch.arcsinh(x / threshold)

def dtype_epsilon(tensor: torch.Tensor) -> float:
    return torch.finfo(tensor.dtype).eps

def kl_divergence(
    status: RectangularTrainingStatus,
    q_test: torch.Tensor,
    threshold: float | None = None,
    normalize_across_training_steps: bool = False,
) -> torch.Tensor:
    f_true = status.ingested_weights.detach().abs()
    q_sample = status.generated_probs
    weight = f_true / q_sample
    if normalize_across_training_steps:
        weight /= status.abs_result.mean
    else:
        weight /= weight.abs().mean()
    if threshold is not None:
        weight = softclip(weight, threshold)
    log_q = torch.log(q_test)
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
    weight = f_true / q_sample
    if normalize_across_training_steps:
        weight /= status.abs_result.mean
    else:
        weight /= weight.abs().mean()
    if threshold is not None:
        weight = softclip(weight, threshold)
    ratio = q_test / q_sample
    log_q = torch.log(q_test)
    log_f = torch.log(weight + dtype_epsilon(weight))
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
