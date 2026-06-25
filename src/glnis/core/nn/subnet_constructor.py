# type: ignore
import torch
import torch.nn as nn
from typing import Callable, Any


def get_subnet_constructor(subnet_type: str, subnet_kwargs: dict | None = None) -> Callable[[int, int],
                                                                                            nn.Module] | None:
    subnet_kwargs = subnet_kwargs if subnet_kwargs is not None else {}
    constructor = None
    match subnet_type.lower():
        case "mlp_full_conditioning":
            constructor = MLP_full_conditioning
        case _:
            return None

    def constructor_with_kwargs(features_in: int, features_out: int) -> nn.Module:
        return constructor(features_in, features_out, **subnet_kwargs)

    return constructor_with_kwargs


class MLP_full_conditioning(nn.Module):
    """
    Class implementing a standard fully-connected network with 
    conditioning applied at every layer.
    """

    def __init__(
        self,
        features_in: int,
        features_out: int,
        features_c: int = 0,
        layers: int = 3,
        units: int = 32,
        activation: Callable[[], nn.Module] = nn.ReLU,
        layer_constructor: Callable[[int, int], nn.Module] = nn.Linear,
    ):
        super().__init__()
        self.features_c = features_c
        self.activation = activation()

        # Track linear layers cleanly using nn.ModuleList
        self.linears = nn.ModuleList()

        input_dim = features_in
        for i in range(layers - 1):
            self.linears.append(layer_constructor(input_dim, units))
            # The next layer will receive the hidden units + conditioning features
            input_dim = units + features_c

        # Final output layer
        self.linears.append(layer_constructor(input_dim, features_out))

        # Optional: Initialize final layer to zeros if that fits your use-case
        nn.init.zeros_(self.linears[-1].weight)
        nn.init.zeros_(self.linears[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract conditioning features from the end of the input tensor
        if self.features_c > 0:
            c = x[:, -self.features_c:]
        else:
            c = None

        # Process through all hidden layers
        for layer in self.linears[:-1]:
            x = layer(x)
            x = self.activation(x)
            if c is not None:
                x = torch.cat([x, c], dim=1)

        # Final output layer (no activation, no conditioning appended after)
        x = self.linears[-1](x)
        return x
