import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from glnis.core.config import dataclass
from abc import ABC, abstractmethod

from glnis.utils.types import BuildFromKind, Config


class SubnetConstructor(nn.Module, BuildFromKind):
    """
    Base class for subnetworks used in coupling transforms.
    """
    KIND = "subnet base class"
    DEFAULT_KIND = "mlp"

    def __init__(
        self,
        features_in: int,
        features_c: int,
        features_out: int,
        **kwargs,
    ):
        super().__init__()
        self.features_in = features_in
        self.features_c = features_c
        self.features_out = features_out
        self.input_dim = features_in + features_c


@dataclass
class MLPConfig(Config):
    layers: int = 3
    units: int = 128


class MLP(SubnetConstructor):
    """
    Class implementing a standard fully-connected network.
    """
    KIND = "mlp"
    DEFAULT_KIND = "mlp"

    def __init__(
        self,
        features_in: int,
        features_c: int,
        features_out: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        layer_constructor: Callable[[int, int], nn.Module] = nn.Linear,
        **kwargs
    ):
        """
        Args:
            features_in: number of input features
            features_out: number of output features
            layers: number of layers
            units: number of hidden nodes
            activation: function that builds a nn.Module used as activation function
            layer_construction: function used to construct the network layers, given the number of
                input and output features
        """
        super().__init__(features_in, features_c, features_out)
        self.cfg = MLPConfig.from_kwargs(**kwargs)
        layer_list = []
        input_dim = self.input_dim
        for i in range(self.cfg.layers - 1):
            layer_list.append(layer_constructor(input_dim, self.cfg.units))
            layer_list.append(activation())
            input_dim = self.cfg.units
        layer_list.append(layer_constructor(input_dim, self.features_out))
        nn.init.zeros_(layer_list[-1].weight)
        nn.init.zeros_(layer_list[-1].bias)
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the network.

        Args:
            x: network input, shape (n, features_in)
        Returns:
            network output, shape (n, features_out)
        """
        return self.layers(x)


class MLP_fully_conditioned(SubnetConstructor):
    KIND = "mlp fully conditioned"
    DEFAULT_KIND = "mlp fully conditioned"
    """
    Class implementing a standard fully-connected network with 
    conditioning applied at every layer.
    """

    def __init__(
        self,
        features_in: int,
        features_c: int,
        features_out: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        layer_constructor: Callable[[int, int], nn.Module] = nn.Linear,
        **kwargs
    ):
        """
        Args:
            features_in: number of input features
            features_out: number of output features
            layers: number of layers
            units: number of hidden nodes
            activation: function that builds a nn.Module used as activation function
            layer_construction: function used to construct the network layers, given the number of
                input and output features
        """
        super().__init__(features_in, features_c, features_out)
        self.cfg = MLPConfig.from_kwargs(**kwargs)
        self.activation = activation()

        # Track linear layers cleanly using nn.ModuleList
        self.linears = nn.ModuleList()

        input_dim = self.input_dim
        for i in range(self.cfg.layers - 1):
            self.linears.append(layer_constructor(input_dim, self.cfg.units))
            # The next layer will receive the hidden units + conditioning features
            input_dim = self.cfg.units + features_c

        # Final output layer
        self.linears.append(layer_constructor(input_dim, features_out))

        # Optional: Initialize final layer to zeros if that fits your use-case
        nn.init.zeros_(self.linears[-1].weight)  # type: ignore
        nn.init.zeros_(self.linears[-1].bias)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract conditioning features from the end of the input tensor
        if self.features_c > 0:
            c = x[..., -self.features_c:]
        else:
            c = None

        # Process through all hidden layers
        for layer in self.linears[:-1]:
            x = layer(x)
            x = self.activation(x)
            if c is not None:
                x = torch.cat([x, c], dim=-1)

        # Final output layer (no activation, no conditioning appended after)
        x = self.linears[-1](x)
        return x


class StackedMLP(SubnetConstructor):
    """
    Builds multiple independent MLPs that can be efficiently evaluated in parallel.
    """

    KIND = "stacked mlp"

    def __init__(
        self,
        features_in: int,
        features_c: int,
        features_out: int,
        channels: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        layer_constructor: Callable[[int, int], nn.Module] = nn.Linear,
        **kwargs
    ):
        """
        Args:
            features_in: number of input features
            features_out: number of output features
            channels: number of channels
            layers: number of layers
            units: number of hidden nodes
            activation: function that builds a nn.Module used as activation function
            layer_construction: function used to construct the network layers, given the number of
                input and output features
        """
        super().__init__(features_in, features_c, features_out)
        self.cfg = MLPConfig.from_kwargs(**kwargs)
        self.channels = channels
        self.activation = activation()

        input_dim = self.input_dim
        layer_dims = []
        for i in range(self.cfg.layers - 1):
            layer_dims.append((input_dim, self.cfg.units))
            input_dim = self.cfg.units
        layer_dims.append((input_dim, self.features_out))

        self.weights = nn.ParameterList(
            [torch.empty((channels, n_out, n_in)) for n_in, n_out in layer_dims]
        )
        self.biases = nn.ParameterList(
            [torch.empty((channels, n_out)) for n_in, n_out in layer_dims]
        )
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the network parameters. The parameters of the last layer are initialized to
        zero. Kaiming uniform initializiation is used for the other layers.
        """
        for ws, bs in zip(self.weights[:-1], self.biases[:-1]):
            for w, b in zip(ws, bs):
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(b, -bound, bound)
        nn.init.zeros_(self.weights[-1])
        nn.init.zeros_(self.biases[-1])

    def _forward_single(self, x: torch.Tensor, channel: int) -> torch.Tensor:
        """
        Evaluates the network for a single channel.

        Args:
            x: network input, shape (n, features_in)
            channel: channel index
        Returns:
            network output, shape (n, features_out)
        """
        if x.shape[0] == 0:
            return x.new_zeros((x.shape[0], self.features_out))
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self.activation(F.linear(x, w[channel], b[channel]))
        return F.linear(x, self.weights[-1][channel], self.biases[-1][channel])

    def _forward_uniform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the network for equal numbers of samples in every channel.

        Args:
            x: network input, shape (n, features_in)
        Returns:
            network output, shape (n, features_out)
        """
        batch_size = x.shape[0]
        x = x.reshape(self.channels, batch_size // self.channels, x.shape[1])
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self.activation(torch.baddbmm(b[:, None, :], x, w.transpose(1, 2)))
        return torch.baddbmm(self.biases[-1][:, None, :], x, self.weights[-1].transpose(1, 2)).reshape(
            batch_size, -1
        )

    def forward(
        self,
        x: torch.Tensor,
        channel: list[int] | int | None = None,
    ) -> torch.Tensor:
        """
        Evaluates the network.

        Args:
            x: network input, shape (n, features_in)
            channel: encodes the channel of the samples. It must have one of the following types:

                - `list`: list of integers, specifying the number of samples in each channel;
                - `int`: integer specifying a single channel containing all the samples;
                - `None`: all channels contain the same number of samples.
        Returns:
            network output, shape (n, features_out)
        """
        if isinstance(channel, list):
            return torch.cat(
                [
                    self._forward_single(xi, i)
                    for i, xi in enumerate(x.split(channel, dim=0))
                ],
                dim=0,
            )
        elif isinstance(channel, int):
            return self._forward_single(x, channel)
        else:
            return self._forward_uniform(x)
