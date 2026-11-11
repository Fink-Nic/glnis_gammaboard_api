from glnis.utils.types import Config
from glnis.utils.types import BuildFromKind



class LayerConstructor(BuildFromKind):
    KIND = "default"
    DEFAULT_KIND = "default"
    """
    A class for constructing layers of a neural network.
    """

    def __init__(self, layer_type: str, **kwargs):
        self.layer_type = layer_type
        self.kwargs = kwargs

    def construct(self):
        """
        Constructs and returns a layer based on the specified type and parameters.
        """
        if self.layer_type == "linear":
            return torch.nn.Linear(**self.kwargs)
        elif self.layer_type == "conv":
            return torch.nn.Conv2d(**self.kwargs)
        elif self.layer_type == "relu":
            return torch.nn.ReLU(**self.kwargs)
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")