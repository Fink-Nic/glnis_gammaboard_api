"""Runtime callables owned by an orchestrator, outside proposal checkpoints."""
from copy import deepcopy
from dataclasses import dataclass, field

import torch

from glnis.utils.types import BuildFromKind, Config


class SamplerExecution(BuildFromKind):
    KIND = "eager"
    DEFAULT_KIND = "eager"

    def __init__(self, module):
        self.sample = module.sample
        self.prob = module.prob
        self.cuda_graphs = False


@dataclass
class TorchCompileConfig(Config):
    backend: str = "inductor"
    sample: dict | bool = field(default_factory=dict)
    prob: dict | bool = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("execution.backend must be a nonempty string")
        for name, dynamic in (("sample", True), ("prob", False)):
            options = getattr(self, name)
            if not isinstance(options, (dict, bool)):
                raise ValueError(f"execution.{name} must be a table or boolean")
            if options is not False:
                options = {} if options is True else deepcopy(options)
                setattr(self, name, {"fullgraph": True, "dynamic": dynamic, **options})


class TorchCompileExecution(SamplerExecution):
    KIND = "torch_compile"
    DEFAULT_KIND = "torch_compile"

    def __init__(self, module, **kwargs):
        super().__init__(module)
        self.cfg = TorchCompileConfig.from_kwargs(**kwargs)
        for name in ("sample", "prob"):
            options = getattr(self.cfg, name)
            if options is False:
                continue
            options = {"backend": self.cfg.backend, **options}
            setattr(self, name, torch.compile(getattr(module, name), **options))
            self.cuda_graphs |= options["backend"] == "inductor" and (
                options.get("mode") in ("reduce-overhead", "max-autotune")
                or (options.get("options") or {}).get("triton.cudagraphs", False)
            )
