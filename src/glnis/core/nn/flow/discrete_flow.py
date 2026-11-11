import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

from glnis.utils.types import BuildFromKind

from .continuous_flow import Distribution
from .subnet_constructor import SubnetConstructor

from glnis.utils.types import Config
from dataclasses import dataclass, field


class DiscreteFlow(nn.Module, Distribution, BuildFromKind):
    KIND = "discrete flow base"
    DEFAULT_KIND = "transformer"

    def __init__(
        self,
        dims_in: list[int],
        dims_c: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.dims_in = dims_in
        self.num_discrete_dims = len(dims_in)
        self.dims_c = dims_c

    def discrete_mass(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """ 
        Computes the probability mass of a potentially partial discrete vector x given conditional input c.
        Args:
            x: Tensor of shape (batch_size, n_dims) containing discrete values.
            c: Optional tensor of shape (batch_size, dims_c) containing conditional input.
        Returns:
            Tensor of shape (batch_size,) containing the probability mass for each sample in the batch.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if c is not None and c.ndim == 1:
            c = c.unsqueeze(0)
        return self.prob(x, c)

@dataclass
class TransformerConfig(Config):
    embedding_dim: int = 64
    feedforward_dim: int = 256
    heads: int = 8
    layers: int = 3
    subnet: dict = field(default_factory=lambda: {"kind": "mlp", "layers": 3, "units": 256})


class DiscreteTransformer(DiscreteFlow):
    KIND = "transformer"
    DEFAULT_KIND = "transformer"

    def __init__(
        self,
        dims_in: list[int],
        dims_c: int = 0,
        **kwargs,
    ):
        super().__init__(dims_in, dims_c, **kwargs)
        self.max_dim = max(dims_in)
        self.cfg = TransformerConfig.from_kwargs(**kwargs)

        n_dims = len(dims_in)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.cfg.embedding_dim,
                nhead=self.cfg.heads,
                dim_feedforward=self.cfg.feedforward_dim,
                dropout=0,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=self.cfg.layers,
        )
        self.subnet = SubnetConstructor.from_kwargs(
            features_in=self.cfg.embedding_dim,
            features_c=dims_c,
            features_out=self.max_dim,
            **self.cfg.subnet
        )
        self.x_embedding = nn.Embedding(
            num_embeddings=self.max_dim, embedding_dim=self.cfg.embedding_dim
        )
        self.c_embedding = nn.Linear(dims_c, self.cfg.embedding_dim) if dims_c > 0 else None
        self.pos_embedding = nn.Embedding(
            num_embeddings=n_dims, embedding_dim=self.cfg.embedding_dim
        )
        self.register_buffer(
            "causal_mask",
            torch.ones((n_dims, n_dims), dtype=torch.bool).triu(diagonal=1),
        )
        prior_mask = torch.zeros((n_dims, self.max_dim))
        for i, n_opts in enumerate(self.dims_in):
            prior_mask[i, :n_opts] = 1
        self.register_buffer("prior_mask", prior_mask)
        self.prior_mask: torch.Tensor
        self.causal_mask: torch.Tensor

    def prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_dim = x.shape[1]
        if x_dim > self.num_discrete_dims:
            raise ValueError("too many discrete coordinates")
        if x_dim == 0:
            return self.prior_mask.new_ones(x.shape[0])
        x_embed = F.pad(self.x_embedding(x[:, :-1]), (0, 0, 1, 0))
        pos_embed = self.pos_embedding(
            torch.arange(x_dim, device=x.device)[None, :]
        )
        embedding = x_embed + pos_embed
        if c is not None and self.c_embedding is not None:
            embedding = embedding + self.c_embedding(c)[:, None, :]
        hidden = self.transformer(embedding, self.causal_mask[:x_dim, :x_dim])
        if c is not None:
            hidden = torch.cat((hidden, c[:, None, :].expand(-1, x_dim, -1)), dim=-1)
        net_log_probs = self.subnet(hidden)

        prior_probs = self.prior_mask.expand(x.shape[0], -1, -1).clone()[:, :x_dim, :]

        log_probs = net_log_probs.masked_fill(prior_probs == 0, -torch.inf).log_softmax(dim=-1)
        return log_probs.gather(2, x[:, :, None])[:, :, 0].sum(dim=1).exp()

    def discrete_mass(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """Exact prefix masses; holes marked -1 are marginalized by enumeration."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[1] > self.num_discrete_dims:
            raise ValueError("too many discrete coordinates")
        results = []
        for index, row in enumerate(x):
            fixed = (row >= 0).nonzero().flatten()
            if len(fixed) == 0:
                results.append(self.prior_mask.new_ones(()))
                continue
            end = int(fixed[-1]) + 1
            prefix = row[:end]
            if any(int(prefix[i]) >= self.dims_in[i] for i in range(end)):
                results.append(self.prior_mask.new_zeros(()))
                continue
            holes = (prefix < 0).nonzero().flatten().tolist()
            options = list(product(*(range(self.dims_in[i]) for i in holes)))
            candidates = prefix[None, :].expand(len(options), -1).clone()
            if holes:
                candidates[:, holes] = torch.tensor(options, device=x.device, dtype=x.dtype)
            condition = None if c is None else c[index:index + 1].expand(len(options), -1)
            results.append(self.prob(candidates, condition).sum())
        return torch.stack(results) if results else self.prior_mask.new_empty(0)

    def sample(
        self,
        n: int | None = None,
        c: torch.Tensor | None = None,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if n is None:
            if c is None:
                raise ValueError("Either n or c must be provided.")
            n = len(c)
        if c is not None:
            dtype = c.dtype
            device = c.device

        x = torch.zeros((n, 0), dtype=torch.int64, device=device)
        prob = torch.ones((n,), dtype=dtype, device=device)
        embedding = None
        c_embedding = self.c_embedding(c) if c is not None and self.c_embedding is not None else None
        for i, dim in enumerate(self.dims_in):
            next_embedding = (
                self.pos_embedding(torch.tensor([i], device=device))
                .expand(n, -1)
                .clone()
            )
            if i > 0:
                next_embedding += self.x_embedding(x[:, i - 1])
            if c_embedding is not None:
                next_embedding += c_embedding
            if embedding is None:
                embedding = next_embedding[:, None, :]
            else:
                embedding = torch.cat((embedding, next_embedding[:, None, :]), dim=1)
            hidden = self.transformer(embedding, self.causal_mask[: i + 1, : i + 1])[:, -1]
            if c is not None:
                hidden = torch.cat((hidden, c), dim=-1)
            logits = self.subnet(hidden).masked_fill(self.prior_mask[None, i, :] == 0, -torch.inf)
            probabilities = logits.softmax(dim=-1)
            samples = torch.multinomial(probabilities, 1)
            x = torch.cat((x, samples), dim=1)
            prob = prob * torch.gather(probabilities, 1, samples)[:, 0]

        if return_prob:
            return x, prob
        return x
