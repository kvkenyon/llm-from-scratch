import torch
from jaxtyping import Float
from torch import Tensor


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        weights = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = 2 / (in_features + out_features)
        torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.W = torch.nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


class Embedding(torch.nn.Module):
    def __init__(self, num_embedings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        embeddings = torch.empty(num_embedings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(embeddings, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.embeddings = torch.nn.Parameter(embeddings)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.d_model = d_model

        self.gain = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model)
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + self.eps)
        result = x.div(rms) * self.gain
        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()

        d_ff = 8 // 3 * d_model if not d_ff else d_ff
        w1: Float[Tensor, " d_ff d_model"] = torch.empty(d_ff, d_model)
        w2: Float[Tensor, " d_model d_ff"] = torch.empty(d_model, d_ff)
        w3: Float[Tensor, " d_ff d_model"] = torch.empty(d_ff, d_model)
        std = 2 / (d_ff + d_model)
        torch.nn.init.trunc_normal_(w1, mean=0.0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(w2, mean=0.0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(w3, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.w1 = torch.nn.Parameter(w1)
        self.w2 = torch.nn.Parameter(w2)
        self.w3 = torch.nn.Parameter(w3)

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def _swiglu(self, x: torch.Tensor) -> torch.Tensor:
        return (self._silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._swiglu(x)
