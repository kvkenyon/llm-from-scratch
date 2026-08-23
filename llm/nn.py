import torch
from jaxtyping import Float
from torch import Tensor
from einops import rearrange

def softmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    x_stable = x - torch.max(x, dim=dim , keepdim=True)[0]
    return torch.exp(x_stable)/torch.sum(torch.exp(x_stable), dim=dim, keepdim=True)

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
    k_transpose = rearrange(k, "... seq_len d_k -> ... d_k seq_len")
    pre_softmax_attn = (q @ k_transpose / torch.sqrt(torch.tensor(q.shape[-1])))
    if m is not None:
        diff = q.ndim - m.ndim
        new_shape = (1,) * diff + m.shape
        m = torch.reshape(m, new_shape)
        pre_softmax_attn[m == False] += float('-inf')
    attn = softmax(pre_softmax_attn, dim=-1)
    return attn @ v

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

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k:int,  max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        cosines = []
        sines = []
        for i in range(0, max_seq_len):
            for k in range(1, (d_k // 2) + 1):
                theta_i_k = torch.tensor(i) / (theta ** ((2.*k - 2.) / d_k))
                cos = torch.cos(theta_i_k)
                sin = torch.sin(theta_i_k)
                cosines.extend([cos, cos])
                sines.extend([-sin, sin])

        cosines = torch.tensor(cosines, device=device)
        sines = torch.tensor(sines, device=device)

        self.cosines = rearrange(cosines, "(seq_len d_k) -> seq_len d_k", d_k=d_k)
        self.sines = rearrange(sines, "(seq_len d_k) -> seq_len d_k", d_k=d_k)

    def forward(self, q: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cosines = torch.index_select(self.cosines, dim=-2, index=token_positions)
        sines = torch.index_select(self.sines, dim=-2, index=token_positions)


        q_a = rearrange(q, '... seq_len (k two) -> ... seq_len k two', two=2)
        q_b = q_a.flip(-1)
        q_interleaved = rearrange(q_b, '... seq_len k two -> ... seq_len (k two)')

        return q * cosines + (q_interleaved * sines)

