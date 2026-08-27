import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


def init_linear_weights(weights: torch.Tensor, in_features: int, out_features: int) -> torch.nn.Parameter:
    std = 2 / (in_features + out_features)
    torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3 * std, b=3 * std)
    return torch.nn.Parameter(weights)


def softmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    x_stable = x - torch.max(x, dim=dim, keepdim=True)[0]
    return torch.exp(x_stable) / torch.sum(torch.exp(x_stable), dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor | None = None
) -> torch.Tensor:
    k_transpose = rearrange(k, "... seq_len d_k -> ... d_k seq_len")
    pre_softmax_attn = q @ k_transpose / torch.sqrt(torch.tensor(q.shape[-1]))
    if m is not None:
        pre_softmax_attn.masked_fill_(~m, float("-inf"))
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
    def __init__(self, d_model: int, eps: float = 1e-5, device=None):
        super().__init__()

        self.eps = eps
        self.d_model = d_model
        self.device = device

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
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        cosines = []
        sines = []
        for i in range(max_seq_len):
            for k in range(1, (d_k // 2) + 1):
                theta_i_k = torch.tensor(i) / (theta ** ((2.0 * k - 2.0) / d_k))
                cos = torch.cos(theta_i_k)
                sin = torch.sin(theta_i_k)
                cosines.extend([cos, cos])
                sines.extend([-sin, sin])

        cosines = torch.tensor(cosines, device=device)
        sines = torch.tensor(sines, device=device)

        self.cosines = rearrange(cosines, "(seq_len d_k) -> seq_len d_k", d_k=d_k)
        self.sines = rearrange(sines, "(seq_len d_k) -> seq_len d_k", d_k=d_k)

    def forward(self, q: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # TODO(kevin): Figure out how to apply batched indeces
        cosines = torch.index_select(self.cosines, dim=-2, index=token_positions)
        sines = torch.index_select(self.sines, dim=-2, index=token_positions)

        q_a = rearrange(q, "... seq_len (k two) -> ... seq_len k two", two=2)
        q_b = q_a.flip(-1)
        q_interleaved = rearrange(q_b, "... seq_len k two -> ... seq_len (k two)")

        return q * cosines + (q_interleaved * sines)


class CausalMultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        *,
        with_rope: bool = False,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = self.d_k = d_model // num_heads

        self.with_rope = with_rope
        self.theta = theta
        self.max_seq_len = max_seq_len

        if self.with_rope:
            assert self.theta is not None and self.max_seq_len is not None, "Invalid rope config"
            self.rope = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len, device)

        self.device = device or torch.device("cpu")

        self.Wq = init_linear_weights(torch.empty(d_model, d_model), d_model, d_model)
        self.Wk = init_linear_weights(torch.empty(d_model, d_model), d_model, d_model)
        self.Wv = init_linear_weights(torch.empty(d_model, d_model), d_model, d_model)
        self.Wo = init_linear_weights(torch.empty(d_model, d_model), d_model, d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        Wq = rearrange(self.Wq, "... dm1 dm2 -> ... dm2 dm1")
        Wk = rearrange(self.Wk, "... dm1 dm2 -> ... dm2 dm1")
        Wv = rearrange(self.Wv, "... dm1 dm2 -> ... dm2 dm1")
        Wo = rearrange(self.Wo, "... dm1 dm2 -> ... dm2 dm1")

        Q = x @ Wq
        K = x @ Wk
        V = x @ Wv

        Q = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)

        if self.with_rope:
            assert token_positions is not None, "no token positions"
            # TODO(kevin): Figure out how to apply batched indeces
            Q = self.rope(Q, token_positions[0])
            K = self.rope(K, token_positions[0])

        mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1)
        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(attn, "... h seq_len d_k -> ... seq_len (h d_k)")

        return attn @ Wo
