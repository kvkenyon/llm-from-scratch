import torch
from einops import rearrange


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
        self.weight = torch.nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = rearrange(self.weight, "... out_features in_features -> ... in_features out_features")
        return x @ W


class Embedding(torch.nn.Module):
    def __init__(self, num_embedings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(num_embedings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None):
        super().__init__()

        self.eps = eps
        self.d_model = d_model
        self.device = device

        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model)
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + self.eps)
        result = x.div(rms) * self.weight
        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()

        d_ff = 8 // 3 * d_model if not d_ff else d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def _swiglu(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * (self.w3(x)))

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
        cosines = self.cosines[token_positions]
        sines = self.sines[token_positions]

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

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)

        if self.with_rope:
            assert token_positions is not None, "no token positions"
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1)
        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(attn, "... h seq_len d_k -> ... seq_len (h d_k)")

        return self.output_proj(attn)


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff)
        self.attn = CausalMultiHeadAttention(
            num_heads, d_model, with_rope=True, theta=theta, max_seq_len=max_seq_len, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_ids = torch.arange(0, x.shape[-2])
        pos_ids = rearrange(pos_ids, "seq -> 1 seq")
        x = x + self.attn(self.ln1(x), pos_ids)
        y = x + self.ffn(self.ln2(x))
        return y


class TransformerLanguageModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.Sequential()
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(d_model, num_heads, d_ff, theta=theta, max_seq_len=max_seq_len, device=device)
            )
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        x = self.layers(x)
        return self.lm_head(self.ln_final(x))
