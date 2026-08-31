import math
from collections.abc import Callable

import torch
from einops import rearrange


def init_linear_weights(weights: torch.Tensor, in_features: int, out_features: int) -> torch.nn.Parameter:
    std = 2 / (in_features + out_features)
    torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3 * std, b=3 * std)
    return torch.nn.Parameter(weights)


def softmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    x_stable = x - torch.max(x, dim=dim, keepdim=True)[0]
    return torch.exp(x_stable) / torch.sum(torch.exp(x_stable), dim=dim, keepdim=True)


def log_softmax(x: torch.Tensor) -> torch.Tensor:
    x_stable = x - torch.max(x, dim=-1, keepdim=True)[0]
    return x_stable - x_stable.exp().sum(-1).log().unsqueeze(-1)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = log_softmax(inputs)
    targets = rearrange(targets, "batch_size -> batch_size 1")
    log_probs = torch.gather(log_probs, -1, targets)
    return -torch.mean(log_probs)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor | None = None
) -> torch.Tensor:
    # FLOPS = 2*b*h*(2*d_k * seq_len * seq_len)

    k_transpose = rearrange(k, "... seq_len d_k -> ... d_k seq_len")

    # (b h seq_len d_k) @ (b h d_k seq_len) -> (b h seq_len seq_len)
    pre_softmax_attn = q @ k_transpose / torch.sqrt(torch.tensor(q.shape[-1]))
    if m is not None:
        pre_softmax_attn.masked_fill_(~m, float("-inf"))
    attn = softmax(pre_softmax_attn, dim=-1)

    # (... seq_len seq_len) @ (... seq_len d_k) -> (b h seq_len d_k)
    # FLOPS = b * h * (2*seq_len * seq_len*d_k )
    return attn @ v


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        # W:   (out_features, in_features)
        weights = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = 2 / (in_features + out_features)
        torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = torch.nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:   (..., in_features)
        # W.T: (in_features, out_features)
        W_transpose = rearrange(self.weight, "... out_features in_features -> ... in_features out_features")
        # y:   (..., out_features)
        return x @ W_transpose


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

        # TODO(kevin): Round to closest multiple of 64
        d_ff = 8 // 3 * d_model if not d_ff else d_ff

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def _swiglu(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size context_length d_model)
        # w1_proj = self.w1(x)
        # w1_proj: (batch_size context_length d_ff)
        # w3_proj = self.w3(x)
        # w3_proj: (batch_size context_length d_ff)
        # silu_w1_proj = silu(w1_proj)
        # silu_w1_proj = (batch_size context_length d_ff)
        # glu = silu_w1_proj * w3_proj
        # glu: (batch_size context_length d_ff)
        # swiglu_out = self.w2(glu)
        # swiglu_out: (batch_size context_length d_model)
        return self.w2(silu(self.w1(x)) * (self.w3(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size context_length d_model)
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
        # x: (batch_size, context_length, d_model)
        # FLOPS =  3 * (batch_size * (2 * d_model * context_length * d_model))
        #        + 2 * batch_size * h * (2 * d_k * context_length * context_length)
        #        +     batch_size * (2 * d_model * seq_len * d_model)
        seq_len = x.shape[-2]

        # x: (batch_size context_length d_model)
        Q = self.q_proj(x)
        # Q: (batch_size, context_length, d_model)
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
        # Q,K,V: (batch_size num_heads context_length d_k)
        attn = scaled_dot_product_attention(Q, K, V, mask)
        # attn: (batch_size num_heads context_length d_k)
        attn = rearrange(attn, "... h seq_len d_k -> ... seq_len (h d_k)")
        # attn: (batch_size, context_length, d_model)
        y = self.output_proj(attn)
        # y: (batch_size context_length d_model)
        return y


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
        # x: (batch_size, context_length, d_model)
        pos_ids = torch.arange(0, x.shape[-2])
        pos_ids = rearrange(pos_ids, "seq -> 1 seq")

        x = x + self.attn(self.ln1(x), pos_ids)
        # x: (batch_size context_length d_model)
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
        # in: (batch_size, context_length, )
        self.token_embeddings = Embedding(vocab_size, d_model)
        # out: (batch_size, context_length, d_model)
        self.layers = torch.nn.Sequential()
        for _ in range(num_layers):
            self.layers.append(
                # (batch_size, context_length, d_model)
                TransformerBlock(d_model, num_heads, d_ff, theta=theta, max_seq_len=max_seq_len, device=device)
            )
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a batched, normalized probability distribution over the vocabulary where
        the predicted distribution is over the next word for each input token.
        """
        # x: (batch_size, context_length)
        e = self.token_embeddings(x)
        # e: (batch_size, context_length, d_model)
        attn = self.layers(e)
        # attn: (batch_size, context_length, d_model)
        attn_normalized = self.ln_final(attn)
        # attn_normalized: (batch_size, context_length, d_model)
        y = self.lm_head(attn_normalized)
        # y: (batch_size, context_length, vocab_size)
        return y


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError("lr must be greater than 0")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr} < 0")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                lr_t = lr * (math.sqrt(1 - betas[1]) / (1 - betas[0]))
                p.data -= lr * weight_decay * p.data
                p.data -= lr / math.sqrt(t + 1) * grad
                m = state.get("m", 0)
                v = state.get("v", 0)
                state["m"] = betas[0] * m + (1 - betas[0]) * grad
                state["v"] = betas[1] * v + (1 - betas[1]) * (grad**2)
                p.data -= lr_t * (m / (math.sqrt(v) + eps))
                state["t"] = t + 1
        return loss
