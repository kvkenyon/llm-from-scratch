import math
from collections.abc import Callable, Iterable

import torch


def lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        return (it * max_learning_rate) / warmup_iters

    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos(((it - warmup_iters) / (cosine_cycle_iters - warmup_iters)) * math.pi)
        ) * (max_learning_rate - min_learning_rate)

    return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    g_l2_norm = torch.sqrt(torch.sum(torch.cat([p.grad.view(-1) for p in parameters if p.grad is not None])))
    if g_l2_norm < max_l2_norm:
        return
    scale = max_l2_norm / (g_l2_norm + 1e-6)
    with torch.no_grad():
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(scale)


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
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data
                lr_t = lr * (math.sqrt(1 - (betas[1] ** t)) / (1 - (betas[0] ** t)))
                p.data -= lr * weight_decay * p.data
                m = (betas[0] * m) + ((1 - betas[0]) * grad)
                v = (betas[1] * v) + ((1 - betas[1]) * (grad**2))
                p.data -= lr_t * (m / (torch.sqrt(v) + eps))
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
