"""Functional PyTorch baselines for the frozen and supplemental runs."""

from collections.abc import Callable
from math import isfinite, sqrt

import torch
from torch import Tensor


Grad = Callable[[Tensor], Tensor]


class SGDM:
    def __init__(self, h: float):
        self.h = h

    def init(self, q: Tensor, p: Tensor) -> tuple[Tensor]:
        return (torch.zeros_like(q),)

    def step(self, q: Tensor, state: tuple[Tensor], grad: Grad, *, gamma: float):
        (velocity,) = state
        velocity = velocity.detach().mul(0.9).add(grad(q).detach())
        return q.detach().add(velocity, alpha=-self.h), (velocity,)


class AdamW:
    def __init__(self, h: float):
        self.h = h

    def init(self, q: Tensor, p: Tensor) -> tuple[Tensor, Tensor, int]:
        return torch.zeros_like(q), torch.zeros_like(q), 0

    def step(self, q: Tensor, state: tuple[Tensor, Tensor, int], grad: Grad, *, gamma: float):
        first, second, count = state
        g = grad(q).detach()
        beta1, beta2 = 0.9, 0.999
        first = first.detach().lerp(g, 1 - beta1)
        second = second.detach().mul(beta2).addcmul(g, g, value=1 - beta2)
        count += 1
        step_size = self.h / (1 - beta1**count)
        denom = second.sqrt().div(sqrt(1 - beta2**count)).add(1e-8)
        q = q.detach().mul(1 - self.h * 0.01).addcdiv(first, denom, value=-step_size)
        return q, (first, second, count)


class EWC(SGDM):
    """SGD with a frozen prior-task anchor and a diagonal quadratic penalty."""

    def __init__(self, h: float, *, fisher: Tensor, anchor: Tensor, strength: float):
        super().__init__(h)
        if not isfinite(strength) or strength < 0:
            raise ValueError("EWC strength must be finite and nonnegative")
        if (anchor.shape, anchor.device, anchor.dtype) != (fisher.shape, fisher.device, fisher.dtype):
            raise ValueError("EWC anchor and Fisher must match in shape, device and dtype")
        if not torch.isfinite(anchor).all() or not (torch.isfinite(fisher) & (fisher >= 0)).all():
            raise ValueError("EWC needs a finite anchor and finite nonnegative Fisher")
        self.anchor = anchor.detach().clone()
        self.precision = fisher.detach().mul(strength)

    def step(self, q: Tensor, state: tuple[Tensor], grad: Grad, *, gamma: float):
        def penalized_grad(x: Tensor) -> Tensor:
            return grad(x).detach() + self.precision * (x.detach() - self.anchor)

        return super().step(q, state, penalized_grad, gamma=gamma)
