"""Verlet phase updates and momentum-reflected replay."""

import math
from collections.abc import Callable

import torch
from torch import Tensor


Grad = Callable[[Tensor], Tensor]
PhaseState = tuple[Tensor]


class PhaseOptimizer:
    def __init__(self, name: str, *, h: float, fisher: Tensor):
        self._h = h
        self._glide = name == "glide"
        self._low_fisher = fisher <= fisher.median() if name == "fisher" else None

    def init(self, q: Tensor, p: Tensor) -> PhaseState:
        return (p.detach().clone(),)

    def brake_diag(self, q: Tensor) -> Tensor:
        if self._low_fisher is None:
            return torch.ones_like(q)
        return self._low_fisher.to(q)

    def step(
        self, q: Tensor, state: PhaseState, grad: Grad, *, gamma: float
    ) -> tuple[Tensor, PhaseState]:
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        (p,) = state
        half_h = 0.5 * self._h
        scale: float | Tensor | None = None
        if gamma and not self._glide:
            # G is frozen, so both endpoint brakes share the same multiplier.
            scale = math.exp(-half_h * gamma)
            if self._low_fisher is not None:
                scale = torch.where(self._low_fisher, torch.full_like(q, scale), 1.0)
            p = p * scale

        p_half = p - half_h * grad(q)
        next_q = q + self._h * p_half
        next_p = p_half - half_h * grad(next_q)
        if scale is not None:
            next_p = scale * next_p
        return next_q.detach(), (next_p.detach(),)


def unwind(
    opt: PhaseOptimizer,
    q: Tensor,
    state: PhaseState,
    steps: int,
    gradient: Callable[[int], Grad],
) -> tuple[Tensor, PhaseState]:
    """Recover a zero-gate window given its endpoint and indexed batch replay."""
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    (p,) = state
    state = (-p.detach(),)
    for index in range(steps - 1, -1, -1):
        q, state = opt.step(q, state, gradient(index), gamma=0.0)
    return q.detach(), (-state[0],)
