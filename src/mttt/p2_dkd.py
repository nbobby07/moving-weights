"""One-force position Verlet with synchronous phase endpoints."""

from collections.abc import Callable

import torch
from torch import Tensor

from .phase import Grad, PhaseState


def normalized_fisher(fisher: Tensor) -> Tensor:
    fisher = fisher.detach()
    return fisher / fisher.mean().clamp_min(1e-12)


class DKD:
    def __init__(self, h: float, *, inv_mass: Tensor | None = None):
        self._h = h
        self._half_h = h / 2
        self._half_drift = None if inv_mass is None else self._half_h * inv_mass.detach()

    def init(self, q: Tensor, p: Tensor) -> PhaseState:
        return (p.detach().clone(),)

    def _drift(self, q: Tensor, p: Tensor) -> Tensor:
        if self._half_drift is None:
            return q.add(p, alpha=self._half_h)
        return torch.addcmul(q, p, self._half_drift)

    def step(
        self, q: Tensor, state: PhaseState, grad: Grad, *, gate: float
    ) -> tuple[Tensor, PhaseState]:
        (p,) = state
        midpoint = self._drift(q, p)
        next_p = p.add(grad(midpoint), alpha=-self._h)
        next_q = self._drift(midpoint, next_p)
        return next_q.detach(), (next_p.detach(),)


def unwind(
    opt, q: Tensor, state: PhaseState, steps: int, gradient: Callable[[int], Grad]
) -> tuple[Tensor, PhaseState]:
    """Replay the original force laws from a native or lattice zero-gate endpoint."""
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    (p,) = state
    reflected = (-p.detach(),)
    for index in range(steps - 1, -1, -1):
        q, reflected = opt.step(q, reflected, gradient(index), gate=0.0)
    return q.detach(), (-reflected[0],)
