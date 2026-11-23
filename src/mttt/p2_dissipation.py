"""Endpoint energy brakes and the exact GENERIC anchor subflow."""

from math import isfinite

import torch

from .p2_dkd import DKD, normalized_fisher


class EnergyBrake(DKD):
    def __init__(self, name, *, h, fisher):
        super().__init__(h)
        if name == "iso":
            self.mask = None
        elif name in ("low", "high"):
            if fisher is None or fisher.ndim != 1 or not fisher.numel() or not (
                fisher.isfinite() & (fisher >= 0)
            ).all():
                raise ValueError("projected brakes require a finite nonnegative Fisher diagonal")
            low = fisher <= fisher.median()
            self.mask = low if name == "low" else ~low
        else:
            raise ValueError(f"unknown energy brake: {name!r}")

    def brake(self, p, joules):
        p = p.detach()
        squares = p.square()
        if self.mask is not None:
            squares = torch.where(self.mask, squares, 0)
        capacity = 0.5 * squares.sum()
        target = torch.as_tensor(joules, device=p.device, dtype=p.dtype).detach()
        if target.ndim or not (capacity.isfinite() & (target >= 0) & (target <= capacity)):
            raise ValueError("brake target must be a scalar between zero and finite capacity")
        denominator = torch.where(capacity > 0, capacity, 1)
        damped = p * (1 - target / denominator).sqrt()
        if self.mask is None:
            return damped
        return torch.where(self.mask, damped, p)


class Generic(DKD):
    def __init__(self, *, h, alpha, strength, fisher, anchor):
        weights = normalized_fisher(fisher)
        inv_mass = (1 + alpha * weights).reciprocal() if alpha else None
        super().__init__(h, inv_mass=inv_mass)
        if anchor is None or not anchor.isfinite().all() or (
            anchor.shape, anchor.device, anchor.dtype
        ) != (weights.shape, weights.device, weights.dtype):
            raise ValueError("GENERIC requires a finite anchor matching the Fisher diagonal")
        self.anchor = anchor.detach().clone()
        self.decay = strength * weights
        self.unit_gate_duration = h / 2
        self.unit_gate_blend = -torch.expm1(-self.unit_gate_duration * self.decay)

    def anchor_flow(self, q, duration):
        if not isfinite(duration) or duration < 0:
            raise ValueError("anchor duration must be finite and nonnegative")
        if duration == 0:
            return q.detach()
        blend = self.unit_gate_blend
        if duration != self.unit_gate_duration:
            blend = -torch.expm1(-duration * self.decay)
        return q.detach().lerp(self.anchor, blend)

    def step(self, q, state, grad, *, gate):
        if gate == 0:
            return super().step(q, state, grad, gate=0)
        duration = self.unit_gate_duration * gate
        anchored = self.anchor_flow(q, duration)
        q, state = super().step(anchored, state, grad, gate=0)
        return self.anchor_flow(q, duration), state
