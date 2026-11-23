"""Auditor oracle. An independent adapter uses the same public methods."""
import math

import torch

from eval.reference import make_optimizer as old_baseline
from .protocol import SCALE, check, unpack


def integer_increment(value):
    # Leave room for integer addition, and keep the next drift conversion exact in binary64.
    check(bool((value.abs() < 2**51).all()), "invalid lattice increment or overflow")
    return value.round().long()


class Adapter:
    def __init__(self, name, h, alpha, strength, fisher, anchor):
        self.name, self.h, self.strength = name, h, strength
        self.anchor = anchor
        self.weight = None if fisher is None else fisher / fisher.mean().clamp_min(1e-12)
        self.inverse_mass = 1 / (1 + alpha * self.weight) if name in ("mass", "generic") else 1.0
        self.mask = None
        if name in ("low", "high"):
            low = fisher <= fisher.median()
            self.mask = low if name == "low" else ~low
        self.prior = None
        if name in ("sgdm", "adamw", "ewc"):
            # Only the baseline recurrence is shared with the independently pinned Phase 1 oracle.
            self.prior = old_baseline("sgdm" if name == "ewc" else name, h=h, fisher=torch.zeros(1))

    def init(self, q, p):
        return self.prior.init(q, p) if self.prior is not None else (p.detach().clone(),)

    def anchor_flow(self, q, duration):
        return self.anchor + torch.exp(-duration * self.strength * self.weight) * (q - self.anchor)

    def brake(self, p, joules):
        local = p if self.mask is None else torch.where(self.mask, p, 0.0)
        capacity = local.square().sum() / 2
        # The coordinator supplies a feasible target. The audit measures the realized removal.
        ratio = joules / torch.where(capacity > 0, capacity, torch.ones_like(capacity))
        scale = torch.sqrt((1 - ratio).clamp_min(0))
        return p + (scale - 1) * local

    def step(self, q, state, grad, *, gate):
        if self.prior is not None:
            if self.name == "ewc":
                def penalized(x):
                    return grad(x) + self.strength * self.weight * (x - self.anchor)
                return self.prior.step(q, state, penalized, gamma=0.0)
            return self.prior.step(q, state, grad, gamma=0.0)
        p, = state
        h = self.h
        if self.name == "lattice":
            mid = q + integer_increment((0.5 * h) * p.double())
            new_p = p - integer_increment((SCALE * h) * grad(unpack(mid)).double())
            end = mid + integer_increment((0.5 * h) * new_p.double())
            check(bool((torch.stack((mid, new_p, end)).abs() < 2**52).all()), "lattice phase range exceeded")
            return end, (new_p,)
        duration = 0.5 * h * gate
        if self.name == "generic" and gate:
            q = self.anchor_flow(q, duration)
        mid = q + (0.5 * h) * (self.inverse_mass * p)
        new_p = p - h * grad(mid)
        end = mid + (0.5 * h) * (self.inverse_mass * new_p)
        if self.name == "generic" and gate:
            end = self.anchor_flow(end, duration)
        return end.detach(), (new_p.detach(),)


def make_optimizer(name, *, h, alpha, strength, fisher, anchor):
    if name not in ("glide", "mass", "generic", "iso", "low", "high", "sgdm", "adamw", "ewc", "lattice"):
        raise ValueError(name)
    if not all(math.isfinite(x) and x >= 0 for x in (h, alpha, strength)) or h == 0:
        raise ValueError("invalid method coefficients")
    return Adapter(name, h, alpha, strength, fisher, anchor)
