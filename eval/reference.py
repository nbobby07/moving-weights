"""Transparent functional audit oracle. Sol supplies an independent module."""
import math

import torch

from .common import METHODS, PHASE


class Reference:
    def __init__(self, name, h, fisher):
        if name not in METHODS or h <= 0 or fisher.ndim != 1:
            raise ValueError("invalid optimizer configuration")
        self.name, self.h = name, h
        if name == "fisher":
            if not torch.isfinite(fisher).all() or (fisher < 0).any():
                raise ValueError("expected finite nonnegative Fisher diagonal")
            # Ties at the lower median belong to the low-curvature subspace.
            self.diag = (fisher <= fisher.median()).to(fisher)
        else:
            self.diag = torch.ones_like(fisher)

    def brake_diag(self, q):
        return self.diag

    def init(self, q, p):
        if self.name in PHASE:
            return (p.clone(),)
        if self.name == "sgdm":
            return (torch.zeros_like(q),)
        return (torch.zeros_like(q), torch.zeros_like(q), 0)

    def step(self, q, state, grad, *, gamma):
        h = self.h
        g = grad(q)
        if self.name == "adamw":
            m, v, n = state
            n += 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g.square()
            update = (m / (1 - 0.9**n)) / ((v / (1 - 0.999**n)).sqrt() + 1e-8)
            return (1 - h * 0.01) * q - h * update, (m, v, n)
        if self.name == "sgdm":
            v = 0.9 * state[0] + g
            return q - h * v, (v,)
        p, = state
        if gamma:
            if gamma < 0 or self.name == "glide":
                raise ValueError("invalid brake gate")
            d = (torch.exp(-0.5 * h * gamma * self.diag) if self.name == "fisher"
                 else math.exp(-0.5 * h * gamma))
            p = d * p
        half = p - (0.5 * h) * g
        q = q + h * half
        p = half - (0.5 * h) * grad(q)
        return q, (d * p if gamma else p,)


def make_optimizer(name, *, h, fisher):
    return Reference(name, h, fisher)
