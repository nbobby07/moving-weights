"""Factory for the frozen Phase 1 evaluation adapter."""

from math import isfinite

from torch import Tensor

from .baselines import AdamW, SGDM
from .phase import PhaseOptimizer


def make_optimizer(name: str, *, h: float, fisher: Tensor):
    if not isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if name in ("glide", "strang", "fisher"):
        return PhaseOptimizer(name, h=h, fisher=fisher)
    if name == "sgdm":
        return SGDM(h)
    if name == "adamw":
        return AdamW(h)
    raise ValueError(f"unknown Phase 1 optimizer: {name!r}")
