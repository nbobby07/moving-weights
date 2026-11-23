"""Factory for the frozen Phase 2 evaluation adapter."""

from math import isfinite

from .p2_baselines import Baseline
from .p2_dissipation import EnergyBrake, Generic
from .p2_dkd import DKD, normalized_fisher
from .p2_lattice import Lattice


def make_optimizer(name, *, h, alpha=0.0, strength=0.0, fisher=None, anchor=None):
    if not isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not all(isfinite(v) and v >= 0 for v in (alpha, strength)):
        raise ValueError("alpha and strength must be finite and nonnegative")
    if name == "glide":
        return DKD(h)
    if name == "mass":
        weights = normalized_fisher(fisher)
        inv_mass = (1 + alpha * weights).reciprocal() if alpha else None
        return DKD(h, inv_mass=inv_mass)
    if name == "generic":
        return Generic(h=h, alpha=alpha, strength=strength, fisher=fisher, anchor=anchor)
    if name in ("iso", "low", "high"):
        return EnergyBrake(name, h=h, fisher=fisher)
    if name in ("sgdm", "adamw", "ewc"):
        return Baseline(name, h=h, strength=strength, fisher=fisher, anchor=anchor)
    if name == "lattice":
        return Lattice(h)
    raise ValueError(f"unknown Phase 2 optimizer: {name!r}")
