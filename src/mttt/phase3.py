"""Factory for the frozen Phase 3 Transformer search adapter."""

from math import isfinite

from .p3_float import FloatEngine
from .p3_lattice import LatticeEngine


def make_engine(method: str, *, h: float, shifts: list[int]):
    if not isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if method in ("native", "sgd", "adam"):
        return FloatEngine(method, h=h)
    if method in ("int32", "int64"):
        return LatticeEngine(method, h=h, shifts=shifts)
    raise ValueError(f"unknown Phase 3 engine: {method!r}")
