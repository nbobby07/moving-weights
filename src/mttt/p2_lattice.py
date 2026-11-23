"""Checked int64 phase shears with a float32 midpoint force."""

import torch
from torch import Tensor

from .phase import Grad, PhaseState


_SCALE = 2**44
_INCREMENT_LIMIT = 2**51
_STATE_LIMIT = 2**52


def _check_range(value: Tensor, bound: int, kind: str) -> None:
    # Compare directly because abs(INT64_MIN) overflows.
    if not bool(((value > -bound) & (value < bound)).all()):
        raise ValueError(f"lattice {kind} is outside (-{bound}, {bound})")


def _increment(value: Tensor) -> Tensor:
    rounded = value.round()
    _check_range(rounded, _INCREMENT_LIMIT, "increment")
    return rounded.to(torch.int64)


def _check_phase(q: Tensor, p: Tensor) -> None:
    if q.dtype != torch.int64 or p.dtype != torch.int64:
        raise TypeError("lattice phase must use int64")
    if q.ndim != 1 or q.shape != p.shape or q.device != p.device:
        raise ValueError("lattice q and p must be flat and match in shape and device")
    _check_range(q, _STATE_LIMIT, "position")
    _check_range(p, _STATE_LIMIT, "momentum")


class Lattice:
    def __init__(self, h: float):
        self._h = h

    def init(self, q: Tensor, p: Tensor) -> PhaseState:
        _check_phase(q, p)
        return (p.detach().clone(),)

    def step(
        self, q: Tensor, state: PhaseState, grad: Grad, *, gate: float
    ) -> tuple[Tensor, PhaseState]:
        (p,) = state
        _check_phase(q, p)
        half_h = self._h / 2
        midpoint = q + _increment(half_h * p.double())
        _check_range(midpoint, _STATE_LIMIT, "midpoint")

        force = grad((midpoint.double() / _SCALE).float())
        if force.dtype != torch.float32:
            raise TypeError("lattice force must use float32")
        if force.shape != q.shape or force.device != q.device:
            raise ValueError("lattice force must match phase shape and device")
        next_p = p - _increment(force.detach().double() * (self._h * _SCALE))
        _check_range(next_p, _STATE_LIMIT, "momentum")
        next_q = midpoint + _increment(half_h * next_p.double())
        _check_range(next_q, _STATE_LIMIT, "position")
        return next_q, (next_p,)
