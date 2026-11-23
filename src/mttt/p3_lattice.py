"""Checked fixed-point position Verlet for the Phase 3 phase arrays."""

import math

import torch


_BLOCK = 2**21
_FORMATS = {"int32": (torch.int32, 2**31 - 1), "int64": (torch.int64, 2**50 - 1)}


class LatticeEngine:
    def __init__(self, method, *, h, shifts):
        self._dtype, self._limit = _FORMATS[method]
        sq, sp = shifts
        self._q_encode, self._p_encode = math.ldexp(1.0, sq), math.ldexp(1.0, sp)
        self._q_decode, self._p_decode = math.ldexp(1.0, -sq), math.ldexp(1.0, -sp)
        self._drift = math.ldexp(h / 2, sq - sp)
        self._kick = math.ldexp(-h, sp)
        self.where = None
        self.diagnostics = None

    def _in_range(self, value):
        low, high = torch.aminmax(value)
        return (low >= -self._limit) & (high <= self._limit)

    def _invalid(self, value):
        return ~((value >= -self._limit) & (value <= self._limit))

    def _range_error(self, shear, start, attempted, increment=None):
        bad = self._invalid(attempted)
        if increment is not None:
            bad |= self._invalid(increment)
        index = bad.to(torch.uint8).argmax().item()
        delta = "" if increment is None else f" increment={increment[index].item()!r}"
        raise ArithmeticError(f"{self.where!r} {shear} index={start + index}{delta} "
                              f"attempted={attempted[index].item()!r} limit={self._limit}")

    @staticmethod
    def _matched(q, p, dtype):
        if q.dtype != dtype or p.dtype != dtype:
            raise TypeError(f"phase arrays must have dtype {dtype}")
        if q.ndim != 1 or not q.numel() or p.shape != q.shape or p.device != q.device:
            raise ValueError("phase arrays must be nonempty, flat, and match in shape and device")

    @staticmethod
    def _decode(value, scale):
        # At the specified shifts the power-of-two scaling commutes with binary32 rounding.
        return value.to(torch.float32, copy=True).mul_(scale)

    def _encode(self, source, scale, inv_scale, shear):
        out = torch.empty_like(source, dtype=self._dtype)
        for start in range(0, source.numel(), _BLOCK):
            values = source[start:start + _BLOCK]
            rounded = values.double().mul_(scale).round_()
            if not bool(self._in_range(rounded)):
                self._range_error(shear, start, rounded)
            block = out[start:start + _BLOCK]
            block.copy_(rounded)
            decoded = self._decode(block, inv_scale)
            mismatch = values.view(torch.int32) != decoded.view(torch.int32)
            if bool(mismatch.any()):
                index = mismatch.to(torch.uint8).argmax().item()
                raise ArithmeticError(f"{self.where!r} {shear} index={start + index} "
                                      f"input={values[index].item()!r} decoded={decoded[index].item()!r} "
                                      "is not losslessly encodable")
        return out

    def init(self, q, p):
        self._matched(q, p, torch.float32)
        return (self._encode(q.detach(), self._q_encode, self._q_decode, "init_q"),
                self._encode(p.detach(), self._p_encode, self._p_decode, "init_p"))

    def weights(self, state):
        return self._decode(state[0].detach(), self._q_decode)

    def _translate(self, base, source, factor, out, shear):
        diagnostics = self.diagnostics
        nonzero_count = lost_count = 0
        for start in range(0, base.numel(), _BLOCK):
            block = slice(start, start + _BLOCK)
            raw = source[block].double().mul_(factor)
            rounded = raw.round_() if diagnostics is None else raw.round()
            attempted = rounded + base[block]
            if not bool(self._in_range(rounded) & self._in_range(attempted)):
                self._range_error(shear, start, attempted, rounded)
            if diagnostics is not None:
                nonzero = raw != 0
                lost = nonzero & (rounded == 0)
                n, k = torch.stack((nonzero.count_nonzero(), lost.count_nonzero())).tolist()
                nonzero_count += n
                lost_count += k
                if k and "first_sub_lsb" not in diagnostics:
                    index = lost.to(torch.uint8).argmax().item()
                    diagnostics["first_sub_lsb"] = [self.where, shear, start + index, raw[index].item()]
            out[block].copy_(attempted)
            del raw, rounded, attempted

        if diagnostics is not None:
            diagnostics["nonzero_components"] = diagnostics.get("nonzero_components", 0) + nonzero_count
            diagnostics["lost_components"] = diagnostics.get("lost_components", 0) + lost_count
            if nonzero_count and lost_count == nonzero_count:
                diagnostics.setdefault("first_whole_shear_lost", [self.where, shear, nonzero_count])

    def step(self, state, grad):
        q, p = state
        self._matched(q, p, self._dtype)
        q_ok, p_ok = self._in_range(q), self._in_range(p)
        if not bool(q_ok & p_ok):
            if not bool(q_ok):
                self._range_error("input_q", 0, q)
            self._range_error("input_p", 0, p)

        next_q = torch.empty_like(q)
        self._translate(q, p, self._drift, next_q, "drift1")
        force = grad(self.weights((next_q, p))).detach()
        if force.dtype != torch.float32 or force.device != q.device or force.shape != q.shape:
            raise ValueError("lattice force must match phase shape and device and have dtype float32")
        next_p = torch.empty_like(p)
        self._translate(p, force, self._kick, next_p, "kick")
        del force
        # The midpoint is private workspace and becomes the final position in place.
        self._translate(next_q, next_p, self._drift, next_q, "drift2")
        return next_q, next_p
