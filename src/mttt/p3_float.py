"""Floating-point engines for reversible search and checkpoint baselines."""

from .baselines import AdamW, SGDM


class FloatEngine:
    def __init__(self, method, *, h):
        self._h = h
        self._half_h = h / 2
        self.where = None
        self.diagnostics = None
        if method == "native":
            self._baseline = None
        elif method == "sgd":
            self._baseline = SGDM(h)
        elif method == "adam":
            self._baseline = AdamW(h)
        else:
            raise ValueError(f"unknown floating-point engine: {method!r}")

    def init(self, q, p):
        if self._baseline is None:
            return q.detach(), p.detach()
        return q.detach(), *self._baseline.init(q, p)

    def weights(self, state):
        return state[0].detach()

    def step(self, state, grad):
        if self._baseline is not None:
            q, buffers = self._baseline.step(state[0], state[1:], grad, gamma=0.0)
            return q, *buffers

        q, p = state
        p = p.detach()
        midpoint = q.detach().add(p, alpha=self._half_h)
        p_next = p.add(grad(midpoint).detach(), alpha=-self._h)
        return midpoint.add(p_next, alpha=self._half_h), p_next
