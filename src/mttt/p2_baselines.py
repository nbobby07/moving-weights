"""Phase 2 gate interface for the Phase 1 functional baselines."""

from .baselines import AdamW, EWC, SGDM
from .p2_dkd import normalized_fisher


class Baseline:
    def __init__(self, name, *, h, strength, fisher, anchor):
        if name == "sgdm":
            self.optimizer = SGDM(h)
        elif name == "adamw":
            self.optimizer = AdamW(h)
        elif name == "ewc":
            if anchor is None:
                raise ValueError("EWC requires a prior-task anchor")
            self.optimizer = EWC(h, fisher=normalized_fisher(fisher), anchor=anchor, strength=strength)
        else:
            raise ValueError(f"unknown baseline: {name!r}")

    def init(self, q, p):
        return self.optimizer.init(q, p)

    def step(self, q, state, grad, *, gate):
        return self.optimizer.step(q, state, grad, gamma=0)
