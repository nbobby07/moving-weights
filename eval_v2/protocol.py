"""Locked shared inputs, accounting and exact-byte provenance."""
from array import array
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch

from eval.bench_continual import BATCH, accuracy, batches, data, gradient, prepare
from eval.common import generator, l2, tick


ROOT = Path(__file__).resolve().parent
PIN1 = "a2c2f60d9fe7508f0f5c62701016d488b365d8d68710e72bac41cec7ebdcc89d"
VAL = (10301, 10303, 10313)
TEST = (20903, 20921, 20929, 20939, 20947, 20959, 20963, 20981, 21001, 21011, 21013, 21017)
HORIZONS = (16, 32, 64, 128, 256, 500)
SCALE = 2**44


def tree_hash(root, python_only=False):
    digest = sha256()
    for p in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if p.is_symlink():
            raise ValueError(f"unexpected source symlink {p}")
        if p.is_file() and (not python_only or p.suffix == ".py"):
            digest.update(p.relative_to(root).as_posix().encode() + b"\0")
            digest.update(sha256(p.read_bytes()).digest())
    return digest.hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def finite_phase(q, state):
    values = [q, *(v for v in state if isinstance(v,torch.Tensor))]
    check(bool(torch.stack([v.isfinite().all() for v in values]).all()),"nonfinite endpoint")


def unpack(x):
    return (x.double() / SCALE).float()


def pack(x):
    check(x.dtype == torch.float32, "lattice input must be binary32")
    raw = x.double() * SCALE
    check(bool((raw.abs() < 2**51).all()), "initial lattice range exceeded")
    z = raw.round().long()
    check(torch.equal(unpack(z).view(torch.int32), x.contiguous().view(torch.int32)), "input not exactly encodable")
    return z


def make(factory, name, cfg, q, fisher=None):
    anchor = q if name in ("generic", "ewc") else None
    return factory(name, h=cfg["h"], alpha=cfg.get("alpha", 0.0),
                   strength=cfg.get("strength", 0.0), fisher=fisher, anchor=anchor)


class ForceCounter:
    def __init__(self):
        self.calls = 0

    def step(self, opt, q, state, batch, gate=0.0):
        force = gradient(batch)
        before = self.calls
        def measured(x):
            self.calls += 1
            return force(x)
        q, state = opt.step(q, state, measured, gate=gate)
        check(self.calls == before + 1, "expected one fresh minibatch gradient")
        return q, state


@dataclass
class Tape:
    # Immutable external event bank. A policy retains only IDs it has received.
    x: torch.Tensor
    y: torch.Tensor
    arrivals: list
    receipts: list

    def fetch(self, event_id):
        return self.x[event_id], self.y[event_id]

    def payload_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.x, self.y))


def episode_tape(seed, regime, dirty, device):
    n = 48 if regime == "a" else 32
    origin = seed * 100 + (50 if regime == "a" else 55)
    x, truth = data(origin, 1, n * BATCH, torch.device("cpu"))
    x, truth = x.reshape(n, BATCH, -1), truth.reshape(n, BATCH)
    if regime == "b":
        # No adaptation labels from the task. The teacher is the geometric prototype rule.
        angles = torch.arange(3, dtype=x.dtype) * (2 * torch.pi / 3) + 25 * torch.pi / 180
        prototypes = torch.stack((angles.cos(), angles.sin()))
        observed = (x[:, :, :2] @ prototypes).argmax(-1)
    else:
        observed = truth.clone()
    rng = generator(origin + 1)
    receipts = []
    for i in range(n):
        shuffle = torch.randperm(BATCH, generator=rng)
        ramp = min(24, i - 15) if regime == "a" and i >= 16 else 0
        if regime == "b":
            ramp = min(32, 2 * (i + 1))
        indices = shuffle[:ramp if dirty else 0]
        observed[i, indices] = (observed[i, indices] + 1) % 3
        if regime == "a":
            audited = torch.randperm(BATCH, generator=rng)[:8]
            verified = truth[i, audited].clone()
            if i % 4 == 0:
                verified[0] = (verified[0] + 1) % 3
            receipts.append(int((observed[i, audited] != verified).sum()))
    # Return block aggregates only. No within-block labels, masks or onset flag reach a policy.
    summaries = [sum(receipts[start:start+8]) / 64 for start in range(0, len(receipts), 8)]
    ids = torch.randperm(n, generator=generator(origin + 2))
    order = ids.argsort()
    return Tape(x[order].to(device), observed[order].to(device), ids.tolist(), summaries)


def id_buffer():
    ids = array("q")
    check(ids.itemsize == 8, "expected 64-bit event IDs")
    return ids
