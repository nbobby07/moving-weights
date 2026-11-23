"""Coupled double well. Absolute rollback and physical/shadow energy refinement."""
import math

import torch

from .common import l2, run, unwind


def initial(device, dtype):
    q = torch.tensor([0.85, -0.72], dtype=dtype, device=device)
    return q, q.new_tensor([0.22, 0.31])


def force(q):
    return q * (q.square() - 1) + 0.12 * (q - q.flip((0,)))


def gradient(i):
    # A non-palindromic batch stream. The two kicks must use the same tilt.
    tilt = (i * 7 % 19 - 9) / 180.0
    return lambda q: force(q) + tilt


def energy(q, p, h):
    diff = q[0] - q[1]
    u = 0.25 * (q.square() - 1).square().sum() + 0.06 * diff.square()
    physical = u + 0.5 * p.square().sum()
    php = ((3 * q.square() - 1) * p.square()).sum() + 0.12 * (p[0] - p[1]).square()
    shadow = physical + h**2 * (php / 12 - force(q).square().sum() / 24)
    return torch.stack((physical, shadow))


def rollback(factory, device, dtype):
    q0, p0 = initial(device, dtype)
    fisher = torch.ones_like(q0)
    opt = factory("glide", h=0.025, fisher=fisher)
    q, state = q0.clone(), opt.init(q0.clone(), p0.clone())
    rows = []
    for i in range(500):
        q, state = opt.step(q, state, gradient(i), gamma=0.0)
        if i + 1 in (1, 10, 50, 100, 250, 500):
            fresh = factory("glide", h=0.025, fisher=fisher)
            back, (p,) = unwind(fresh, q.clone(), (state[0].clone(),), i + 1, gradient)
            rows.append({"steps": i + 1, "q_l2": l2(back, q0), "p_l2": l2(p, p0)})
    return rows


def energy_test(factory, device, h, steps):
    q, p = initial(device, torch.float64)
    opt = factory("glide", h=h, fisher=torch.ones_like(q))
    state = opt.init(q, p)
    samples = [energy(q, p, h)]
    for _ in range(steps):
        q, state = opt.step(q, state, force, gamma=0.0)
        samples.append(energy(q, state[0], h))
    trace = torch.stack(samples).cpu()
    diff = torch.diff(trace[:, 0])
    return {"h": h, "steps": steps, "time": h * steps,
            "span": (trace.amax(0) - trace.amin(0)).tolist(),
            "physical_turns": int(((diff[1:] * diff[:-1]) < 0).sum()),
            "physical_and_shadow_trace": trace.tolist()}


def evaluate(factory, device):
    f64 = rollback(factory, device, torch.float64)
    f32 = rollback(factory, device, torch.float32)
    energy_rows = [energy_test(factory, device, 0.08 / 2**k, 100 * 2**k) for k in range(3)]
    orders = []
    for a, b in zip(energy_rows, energy_rows[1:]):
        orders.append([math.log2(x / y) if x > 0 and y > 0 else None
                       for x, y in zip(a["span"], b["span"])])
    checks = {
        "f64_q_and_p": all(max(row["q_l2"], row["p_l2"]) < 1e-11 for row in f64),
        "physical_order": all(o[0] is not None and 1.8 <= o[0] <= 2.2 for o in orders),
        "shadow_order": all(o[1] is not None and 3.5 <= o[1] <= 4.5 for o in orders),
        "oscillatory": all(row["physical_turns"] >= 2 for row in energy_rows),
    }
    return {"rollback_f64": f64, "rollback_f32": f32, "energy": energy_rows,
            "orders_physical_shadow": orders, "checks": checks, "checks_passed": all(checks.values())}


if __name__ == "__main__":
    run("analytic", evaluate)
