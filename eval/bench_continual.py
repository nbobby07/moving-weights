"""Five shared-head tasks, blinded window decisions and paired retention contrasts."""
import math
from statistics import mean, pstdev

import torch
import torch.nn.functional as F

from .common import GAMMA, H, METHODS, PHASE, SEEDS, generator, l2, run, tick, unwind


D, WIDTH, CLASSES = 7, 16, 3
BATCH, STEPS, WINDOW, FOLLOWUP = 32, 48, 24, 32
SIZES = (D * WIDTH, WIDTH, WIDTH * CLASSES, CLASSES)


def logits(q, x):
    w, b, v, c = q.split(SIZES)
    return (x @ w.reshape(D, WIDTH) + b).tanh() @ v.reshape(WIDTH, CLASSES) + c


def loss(q, x, y):
    return F.cross_entropy(logits(q, x), y)


def gradient(batch):
    def grad(q):
        with torch.enable_grad():
            leaf = q.detach().requires_grad_()
            return torch.autograd.grad(loss(leaf, *batch), leaf)[0]
    return grad


def accuracy(q, batch):
    x, y = batch
    return (logits(q, x).argmax(-1) == y).double().mean().item()


def data(seed, task, count, device):
    rng = generator(seed)
    y = (torch.arange(count) % CLASSES)[torch.randperm(count, generator=rng)]
    a = y.double() * (2 * math.pi / CLASSES) + math.radians(25 * task)
    x = 0.48 * torch.randn(count, D, generator=rng, dtype=torch.float64)
    x[:, 0] += 1.45 * a.cos()
    x[:, 1] += 1.45 * a.sin()
    x[:, 2 + task] += 0.6
    return x.to(device), y.to(device)


def batches(dataset, steps, seed):
    x, y = dataset
    ids = torch.randint(len(y), (steps, BATCH), generator=generator(seed)).to(y.device)
    return list(zip(x[ids].unbind(), y[ids].unbind()))


def prepare(seed, device):
    # Fixed disjoint RNG namespaces for training, tests, Fisher, calibration and streams.
    base = seed * 100
    training = [data(base + t, t, 384, device) for t in range(5)]
    tests = [data(base + 10 + t, t, 300, device) for t in range(5)]
    q = (0.2 * torch.randn(sum(SIZES), generator=generator(base + 20), dtype=torch.float64)).to(device)
    for batch in batches(training[0], 120, base + 21):
        q = q - 0.12 * gradient(batch)(q)
    x, y = data(base + 22, 0, 96, device)
    def individual(v, xi, yi):
        return loss(v, xi.unsqueeze(0), yi.unsqueeze(0))
    scores = torch.func.vmap(torch.func.grad(individual), in_dims=(None, 0, 0))(q, x, y)
    fisher = scores.square().mean(0)
    calibration = batches(data(base + 23, 0, 384, device), 32, base + 24)
    cal_losses = [loss(q, *b).item() for b in calibration]
    threshold = mean(cal_losses) + 4 * pstdev(cal_losses)
    guard = batches(data(base + 25, 0, 384, device), WINDOW, base + 26)
    future = batches(training[0], FOLLOWUP, base + 27)
    streams = [batches(training[t], STEPS, base + 30 + t) for t in range(1, 5)]
    return q, fisher, tests, streams, cal_losses, threshold, guard, future


def episode(factory, name, q0, fisher, test, stream, future, threshold):
    opt = factory(name, h=H[name], fisher=fisher)
    q = q0.clone()
    state = opt.init(q, torch.zeros_like(q))
    first_alarm = None
    losses, entropies = [], []
    for i, batch in enumerate(stream):
        x, y = batch
        z = logits(q, x)
        value = F.cross_entropy(z, y).item()
        logp = z.log_softmax(-1)
        entropies.append(-(logp.exp() * logp).sum(-1).mean().item())
        losses.append(value)
        if first_alarm is None and value > threshold:
            first_alarm = i
        q, state = opt.step(q, state, gradient(batch), gamma=0.0)
    raw = accuracy(q, test)
    raw_displacement = l2(q, q0)
    rolled = first_alarm is not None and name in PHASE
    q_error = p_error = rewind_ms = None
    if rolled:
        # Rebuild the optimizer. Only endpoint phase state crosses the reversal boundary.
        opt = factory(name, h=H[name], fisher=fisher)
        start = tick(q.device)
        q, state = unwind(opt, q, state, len(stream), lambda i: gradient(stream[i]))
        rewind_ms = (tick(q.device) - start) * 1000
        q_error, p_error = l2(q, q0), l2(state[0], torch.zeros_like(q))
    post = accuracy(q, test)
    for batch in future:
        q, state = opt.step(q, state, gradient(batch), gamma=0.0)
    return {"first_alarm_zero_based": first_alarm, "rewound": rolled, "loss_trace": losses,
            "entropy_trace": entropies, "pre_acc": accuracy(q0, test), "uncorrected_acc": raw,
            "uncorrected_displacement_l2": raw_displacement, "post_policy_acc": post,
            "followup_acc": accuracy(q, test), "rollback_q_l2": q_error,
            "rollback_p_l2": p_error, "rewind_ms": rewind_ms}


def sequential(factory, name, q0, fisher, tests, streams):
    opt = factory(name, h=H[name], fisher=fisher)
    q = q0.clone()
    state = opt.init(q, torch.zeros_like(q))
    gamma = GAMMA if name in ("strang", "fisher") else 0.0
    matrix = [[accuracy(q, b) for b in tests]]
    curves, seconds = [], []
    for task, stream in enumerate(streams, 1):
        curve = [matrix[-1][task]]
        elapsed = 0.0
        for offset in range(0, STEPS, 12):
            start = tick(q.device)
            for batch in stream[offset:offset + 12]:
                q, state = opt.step(q, state, gradient(batch), gamma=gamma)
            elapsed += tick(q.device) - start
            curve.append(accuracy(q, tests[task]))
        seconds.append(elapsed)
        curves.append(curve)
        matrix.append([accuracy(q, b) for b in tests])
    return {"test_accuracy_matrix": matrix, "arrival_to_live_12_step_curves": curves,
            "task1_bwt": matrix[-1][0] - matrix[0][0],
            "live_acc": mean(curve[-1] for curve in curves),
            "arrival_to_live_gain": mean(curve[-1] - curve[0] for curve in curves),
            "optimizer_wall_seconds_by_task": seconds, "ms_per_step": 1000 * sum(seconds) / (4 * STEPS)}


def sign_test(deltas, margin):
    # Test sets have 300 cases. Classify ties on the integer count scale.
    counts = [round(300 * d) for d in deltas]
    margin_count = round(300 * margin)
    n = sum(d != margin_count for d in counts)
    wins = sum(d > margin_count for d in counts)
    return {"paired_deltas": deltas, "mean": mean(deltas), "margin": margin,
            "non_ties": n, "wins": wins,
            "p_one_sided": sum(math.comb(n, j) for j in range(wins, n + 1)) / 2**n}


def summarize(rows):
    def contrast(field, other):
        return [r["methods"]["fisher"]["sequential"][field] -
                r["methods"][other]["sequential"][field] for r in rows]
    retention = {n: sign_test(contrast("task1_bwt", n), 0.02) for n in ("sgdm", "adamw", "strang")}
    plasticity = {n: mean(contrast("live_acc", n)) for n in ("sgdm", "adamw", "strang")}
    glide = [r["methods"]["glide"] for r in rows]
    adam = [r["methods"]["adamw"] for r in rows]
    detected = sum(g["poison"]["rewound"] for g in glide)
    false_alarms = sum(g["clean"]["rewound"] for g in glide)
    def damage(items, metric):
        return mean(g["poison"]["pre_acc"] - g["poison"][metric] for g in items)
    matched_residual = mean(g["clean"]["followup_acc"] - g["poison"]["followup_acc"] for g in adam)
    b = {
        "all_poisons_detected": detected == len(rows), "at_most_one_clean_alarm": false_alarms <= 1,
        "all_glide_roundtrips": all(g["poison"]["rewound"] and
                                    max(g["poison"]["rollback_q_l2"], g["poison"]["rollback_p_l2"]) < 1e-11
                                    for g in glide),
        "all_glide_immediate_recoveries": all(abs(g["poison"]["post_policy_acc"] -
                                                  g["poison"]["pre_acc"]) <= 0.01 for g in glide),
        "glide_poison_damage": damage(glide, "uncorrected_acc") >= 0.05,
        "adamw_immediate_damage": damage(adam, "post_policy_acc") >= 0.05,
        "adamw_matched_residual_after_32": matched_residual >= 0.02,
    }
    live = mean(r["methods"]["fisher"]["sequential"]["live_acc"] for r in rows)
    gain = mean(r["methods"]["fisher"]["sequential"]["arrival_to_live_gain"] for r in rows)
    c = {"retention_vs_both": all(retention[n]["p_one_sided"] <= 0.025 for n in ("sgdm", "adamw")),
         "live_at_least_80pct": live >= 0.80, "mean_gain_at_least_2pp": gain >= 0.02,
         "live_within_3pp_of_both": min(plasticity[n] for n in ("sgdm", "adamw")) >= -0.03}
    return {"B_components": b, "B_finite_horizon_pass": all(b.values()), "poison_detections": detected,
            "clean_alarms": false_alarms, "adamw_matched_residual": matched_residual,
            "glide_uncorrected_damage": damage(glide, "uncorrected_acc"),
            "adamw_immediate_damage": damage(adam, "post_policy_acc"),
            "C_components": c, "C_pass": all(c.values()), "fisher_live_acc": live,
            "fisher_arrival_gain": gain, "paired_retention": retention, "paired_live_means": plasticity}


def neural_roundtrip(factory, q0, fisher, stream):
    opt = factory("glide", h=0.04, fisher=fisher)
    p0 = 0.02 * torch.cos(torch.arange(len(q0), device=q0.device, dtype=q0.dtype))
    q, state = q0.clone(), opt.init(q0, p0)
    def grad(i):
        return gradient(stream[i % len(stream)])
    for i in range(500):
        q, state = opt.step(q, state, grad(i), gamma=0.0)
    fresh = factory("glide", h=0.04, fisher=fisher)
    q, (p,) = unwind(fresh, q, state, 500, grad)
    return {"seed": SEEDS[0], "h": 0.04, "steps": 500, "q_l2": l2(q0, q), "p_l2": l2(p0, p)}


def evaluate(factory, device):
    rows = []
    for j, seed in enumerate(SEEDS):
        q, fisher, tests, streams, cal, threshold, clean, future = prepare(seed, device)
        poison = [(x, (y + 1) % CLASSES) for x, y in clean]
        if j == 0:
            neural = neural_roundtrip(factory, q, fisher, streams[0])
        methods = {}
        # Rotate benchmark order across seeds to reduce wall-clock ordering effects.
        for name in METHODS[j % len(METHODS):] + METHODS[:j % len(METHODS)]:
            methods[name] = {
                "clean": episode(factory, name, q, fisher, tests[0], clean, future, threshold),
                "poison": episode(factory, name, q, fisher, tests[0], poison, future, threshold),
                "sequential": sequential(factory, name, q, fisher, tests, streams)}
        base_time = methods["sgdm"]["sequential"]["ms_per_step"]
        for item in methods.values():
            item["sequential"]["time_ratio_to_sgdm"] = item["sequential"]["ms_per_step"] / base_time
        rows.append({"seed": seed, "calibration_losses": cal, "threshold": threshold,
                     "fisher_diag": fisher.tolist(), "methods": methods})
    return {"rows": rows, "neural_rollback_f64": neural,
            "checks_passed": max(neural["q_l2"], neural["p_l2"]) < 1e-11,
            "verdict": summarize(rows)}


if __name__ == "__main__":
    run("continual", evaluate)
