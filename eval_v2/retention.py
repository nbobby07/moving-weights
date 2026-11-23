"""Four new tasks, identical batch order and 192 measured gradient calls per arm."""
import math
from statistics import mean, median

import torch

from .protocol import ForceCounter, accuracy, check, finite_phase, make, tick


GRID = {
    "sgdm": [{"h": h} for h in (0.0036, 0.0072, 0.0144, 0.0288)],
    "adamw": [{"h": h} for h in (0.0025, 0.005, 0.01, 0.02)],
    "ewc": [{"h": h, "strength": s} for h in (0.0072, 0.0144) for s in (0.05, 0.5)],
    "mass": [{"h": h, "alpha": a} for h in (0.06, 0.12) for a in (1.0, 8.0)],
    "energy": [{"h": h, "gamma": g} for h in (0.06, 0.12) for g in (0.03, 0.3)],
}
ARMS = ("iso", "low", "high")


def scores(matrix):
    bwt = matrix[-1][0] - matrix[0][0]
    live = mean(matrix[t][t] for t in range(1, 5))
    gain = mean(matrix[t][t] - matrix[t-1][t] for t in range(1, 5))
    return {"bwt": bwt, "live": live, "gain": gain}


def single(factory, method, cfg, prepared):
    q, fisher, tests, streams = prepared[:4]
    opt = make(factory, method, cfg, q, fisher)
    state = opt.init(q, torch.zeros_like(q))
    counter = ForceCounter()
    matrix = [[accuracy(q, b) for b in tests]]
    elapsed = []
    for stream in streams:
        start = tick(q.device)
        for batch in stream:
            q, state = counter.step(opt, q, state, batch, gate=float(method == "generic"))
        elapsed.append(tick(q.device) - start)
        finite_phase(q,state)
        matrix.append([accuracy(q, b) for b in tests])
    check(counter.calls == 192, "continual budget mismatch")
    return {**scores(matrix), "matrix": matrix, "calls": counter.calls, "seconds_by_task": elapsed}


def matched(factory, cfg, prepared):
    q0, fisher, tests, streams = prepared[:4]
    opts = [make(factory, n, cfg, q0, fisher) for n in ARMS]
    states = [o.init(q0, torch.zeros_like(q0)) for o in opts]
    qs = [q0.clone() for _ in opts]
    counters = [ForceCounter() for _ in opts]
    matrices = [[[accuracy(q0, b) for b in tests]] for _ in opts]
    low = fisher <= fisher.median()
    fraction = -math.expm1(-2 * cfg["h"] * cfg["gamma"])
    ledger, elapsed = [], []
    for stream in streams:
        start = tick(q0.device)
        for batch in stream:
            for i in range(3):
                qs[i], states[i] = counters[i].step(opts[i], qs[i], states[i], batch)
            ps = [s[0] for s in states]
            before = torch.stack([p.square().sum() / 2 for p in ps])
            cap_l = torch.where(low, ps[1].square(), 0).sum() / 2
            cap_h = torch.where(~low, ps[2].square(), 0).sum() / 2
            nominal = fraction * before[0]
            target = torch.minimum(nominal, torch.minimum(cap_l, cap_h))
            after = [o.brake(p, target) for o, p in zip(opts, ps)]
            removed = before - torch.stack([p.square().sum() / 2 for p in after])
            ledger.append(torch.cat((torch.stack((nominal, cap_l, cap_h, target)), removed)))
            states = [(p,) for p in after]
        elapsed.append(tick(q0.device)-start)
        for i, q in enumerate(qs):
            finite_phase(q,states[i])
            matrices[i].append([accuracy(q, b) for b in tests])
    log = torch.stack(ledger).cpu()
    residual = (log[:, 4:] - log[:, 3:4]).abs()
    check(torch.isfinite(log).all() and (residual <= 1e-12 + 5e-11 * log[:, 3:4]).all(),
          "realized kinetic removals differ")
    check(all(c.calls == 192 for c in counters), "energy arm budget mismatch")
    return {"arms": {n: {**scores(m), "matrix": m, "calls": c.calls}
                     for n, m, c in zip(ARMS, matrices, counters)},
            "ledger": log.tolist(), "joint_seconds_by_task": elapsed, "caps": int((log[:, 3] < log[:, 0]).sum()),
            "positive_steps": int((log[:, 3] > 0).sum()), "max_residual": residual.max().item()}


def ordinary_trials(factory, name, configs, prepared):
    trials = []
    for cfg in configs:
        rows = [single(factory, name, cfg, p) for p in prepared]
        trials.append({"cfg": cfg, "rows": rows})
    return trials


def selection(trials, energy=False):
    def rank(trial):
        rows = trial["rows"]
        groups = [[r["arms"][a] for r in rows] for a in ARMS] if energy else [rows]
        live = min(mean(r["live"] for r in group) for group in groups)
        bwt = min(mean(r["bwt"] for r in group) for group in groups)
        strict = all(r["caps"] == 0 for r in rows) if energy else True
        return strict, live >= 0.92, bwt if live >= 0.92 else live, live
    return max(trials, key=rank)["cfg"]


def validate(factory, prepared):
    trials = {n: ordinary_trials(factory, n, GRID[n], prepared) for n in ("sgdm","adamw","ewc","mass")}
    mass_cfg = selection(trials["mass"])
    trials["generic"] = ordinary_trials(factory, "generic", [
        {**mass_cfg, "strength": s} for s in (0.01, 0.05, 0.2, 1.0)], prepared)
    trials["energy"] = [{"cfg": cfg, "rows": [matched(factory, cfg, p) for p in prepared]} for cfg in GRID["energy"]]
    return trials


def choose(trials):
    return {n: selection(grid, energy=n == "energy") for n, grid in trials.items()}


def trial(factory, cfg, prepared, index):
    names = ["sgdm", "adamw", "ewc", "mass", "generic", "glide"]
    params = {**cfg, "glide": {"h": cfg["mass"]["h"]}}
    runs = {}
    for name in names[index % 6:] + names[:index % 6]:
        runs[name] = single(factory, name, params[name], prepared)
    pair = matched(factory, cfg["energy"], prepared)
    runs.update(pair.pop("arms"))
    return {"runs": runs, "energy": pair}


def paired(rows, method, other):
    # Test the signed change in held-out correct counts (300 per seed).
    d = [round(300 * (r["runs"][method]["bwt"] - r["runs"][other]["bwt"])) for r in rows]
    n = sum(x != 6 for x in d)
    wins = sum(x > 6 for x in d)
    return {"differences_correct": d, "mean_pp": mean(d) / 3, "wins": wins, "non_ties": n,
            "p": sum(math.comb(n, k) for k in range(wins, n+1)) / 2**n}


def verdict(rows):
    baselines = ("sgdm", "adamw", "ewc")
    lives = {n: mean(r["runs"][n]["live"] for r in rows) for n in rows[0]["runs"]}
    result = {}
    for n, control in (("mass","glide"), ("generic","mass")):
        tests = {o: paired(rows,n,o) for o in (*baselines, control)}
        plastic = lives[n] >= 0.90 and lives[n] >= max(lives[o] for o in baselines) - 0.02
        result[n] = {"live": lives[n], "plasticity_pass": plastic, "contrasts": tests,
                     "pass": plastic and all(t["p"] <= 0.05/9 for t in tests.values())}
    high = paired(rows,"high","low")
    no_cap = all(r["energy"]["caps"] == 0 for r in rows)
    active = all(r["energy"]["positive_steps"] == 192 for r in rows)
    result["energy"] = {"high_vs_low": high, "fixed_gamma_feasible_all_steps": no_cap,
                        "active_every_step": active, "capped_steps": sum(r["energy"]["caps"] for r in rows),
                        "pass": no_cap and active and high["p"] <= 0.05/9 and lives["high"] >= lives["low"]-0.02,
                        "descriptive_vs_tuned": {a:{b:paired(rows,a,b)["mean_pp"] for b in baselines} for a in ARMS}}
    result["live_means"] = lives
    return result


def timing(factory, cfg, prep):
    # Different batches, same 48-step tape. Warm each method before each paired block.
    q, f, _, streams = prep[:4]
    names = ("sgdm", "glide", "mass", "generic", "adamw")
    seconds = {n: [] for n in names}
    for repeat in range(5):
        for name in names[repeat:] + names[:repeat]:
            params = cfg[name] if name != "glide" else {"h": cfg["mass"]["h"]}
            opt = make(factory, name, params, q, f)
            x, state = q.clone(), opt.init(q, torch.zeros_like(q))
            counter = ForceCounter()
            for batch in streams[0][:8]:
                x, state = counter.step(opt,x,state,batch, float(name=="generic"))
            start = tick(q.device)
            for batch in streams[0]:
                x, state = counter.step(opt,x,state,batch, float(name=="generic"))
            seconds[name].append(tick(q.device)-start)
            check(counter.calls == 56, "timing budget mismatch")
    ratios = {n: median(a/b for a,b in zip(times, seconds["sgdm"])) for n,times in seconds.items()}
    return {"calls_per_timed_block": 48, "seconds": seconds, "paired_median_ratio_to_sgdm": ratios,
            "at_most_1_05x": {n:r<=1.05 for n,r in ratios.items()}}
