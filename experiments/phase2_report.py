"""Tabulate and plot the locked Phase 2 artifacts without rerunning evaluation."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = {
    "rewind": "DKD rewind", "glide_skip": "DKD skip", "glide_continue": "DKD continue",
    "adam_skip": "AdamW skip", "adam_continue": "AdamW continue", "always_skip": "Always skip",
}
METHODS = {
    "sgdm": "Momentum SGD", "adamw": "AdamW", "ewc": "EWC", "glide": "Unit mass DKD",
    "mass": "Fisher mass", "generic": "GENERIC", "iso": "Energy iso", "low": "Energy low",
    "high": "Energy high",
}
ENERGY = ("iso", "low", "high")
COLORS = ("#0072b2", "#d55e00", "#009e73", "#7057a3", "#b34a77", "#525252",
          "#007d86", "#995b00", "#3c4d9a")


def stats(values):
    samples = [value for value in values if value is not None]
    return {"n": len(samples), "mean": mean(samples) if samples else None,
            "sample_sd": stdev(samples) if len(samples) > 1 else None}


def csv_file(directory, name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(reports):
    for backend, report in reports.items():
        for seed, recovery, continual in zip(report["seeds"], report["B"], report["C"], strict=True):
            yield backend, seed, recovery, continual


def b2_tables(directory, reports):
    summary, paired = [], []
    for backend, report in reports.items():
        for regime, origin in (("a", 16), ("b", 0)):
            for condition in ("clean", "dirty"):
                arms = [row[regime][condition]["policies"] for row in report["B"]]
                group = {"backend": backend, "regime": f"B2{regime}", "condition": condition}
                for name in POLICIES:
                    values = []
                    for policies in arms:
                        row = policies[name]
                        alarm = row["alarm_after_commits"]
                        cut = row["estimated_cut"]
                        damage = None if alarm is None else row["target_acc"] - row["raw_acc_at_alarm"]
                        values.append({
                            **{key: value for key, value in row.items() if key not in ("calls", "memory_bytes")},
                            **{f"calls_{key}": value for key, value in row["calls"].items()},
                            **{f"bytes_{key}": value for key, value in row["memory_bytes"].items()},
                            "alarm": alarm is not None, "true_origin_step": origin,
                            "cut_minus_true_origin": None if cut is None else cut - origin,
                            "commits_since_origin_at_alarm": None if alarm is None else alarm - origin,
                            "accuracy_damage_at_alarm": damage,
                            "alarm_with_damage_at_least_2pp": damage is not None and round(300 * damage) >= 6,
                        })
                    for metric in values[0]:
                        series = [row[metric] for row in values]
                        count_true = sum(series) if isinstance(series[0], bool) else None
                        summary.append({**group, "policy": name, "metric": metric, **stats(series),
                                        "count_true": count_true})
                contrasts = (("always_skip", "live_acc"),) if condition == "clean" else (
                    ("adam_skip", "endpoint_acc"), ("glide_skip", "endpoint_acc"),
                    ("adam_skip", "equal_budget_acc"),
                )
                for control, metric in contrasts:
                    delta = [100 * (p["rewind"][metric] - p[control][metric]) for p in arms]
                    paired.append({**group, "policy": "rewind", "control": control, "metric_pp": metric,
                                   **stats(delta), "seed_deltas_pp": json.dumps(dict(zip(report["seeds"], delta)))})
    csv_file(directory, "b2_summary", summary)
    csv_file(directory, "b2_paired", paired)


def c2_tables(directory, reports):
    summary, matrices, contrasts, ledger, timing = [], [], [], [], []
    for backend, report in reports.items():
        for name in METHODS:
            trials = [row["runs"][name] for row in report["C"]]
            arm = {"backend": backend, "method": name}
            series = {metric: [t[metric] for t in trials] for metric in ("bwt", "live", "gain", "calls")}
            if name not in ENERGY:
                series["standalone_seconds_total"] = [sum(t["seconds_by_task"]) for t in trials]
            summary.extend({**arm, "metric": key, **stats(values)} for key, values in series.items())
            for trained in range(5):
                for evaluated in range(5):
                    values = [t["matrix"][trained][evaluated] for t in trials]
                    matrices.append({**arm, "trained_through_task": trained + 1, "evaluated_task": evaluated + 1,
                                     **stats(values)})
        energy = [row["energy"] for row in report["C"]]
        series = {metric: [r[metric] for r in energy] for metric in ("caps", "positive_steps", "max_residual")}
        series["joint_seconds_total"] = [sum(row["joint_seconds_by_task"]) for row in energy]
        for metric, values in series.items():
            summary.append({"backend": backend, "method": "energy_triplet", "metric": metric, **stats(values)})

        decision = report["decisions"]["C"]
        comparisons = [(name, control, result) for name in ("mass", "generic")
                       for control, result in decision[name]["contrasts"].items()]
        comparisons.append(("high", "low", decision["energy"]["high_vs_low"]))
        comparisons.extend((name, control, None) for name, controls in decision["energy"]["descriptive_vs_tuned"].items()
                           for control in controls)
        for name, control, primary in comparisons:
            bwt = [round(300 * (r["runs"][name]["bwt"] - r["runs"][control]["bwt"])) for r in report["C"]]
            live = [100 * (r["runs"][name]["live"] - r["runs"][control]["live"]) for r in report["C"]]
            contrasts.append({"backend": backend, "method": name, "control": control, "primary": primary is not None,
                              "bwt_mean_pp": mean(bwt) / 3, "bwt_sample_sd_pp": stdev(bwt) / 3,
                              "live_mean_pp": mean(live), "live_sample_sd_pp": stdev(live),
                              "seed_bwt_correct_deltas": json.dumps(dict(zip(report["seeds"], bwt))),
                              "seed_live_pp_deltas": json.dumps(dict(zip(report["seeds"], live))),
                              "wins": primary["wins"] if primary else None,
                              "non_ties": primary["non_ties"] if primary else None,
                              "p_one_sided": primary["p"] if primary else None})

        series = report["timing"]
        for name, repeats in series["seconds"].items():
            for repeat, (seconds, sgdm) in enumerate(zip(repeats, series["seconds"]["sgdm"], strict=True), 1):
                timing.append({"backend": backend, "seed": report["seeds"][0], "scope": "timed_repeat",
                               "method": name, "repeat": repeat, "task": None, "calls": series["calls_per_timed_block"],
                               "seconds": seconds, "ratio_to_sgdm": seconds / sgdm,
                               "paired_median_ratio": series["paired_median_ratio_to_sgdm"][name],
                               "at_most_1_05x": series["at_most_1_05x"][name]})

    for backend, seed, _, row in collect(reports):
        runs, energy = row["runs"], row["energy"]
        for name in (*[n for n in METHODS if n not in ENERGY], "energy_triplet"):
            joint = name == "energy_triplet"
            seconds_by_task = energy["joint_seconds_by_task"] if joint else runs[name]["seconds_by_task"]
            for task, seconds in enumerate(seconds_by_task, 2):
                timing.append({"backend": backend, "seed": seed, "scope": "continual_held_out", "method": name,
                               "repeat": None, "task": task, "calls": 144 if joint else 48, "seconds": seconds,
                               "ratio_to_sgdm": None if joint else seconds / runs["sgdm"]["seconds_by_task"][task - 2]})
        for step, (nominal, low, high, target, iso, lo, hi) in enumerate(energy["ledger"], 1):
            ledger.append({"backend": backend, "seed": seed, "step": step, "nominal_iso_j": nominal,
                           "low_capacity_j": low, "high_capacity_j": high, "target_j": target,
                           "realized_iso_j": iso, "realized_low_j": lo, "realized_high_j": hi,
                           "capped": target < nominal, "max_abs_residual_j": max(abs(x - target) for x in (iso, lo, hi))})
    for name, rows in (("c2_summary", summary), ("c2_matrices", matrices), ("c2_paired", contrasts),
                       ("energy_ledger", ledger), ("timing", timing)):
        csv_file(directory, name, rows)


def horizon_tables(directory, reports):
    raw, summary = [], []
    for backend, report in reports.items():
        runs = report["horizons"]
        for seed, run in zip(report["horizon_seeds"], runs, strict=True):
            raw.extend({"backend": backend, "seed": seed, **row} for row in run)
        horizons = sorted({row["T"] for row in runs[0]})
        for method, result in report["decisions"]["horizons"].items():
            for horizon, passes in zip(horizons, result["passes_on_grid"], strict=True):
                rows = [r for run in runs for r in run if r["method"] == method and r["T"] == horizon]
                for metric in (key for key in rows[0] if key not in ("method", "T")):
                    values = [r[metric] for r in rows]
                    summary.append({"backend": backend, "method": method, "T": horizon, "metric": metric,
                                    **stats(values), "max": max(values), "all_seeds_pass_grid_point": passes,
                                    **{k: v for k, v in result.items() if k != "passes_on_grid"}})
    csv_file(directory, "horizon_rows", raw)
    csv_file(directory, "horizon_summary", summary)


def validation_table(directory, validation):
    chosen = validation["chosen"]
    rows = []
    for family, grid in validation["raw"]["C"].items():
        for trial in grid:
            config = trial["cfg"]
            for method in ENERGY if family == "energy" else (family,):
                data = [r["arms"][method] if family == "energy" else r for r in trial["rows"]]
                for metric in ("live", "bwt", "gain", "calls"):
                    rows.append({"protocol": "C2", "family": family, "method": method, "config": json.dumps(config),
                                 "selected": config == chosen["C"][family], "metric": metric,
                                 **stats([row[metric] for row in data])})
            if family == "energy":
                rows.append({"protocol": "C2", "family": family, "method": "energy_triplet", "config": json.dumps(config),
                             "selected": config == chosen["C"][family], "metric": "caps",
                             **stats([row["caps"] for row in trial["rows"]])})
    for name, rates in validation["raw"]["B"]["rates"].items():
        for row in rates:
            rows.append({"protocol": "B2", "family": name, "method": name,
                         "config": json.dumps({"h": row["h"]}), "selected": row["h"] == chosen["B"][name],
                         "metric": "clean_live_both_regimes", **stats(row["clean_live"])})
    receipts = validation["raw"]["B"]["audit_receipts"]
    for threshold in (0.04, 0.08, 0.12):
        clean = [any(rate > threshold for rate in r["clean"]) for r in receipts]
        dirty = [next((i for i, rate in enumerate(r["dirty"]) if rate > threshold), None) == 2 for r in receipts]
        for metric, values in (("clean_alarm", clean), ("first_corrupted_block_alarm", dirty)):
            rows.append({"protocol": "B2a", "family": "receipt", "method": "receipt",
                         "config": json.dumps({"threshold": threshold}), "selected": threshold == chosen["B"]["threshold"],
                         "metric": metric, **stats(values), "count_true": sum(values)})
    csv_file(directory, "validation_summary", rows)


def save(figure, directory, name):
    figure.savefig(directory / f"{name}.png", dpi=155)
    plt.close(figure)


def plot_recovery(directory, reports):
    # Both backends share test seeds, so show CUDA and retain both in the tables.
    report = reports["cuda"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.8), layout="constrained", sharex=True, sharey=True)
    stages = ("target_acc", "raw_acc_at_alarm", "endpoint_acc", "equal_budget_acc")
    labels = ("True origin", "At alarm", "Window end", "Equal budget")
    for column, (regime, budget) in enumerate((("a", 96), ("b", 64))):
        for row, condition in enumerate(("dirty", "clean")):
            ax = axes[row, column]
            data = [item[regime][condition]["policies"] for item in report["B"]]
            for (name, label), color, marker in zip(POLICIES.items(), COLORS, ("o", "^", "s", "D", "x", "+")):
                values = [stats([policies[name][metric] for policies in data])["mean"] for metric in stages]
                ax.plot(range(4), [float("nan") if v is None else 100 * v for v in values],
                        color=color, marker=marker, label=label)
            ax.set(title=f"B2{regime} {condition} · {budget} gradients", ylabel="Task 1 accuracy (%)" if not column else None)
            ax.set_xticks(range(4), labels=labels)
            ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="outside lower center")
    figure.suptitle("Retroactive recovery on CUDA · 12 paired seeds\nAt-alarm means use windows with an alarm")
    save(figure, directory, "poison_recovery")


def plot_horizon(directory, reports):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.3), layout="constrained", sharex=True)
    for row, (backend, report) in enumerate(reports.items()):
        back, forward = axes[row]
        horizons = sorted({r["T"] for r in report["horizons"][0]})
        for name, method in (("glide", "Native f32"), ("lattice", "Integer")):
            runs = [[r for r in run if r["method"] == name] for run in report["horizons"]]
            color = COLORS[0 if name == "glide" else 1]
            for part, style, marker in (("q", "-", "o"), ("p", "--", "s")):
                for run in runs:
                    back.plot(horizons, [r[f"{part}_l2"] for r in run], style, color=color, alpha=0.2)
                values = [mean(r[f"{part}_l2"] for r in rows) for rows in zip(*runs)]
                back.plot(horizons, values, style, color=color, marker=marker, label=f"{method} {part}")
                if name == "lattice":
                    values = [mean(r[f"fwd_distance_from_native_{part}"] for r in rows) for rows in zip(*runs)]
                    forward.plot(horizons, values, style, color=color, marker=marker, label=f"Integer vs native {part}")
        for ax in (back, forward):
            ax.set_xscale("log")
            ax.set_xticks(horizons, labels=horizons)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_yscale("symlog", linthresh=1e-9)
            ax.set_ylim(-0.2e-9, max(1e-8, 1.3 * ax.get_ylim()[1]))
            ax.set_ylabel("Absolute L2 error")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        back.axhline(1e-5, color="0.4", linestyle=":", label="Native tolerance")
        back.legend(fontsize=8)
        back.set_title(f"{backend.upper()} rollback error")
        forward.set_title(f"{backend.upper()} forward disagreement")
    figure.suptitle("Replay horizon on the frozen grid · 3 seeds per backend")
    figure.supxlabel("Replayed steps T")
    save(figure, directory, "rollback_horizon")


def plot_pareto(directory, reports):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9.5), layout="constrained", width_ratios=(1, 1.25))
    artists = []
    for row, (backend, report) in enumerate(reports.items()):
        accuracy, timing = axes[row]
        means = {name: (100 * mean(r["runs"][name]["live"] for r in report["C"]),
                        100 * mean(r["runs"][name]["bwt"] for r in report["C"])) for name in METHODS}
        frontier = sorted({(x, y) for x, y in means.values()
                           if not any(a >= x and b >= y and (a, b) != (x, y) for a, b in means.values())})
        accuracy.plot(*zip(*frontier), linestyle=":", color="0.35", zorder=1)
        for index, (name, (live, bwt)) in enumerate(means.items()):
            dot = accuracy.scatter(live, bwt, s=65, marker=("o", "s", "^", "D", "P", "X", "o", "s", "^")[index],
                                   facecolors="none" if name in ENERGY else COLORS[index],
                                   edgecolors=COLORS[index], zorder=2, label=METHODS[name])
            if row == 0:
                artists.append(dot)
        accuracy.set(title=f"{backend.upper()} retention vs adaptation", xlabel="Mean live accuracy (%)",
                     ylabel="Mean Task 1 backward transfer (pp)")
        accuracy.grid(alpha=0.25)
        standalone = [name for name in METHODS if name not in ENERGY]
        series = [[1000 * sum(r["runs"][name]["seconds_by_task"]) for r in report["C"]] for name in standalone]
        series.append([1000 * sum(r["energy"]["joint_seconds_by_task"]) for r in report["C"]])
        labels = [METHODS[name] for name in standalone] + ["Energy triplet · 576 gradients"]
        bars = timing.barh(labels, [mean(v) for v in series], xerr=[stdev(v) for v in series],
                           color=(*COLORS[:6], "#bbbbbb"), capsize=3)
        bars[-1].set_hatch("//")
        timing.invert_yaxis()
        timing.set(title=f"{backend.upper()} cost per seed", xlabel="Elapsed milliseconds across Tasks 2–5 (mean ± SD)")
        timing.grid(axis="x", alpha=0.25)
    figure.legend(handles=artists, labels=list(METHODS.values()), ncol=5, loc="outside lower center")
    figure.suptitle("Continual learning · 192 gradients per arm\nDotted lines show the accuracy frontier; hollow points share triplet timing")
    save(figure, directory, "retention_pareto")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing validation.json and both held-out reports")
    directory = parser.parse_args().directory.resolve()
    outputs = Path(__file__).resolve().parents[1] / "artifacts" / "sol" / "p2"
    if directory.parent != outputs.resolve():
        parser.error("directory must be directly under artifacts/sol/p2")
    receipt = (directory / "validation.json").read_bytes()
    validation = json.loads(receipt)
    reports = {backend: json.loads((directory / f"test_{backend}.json").read_bytes()) for backend in ("cpu", "cuda")}
    for backend, report in reports.items():
        if (report["device"] != backend or report["stage"] != "test"
                or report["validation_sha256"] != sha256(receipt).hexdigest()
                or any(report[k] != validation[k] for k in ("eval_v2_sha256", "eval_v1_sha256", "factory",
                                                            "adapter_python_tree_sha256", "chosen"))):
            raise ValueError(f"{backend} report does not match its validation receipt")
    if reports["cpu"]["seeds"] != reports["cuda"]["seeds"]:
        raise ValueError("CPU and CUDA test seeds differ")
    b2_tables(directory, reports)
    c2_tables(directory, reports)
    horizon_tables(directory, reports)
    validation_table(directory, validation)
    plot_recovery(directory, reports)
    plot_horizon(directory, reports)
    plot_pareto(directory, reports)


if __name__ == "__main__":
    main()
