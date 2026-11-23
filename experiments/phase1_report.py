"""Summarize the frozen Phase 1 JSON artifacts and render four figures."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = {"glide": "Glide", "strang": "Isotropic brake", "fisher": "Fisher brake",
           "sgdm": "Momentum SGD", "adamw": "AdamW"}
ACCURACIES = {"pre_acc": "Before window", "uncorrected_acc": "After updates",
              "post_policy_acc": "After policy", "followup_acc": "32 clean steps later"}
EPISODE_METRICS = (*ACCURACIES, "uncorrected_displacement_l2", "rollback_q_l2",
                   "rollback_p_l2", "rewind_ms", "first_alarm_zero_based", "alarm", "rewound",
                   "uncorrected_damage", "post_policy_damage")
SEQUENTIAL_METRICS = ("task1_bwt", "live_acc", "arrival_to_live_gain", "ms_per_step",
                      "time_ratio_to_sgdm")


def describe(values):
    valid = [value for value in values if value is not None]
    return len(valid), mean(valid) if valid else None, stdev(valid) if len(valid) > 1 else None


def write_tables(directory, methods):
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["method", "protocol", "metric", "n", "mean", "sample_std", "count_true"])
        for name, seeds in methods.items():
            for condition in ("clean", "poison"):
                episodes = [item[condition] for item in seeds]
                for metric in EPISODE_METRICS:
                    values = [episode[metric] for episode in episodes]
                    total = sum(values) if metric in ("alarm", "rewound") else None
                    writer.writerow([name, condition, metric, *describe(values), total])
            for metric in SEQUENTIAL_METRICS:
                values = [item["sequential"][metric] for item in seeds]
                writer.writerow([name, "continual", metric, *describe(values), None])
            for index in range(4):
                curves = [item["sequential"]["arrival_to_live_12_step_curves"][index] for item in seeds]
                for checkpoint in range(5):
                    values = [curve[checkpoint] for curve in curves]
                    writer.writerow([name, f"task_{index + 2}", f"accuracy_at_step_{12 * checkpoint}",
                                     *describe(values), None])
                values = [curve[-1] - curve[0] for curve in curves]
                writer.writerow([name, f"task_{index + 2}", "arrival_to_live_gain",
                                 *describe(values), None])
                values = [item["sequential"]["optimizer_wall_seconds_by_task"][index] for item in seeds]
                writer.writerow([name, f"task_{index + 2}", "optimizer_wall_seconds",
                                 *describe(values), None])
            values = [item["clean"]["followup_acc"] - item["poison"]["followup_acc"] for item in seeds]
            writer.writerow([name, "matched_windows", "clean_minus_poison_followup_acc",
                             *describe(values), None])

    matrices = {}
    with (directory / "retention.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["method", "trained_through_task", "evaluated_task", "n", "mean", "sample_std"])
        for name, seeds in methods.items():
            matrix = []
            for trained in range(5):
                row = []
                for evaluated in range(5):
                    values = [item["sequential"]["test_accuracy_matrix"][trained][evaluated]
                              for item in seeds]
                    count, average, std = describe(values)
                    writer.writerow([name, trained + 1, evaluated + 1, count, average, std])
                    row.append(100 * average)
                matrix.append(row)
            matrices[name] = matrix
    return matrices


def save(figure, directory, name):
    figure.savefig(directory / f"{name}.png", dpi=150)
    plt.close(figure)


def plot_rollback(directory, analytic):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, dtype, threshold in zip(axes, ("f64", "f32"), (1e-17, 1e-9)):
        rows = analytic[f"rollback_{dtype}"]
        steps = [row["steps"] for row in rows]
        for key, label, marker in (("q_l2", "Weights", "o"), ("p_l2", "Momentum", "s")):
            ax.plot(steps, [row[key] for row in rows], marker=marker, label=label)
        ax.set_xscale("log")
        ax.set_xticks(steps, labels=steps)
        ax.set_yscale("symlog", linthresh=threshold)
        peak = max(max(row["q_l2"], row["p_l2"]) for row in rows)
        ax.set_ylim(-0.25 * threshold, 2 * max(threshold, peak))
        ax.set(title=dtype, xlabel="Replayed steps", ylabel="Absolute L2 error")
        ax.grid(alpha=0.25)
        ax.legend()
    figure.suptitle("Double-well rollback error")
    save(figure, directory, "rollback")


def plot_energy(directory, analytic):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharex=True)
    for index, (ax, label) in enumerate(zip(axes, ("Physical energy", "Second-order shadow energy"))):
        for row in analytic["energy"]:
            trace = row["physical_and_shadow_trace"]
            times = [step * row["h"] for step in range(len(trace))]
            ax.plot(times, [point[index] - trace[0][index] for point in trace], label=f"h = {row['h']:g}")
        ax.set(title=label, xlabel="Integration time", ylabel="Change from initial energy")
        ax.grid(alpha=0.25)
        ax.legend()
    figure.suptitle("Double-well energy across step sizes")
    save(figure, directory, "energy")


def plot_poison(directory, methods):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained", sharey=True)
    for ax, condition in zip(axes, ("poison", "clean")):
        for marker, (name, seeds) in zip(("o", "s", "^", "D", "x"), methods.items()):
            averages = [100 * mean(item[condition][metric] for item in seeds) for metric in ACCURACIES]
            ax.plot(range(len(averages)), averages, marker=marker, label=METHODS[name])
        ax.set_xticks(range(len(ACCURACIES)), labels=list(ACCURACIES.values()), rotation=15, ha="right")
        ax.set(title=f"{condition.capitalize()} window", ylabel="Mean holdout accuracy (%)")
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(loc="lower right")
    figure.suptitle("Accuracy before adaptation, after policy, and after follow-up")
    save(figure, directory, "poison_recovery")


def plot_retention(directory, matrices):
    figure, grid = plt.subplots(2, 3, figsize=(11, 6.4), layout="constrained")
    axes = list(grid.flat)
    axes.pop().remove()
    for ax, (name, matrix) in zip(axes, matrices.items()):
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis")
        ax.set_title(METHODS[name])
        ax.set_xticks(range(5), labels=range(1, 6))
        ax.set_yticks(range(5), labels=range(1, 6))
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                ax.text(column, row, f"{value:.1f}", ha="center", va="center",
                        color="black" if value >= 55 else "white", fontsize=8)
    figure.colorbar(image, ax=axes, label="Mean holdout accuracy (%)", shrink=0.85)
    figure.suptitle("Five-task continual learning")
    figure.supxlabel("Evaluated task")
    figure.supylabel("Trained through task")
    save(figure, directory, "retention")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="frozen run directory containing analytic.json and continual.json")
    directory = parser.parse_args().directory.resolve()
    root = Path(__file__).resolve().parents[1]
    if not directory.is_relative_to((root / "experiments" / "results").resolve()):
        parser.error("the run directory must be under experiments/results")
    analytic = json.loads((directory / "analytic.json").read_text(encoding="utf-8"))
    continual = json.loads((directory / "continual.json").read_text(encoding="utf-8"))
    assert analytic["eval_sha256"] == continual["eval_sha256"]
    assert analytic["factory_module_sha256"] == continual["factory_module_sha256"]
    assert analytic["device"] == continual["device"]
    methods = {name: [row["methods"][name] for row in continual["rows"]] for name in METHODS}
    for seeds in methods.values():
        for item in seeds:
            for condition in ("clean", "poison"):
                episode = item[condition]
                episode["alarm"] = episode["first_alarm_zero_based"] is not None
                episode["uncorrected_damage"] = episode["pre_acc"] - episode["uncorrected_acc"]
                episode["post_policy_damage"] = episode["pre_acc"] - episode["post_policy_acc"]
    matrices = write_tables(directory, methods)
    plot_rollback(directory, analytic)
    plot_energy(directory, analytic)
    plot_poison(directory, methods)
    plot_retention(directory, matrices)


if __name__ == "__main__":
    main()
