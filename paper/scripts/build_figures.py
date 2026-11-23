"""Render the four paper figures using SHA-256-verified raw artifacts.

Run from the repository root with Python and Matplotlib. Figure 4 evaluates
closed-form byte counts. The other panels show measured values.
"""

from math import floor
import os
from pathlib import Path
from statistics import median

OUTPUT = Path(__file__).resolve().parents[1] / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT.parent / ".build" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

from build_tables import P1, P2, P3, load

HEAD = "theory/evidence/phase3/sol_head_control.json"
BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#D55E00", "#007C5B", "#925995"
GREY, BLACK = "#6C757D", "#222831"
MIB, GIB = 2**20, 2**30


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": BLACK,
        "axes.labelcolor": BLACK, "axes.edgecolor": "#8A969D", "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 6,
        "xtick.color": BLACK, "ytick.color": BLACK, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": "#CDD4D8", "grid.linewidth": 0.55, "grid.alpha": 0.65,
        "legend.frameon": False, "legend.fontsize": 7.7, "lines.linewidth": 1.4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })


def save(figure, name, *, pad_inches=0.07):
    for extension in ("pdf", "png"):
        path = OUTPUT / f"{name}.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=pad_inches, metadata=metadata)
    plt.close(figure)
    print(OUTPUT / name)


def log_steps(ax, steps):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(steps))
    ax.set_xticklabels([str(t) for t in steps])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_xlabel("Replayed steps")


def phase_error(rows, key):
    maxima = {}
    for row in rows:
        t = row["T"]
        maxima[t] = max(maxima.get(t, 0.0), key(row))
    return sorted(maxima.items())


def fraction(h):
    denominator = round(1 / h)
    if h != 1 / denominator:
        raise ValueError(f"expected reciprocal-integer step size, got {h}")
    return f"1/{denominator}"


def figure_precision():
    figure = plt.figure(figsize=(7.25, 4.9), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(3.4, 1.25))
    mlp, transformer, exact = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, :])
    phase2_lattice = None
    for backend, color, marker in (("cpu", BLUE, "o"), ("cuda", ORANGE, "s")):
        report = load(f"{P2}/test_{backend}.json")
        dense = load(f"{P2}/dense_{backend}.json")
        rows = [r for seed in report["horizons"] for r in seed if r["method"] == "glide"]
        lattice = report["decisions"]["horizons"]["lattice"]
        if not all(lattice["passes_on_grid"]) or not lattice["right_censored"]:
            raise ValueError("figure requires exact lattice replay on the full Phase 2 grid")
        if phase2_lattice not in (None, lattice["T_max_grid_prefix"]):
            raise ValueError("Phase 2 lattice horizons differ between devices")
        phase2_lattice = lattice["T_max_grid_prefix"]
        dense_points = phase_error(dense["rows"], lambda r: max(r["q_l2"], r["p_l2"]))
        frozen_points = phase_error(rows, lambda r: max(r["q_l2"], r["p_l2"]))
        mlp.plot(*zip(*dense_points), color=color, label=backend.upper())
        mlp.plot(*zip(*frozen_points), linestyle="none", marker=marker, ms=4.4,
                 color=color, markeredgecolor="white", markeredgewidth=0.45, zorder=4)
    mlp.axhline(dense["absolute_l2_limit"], color=GREY, linestyle=":", lw=1)
    mlp.legend(loc="upper left", handlelength=2.2)
    mlp.set_title("(a) Native float32, multilayer perceptron")
    mlp.set_ylabel(r"Max absolute error, $\max(\|\Delta q\|_2,\|\Delta p\|_2)$")
    mlp.set_yscale("log")
    log_steps(mlp, (1, 8, *[t for t, _ in frozen_points]))

    physics = load(f"{P3}/test_parts/physics.json")["physics"]
    runs = physics["native_runs"]
    colors = dict(zip(sorted({r["h"] for r in runs}), (BLUE, GREEN, ORANGE), strict=True))
    for j, seed in enumerate(dict.fromkeys(r["seed"] for r in runs)):
        for run in (r for r in runs if r["seed"] == seed):
            samples = run["samples"]
            transformer.plot([r["T"] for r in samples], [r["relative_error"] for r in samples],
                             marker=("o", "s")[j], ms=3.6, linestyle=("-", "--")[j], color=colors[run["h"]],
                             label=rf"$h={fraction(run['h'])}$" if j == 0 else None)
    transformer.axhline(1e-5, color=GREY, linestyle=":", lw=1)
    transformer.legend(loc="upper left", ncol=1, handlelength=2.2)
    transformer.set_title("(b) Native float32, Transformer")
    transformer.set_yscale("log")
    transformer.set_ylabel(r"Relative phase error, $\rho_T$")
    log_steps(transformer, tuple(r["T"] for r in samples))

    prefix = load(f"{P3}/prefix.json")["formats"]
    exact.set_axis_off()
    exact.set_title("(c) Exact integer replay and range checks")
    columns = (0.02, 0.45, 0.77)
    for x, label in zip(columns, ("Model and format", "Last audited exact", "First rejected step"), strict=True):
        exact.text(x, 0.84, label, transform=exact.transAxes, fontsize=8.1, fontweight="bold")
    records = [("MLP int64 (grid)", phase2_lattice, "Unobserved")]
    for method in ("int32", "int64"):
        row = prefix[method]
        if row["status"] != "prefix_exact":
            raise ValueError(f"expected an exact pre-rejection prefix for {method}")
        records.append((f"Transformer {method} (audit)", row["prefix_steps"], str(row["first_rejected_step"])))
    for y, (label, last_step, rejected) in zip((0.58, 0.35, 0.12), records, strict=True):
        for x, text, color in zip(columns, (label, f"{last_step} steps", rejected), (BLACK, GREEN, ORANGE), strict=True):
            exact.text(x, y, text, transform=exact.transAxes, fontsize=8.1, color=color)
    save(figure, "fig1_precision_horizon")


def figure_continual():
    one = load(f"{P1}/continual.json")["verdict"]["paired_retention"]
    phase2 = load(f"{P2}/test_cuda.json")
    two = phase2["decisions"]["C"]

    def paired(method, peer):
        return [100 * (r["runs"][method]["bwt"] - r["runs"][peer]["bwt"]) for r in phase2["C"]]

    first = [
        (f"Fisher brake vs. {name}", [100 * d for d in one[k]["paired_deltas"]],
         100 * one[k]["mean"], one[k]["p_one_sided"])
        for k, name in (("sgdm", "momentum SGD"), ("adamw", "AdamW"), ("strang", "isotropic brake"))
    ]
    second = []
    baselines = {"sgdm": "momentum SGD", "adamw": "AdamW", "ewc": "EWC", "glide": "unit mass", "mass": "Fisher mass"}
    for method, label in (("mass", "Fisher mass"), ("generic", "GENERIC")):
        for peer, row in two[method]["contrasts"].items():
            second.append((f"{label} vs. {baselines[peer]}", paired(method, peer), row["mean_pp"], row["p"]))
    row = two["energy"]["high_vs_low"]
    second.append(("High- vs. low-Fisher brake", paired("high", "low"), row["mean_pp"], row["p"]))

    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.7),
                                gridspec_kw={"height_ratios": (3.5, 9.5), "width_ratios": (8.2, 1.6)})
    figure.subplots_adjust(left=0.32, right=0.98, top=0.93, bottom=0.13, hspace=0.38, wspace=0.04)
    all_contrasts = [v for _, values, _, _ in first + second for v in values]
    bound = min(all_contrasts) - 3, max(all_contrasts) + 5
    margin = 100 * one["sgdm"]["margin"]
    for i, (records, title, color) in enumerate(((first, "(a) Phase 1 Fisher brake", BLUE), (second, "(b) Phase 2 alternatives", ORANGE))):
        ax, side = axes[i]
        count = len(records)
        ax.set_title(title)
        ax.set_xlim(bound)
        ax.set_ylim(count - 0.5, -0.55)
        ax.set_yticks(range(count), [label for label, *_ in records])
        ax.tick_params(axis="y", which="both", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.axvline(0, color=GREY, linewidth=0.85)
        ax.axvline(margin, color=GREEN, linewidth=1.2, linestyle="--")
        for j, (_, values, average, p) in enumerate(records):
            offsets = [j + 0.16 * (2 * k / (len(values)-1) - 1) for k in range(len(values))]
            ax.scatter(values, offsets, color=color, s=15, alpha=0.32, edgecolors="none")
            ax.scatter(average, j, color=color, s=30, marker="D", edgecolors="white", linewidths=0.7, zorder=4)
            side.text(0.58, j, f"{p:.4f}" if p != 1 else "1.000", va="center", ha="right", fontsize=8.1)
        side.set_ylim(ax.get_ylim())
        side.set_axis_off()
        side.set_title(r"One-sided $p$", fontsize=8.0)
        if i == 0:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1, 0].set_xlabel("Paired Task 1 backward transfer difference (percentage points)")
    figure.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=GREY, alpha=0.6, ms=4, label="Paired seed"),
        Line2D([], [], marker="D", linestyle="none", color=GREY, ms=4, label="Mean"),
        Line2D([], [], color=GREEN, linestyle="--", lw=1.2, label=f"+{margin:g} pp claim margin"),
    ], loc="lower center", bbox_to_anchor=(0.62, -0.01), ncol=3, columnspacing=1.5)
    save(figure, "fig2_continual_falsification", pad_inches=0.13)


def pareto(points, resource):
    return {method for method, point in points.items() if not any(
        other[resource] <= point[resource] and other["time"] <= point["time"] and
        (other[resource] < point[resource] or other["time"] < point["time"])
        for other in points.values()
    )}


def figure_systems():
    manifest = load(f"{P3}/test.json")
    depth = max(map(int, manifest["decisions"]["depths"]))
    aggregate = manifest["decisions"]["depths"][str(depth)]["methods"]
    styles = {
        "native": ("Native DKD", BLUE, "o"), "int32": ("Int32 DKD", BLUE, "s"),
        "int64": ("Int64 DKD", BLUE, "D"), "sgd_gpu": ("GPU SGD", ORANGE, "o"),
        "sgd_cpu": ("Pinned SGD", ORANGE, "s"), "adam_gpu": ("GPU AdamW", GREEN, "o"),
        "adam_cpu": ("Pinned AdamW", GREEN, "s"), "single": ("Single path", GREY, "^"),
        "head": ("Head-only SGD (post-hoc)", PURPLE, "*"),
    }
    points = {}
    for method in styles:
        if method == "head":
            rows = load(HEAD)["cell"]["rows"]
            full_solve = all(any(leaf["terminal_pass"][secret] for leaf in row["leaves"])
                             for row in rows for secret in range(row["D"]+1))
        else:
            rows = load(f"{P3}/test_parts/{method}_{depth}.json")["cell"]["rows"]
            full_solve = aggregate[method]["exact_prior"] == 1
        points[method] = {
            "gpu": max(r["memory"]["cuda_peak_allocated"] for r in rows) / MIB,
            "host": max(r["memory"]["os_process_lifetime_peak_rss"] for r in rows) / GIB,
            "time": median(r["elapsed"] for r in rows),
            "full_solve": full_solve,
        }

    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.4), sharey=True)
    figure.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.32, wspace=0.23)
    measurement = {m: point for m, point in points.items() if point["full_solve"] and m != "head"}
    for ax, resource, title, xlabel in zip(
        axes, ("gpu", "host"), ("(a) GPU allocation", "(b) Host working set"),
        ("Peak allocated GPU memory (MiB)", "Peak OS working set (GiB)"), strict=True,
    ):
        front = pareto(measurement, resource)
        for method, point in points.items():
            _, color, marker = styles[method]
            size = (52 if method == "head" else 15) if resource == "host" else (78 if method == "head" else 44)
            if method in front:
                ax.scatter(point[resource], point["time"], s=3 * size, color="none", edgecolor=GREY,
                           alpha=0.7, linewidths=0.9, zorder=2)
            ax.scatter(point[resource], point["time"], s=size,
                       marker=marker, facecolor=color if point["full_solve"] else "white", edgecolor=color,
                       linewidths=0.75 if resource == "host" else 1.15, zorder=3)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylim(0, 2.65)
    axes[0].set_ylabel("Median seconds to exhaust search")
    legend = [Line2D([], [], marker=marker, linestyle="none", markerfacecolor=color if points[m]["full_solve"] else "white",
                     markeredgecolor=color, markeredgewidth=1.1, markersize=6.2, label=label)
              for m, (label, color, marker) in styles.items()]
    order = (0, 3, 5, 1, 4, 6, 2, 7, 8)
    figure.legend(handles=[legend[i] for i in order], ncol=3, loc="lower center", bbox_to_anchor=(0.54, 0.01),
                  columnspacing=1.5, handletextpad=0.35)
    save(figure, "fig3_systems_pareto")


def figure_scale():
    full = load(f"{P3}/test_parts/native_8.json")["cell"]["rows"][0]
    head = load(HEAD)
    head_n, full_n, depth, steps = head["trainable_parameters"], full["n"], full["D"], full["K"]
    if head["full_parameters"] != full_n or full["peak_tape_payload"] != 8 * steps * depth:
        raise ValueError("measured topology disagrees with the analytical model")
    if not 0 < steps < head_n < full_n:
        raise ValueError("invalid analytical scales")

    figure, (left, right) = plt.subplots(1, 2, figsize=(7.25, 3.8))
    figure.subplots_adjust(left=0.105, right=0.98, top=0.87, bottom=0.27, wspace=0.37)
    names = ("Int32 lattice + IDs", "Int64 lattice + IDs", "GPU SGD + checkpoints", "GPU AdamW + checkpoints")
    colors = (BLUE, GREEN, ORANGE, PURPLE)

    def footprints(n, forks):
        tape = 8 * steps * forks
        return 8*n + tape, 16*n + tape, 8*n*(1+forks), (12*n+8)*(1+forks)

    counts = [head_n * (full_n / head_n) ** (i / 80) for i in range(81)]
    for ys, color, line in zip(zip(*(footprints(n, depth) for n in counts)), colors, ("-", "-", "--", "--"), strict=True):
        left.plot(counts, [y / MIB for y in ys], color=color, linestyle=line)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xticks((head_n, 10**5, 10**6, full_n), (f"{head_n:,}", "100k", "1M", f"{full_n/10**6:.2f}M"))
    left.xaxis.set_minor_formatter(NullFormatter())
    left.set_xlabel("Trainable coordinates")
    left.set_ylabel("Modeled bytes (MiB)")
    left.set_title(f"(a) Footprint at {depth} unresolved forks")

    forks = list(range(depth + 1))
    for i, color in enumerate(colors[:2]):
        ratios = [footprints(full_n, f)[i] / footprints(full_n, f)[2] for f in forks]
        right.plot(forks, ratios, marker=("o", "s")[i], ms=4.0, color=color)
    right.axhline(1.0, color=ORANGE, linestyle="--", lw=1)
    threshold = floor(full_n / (full_n - steps)) + 1
    right.annotate(f"Int64 first wins at {threshold} forks", xy=(threshold, footprints(full_n, threshold)[1] / footprints(full_n, threshold)[2]),
                   xytext=(threshold + 0.8, 1.48), color=GREEN, fontsize=7.9,
                   arrowprops={"arrowstyle": "->", "color": GREEN, "linewidth": 0.75})
    right.set_xticks(forks)
    right.set_xlim(-0.3, depth + 0.3)
    right.set_ylim(0, 2.12)
    right.set_xlabel("Unresolved forks")
    right.set_ylabel("Modeled bytes / GPU SGD bytes")
    right.set_title(f"(b) Full model ({full_n/10**6:.2f}M)")
    for ax in (left, right):
        ax.grid(True)
        ax.set_axisbelow(True)
    figure.legend(handles=[Line2D([], [], color=c, linestyle=ls, label=n) for n, c, ls in zip(names, colors, ("-", "-", "--", "--"), strict=True)],
                  loc="lower center", ncol=2, bbox_to_anchor=(0.54, 0.01), columnspacing=1.5)
    save(figure, "fig4_scale_crossover")


if __name__ == "__main__":
    style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure_precision()
    figure_continual()
    figure_systems()
    figure_scale()
