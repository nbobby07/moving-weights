"""Reduce hash-pinned Phase 1--3 evidence into manuscript macros and tables."""

from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite
from os import environ
from pathlib import Path
from statistics import mean, median

from restore_evidence import read_pins


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
EVIDENCE_ROOT = Path(environ.get("MTTT_EVIDENCE_ROOT", ROOT)).resolve(strict=True)
P1 = "artifacts/sol"
P2 = f"{P1}/p2/sol-20261122-cpu-lock"
P3 = f"{P1}/p3/sol-20261122-cuda-lock"
AUDIT1 = "theory/evidence/phase1_replication_audit.json"
AUDIT2 = "theory/evidence/phase2/sol_replication_audit.json"
AUDIT3 = "theory/evidence/phase3/sol_replication_audit.json"
HEAD = "theory/evidence/phase3/sol_head_control.json"
PINS = read_pins()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or isfinite(value)


@lru_cache(maxsize=None)
def load(relative_repo_path):
    """Read pinned JSON without importing or running the frozen evaluators."""
    key = Path(relative_repo_path).as_posix()
    check(key in PINS, f"unregistered evidence path: {key}")
    path = EVIDENCE_ROOT / key
    check(path.resolve() == path, f"unexpected symlink for evidence path: {key}")
    payload = path.read_bytes()
    check(sha256(payload).hexdigest() == PINS[key], f"SHA-256 mismatch: {key}")
    value = json.loads(payload)
    check(finite(value), f"nonfinite number: {key}")
    return value


def pinned_digest(key):
    load(key)
    return PINS[key]


def verify():
    """Check every byte pin and independent audit-to-submission link."""
    for key in PINS:
        load(key)
    a1, a2, a3 = (load(key) for key in (AUDIT1, AUDIT2, AUDIT3))
    for backend, data in a1["backends"].items():
        for name, reproduction in data["reproduced"].items():
            key = reproduction["archive"]
            check(pinned_digest(key) == reproduction["raw_sha256"], f"Phase 1 audit mismatch: {key}")
            raw = load(key)
            check(raw["eval_sha256"] == a1["eval_sha256"] and raw["device"] == backend,
                  f"Phase 1 audit device or evaluator mismatch: {key}")
            if backend == "cuda":
                check(pinned_digest(f"{P1}/{name}.json") == reproduction["raw_sha256"],
                      f"Phase 1 accepted artifact mismatch: {name}")
    check(pinned_digest(f"{P2}/validation.json") == a2["validation_sha256"], "Phase 2 lock changed")
    for backend, data in a2["backends"].items():
        key = f"{P2}/test_{backend}.json"
        check(pinned_digest(key) == data["submitted_sha256"], f"Phase 2 audit mismatch: {key}")
        report = load(key)
        check(report["validation_sha256"] == a2["validation_sha256"] and
              report["adapter_python_tree_sha256"] == a2["adapter_tree_sha256"] and
              report["chosen"] == a2["validation_choices_rederived_from_counts"],
              f"Phase 2 selection mismatch: {key}")
        dense = f"{P2}/dense_{backend}.json"
        check(pinned_digest(dense) == data["replay"]["dense_raw_sha256_reproduced"] and
              load(dense)["first_failed_horizon"] == data["replay"]["earliest_native_f32_miss"],
              f"Phase 2 dense replay mismatch: {dense}")
    for name, digest in a3["input_hashes"].items():
        check(pinned_digest(f"{P3}/{name}.json") == digest, f"Phase 3 audit mismatch: {name}")
    check(pinned_digest(a3["receipt"]["path"]) == a3["receipt"]["sha256"] and
          a3["receipt"]["verified_before_first_worker_as_recorded"], "Phase 3 pre-test receipt mismatch")
    test = load(f"{P3}/test.json")
    for name, worker in test["parts"].items():
        key = f"{P3}/{worker['path']}"
        raw = load(key)
        check(pinned_digest(key) == worker["sha256"] == a3["worker_sha256"][name] and
              raw["validation_sha256"] == test["validation_sha256"] and
              raw["adapter_tree_sha256"] == test["adapter_tree_sha256"],
              f"Phase 3 worker provenance mismatch: {key}")
    check(pinned_digest(HEAD) == a3["head_only_control"]["raw_sha256"] and
          load(HEAD)["submission_lock"] == test["validation_sha256"], "Phase 3 head-control provenance mismatch")


def decimal(x, places=2):
    return format(x, f".{places}f")


def sci(x, places=2):
    if x == 0:
        return "0"
    mantissa, exp = format(x, f".{places}e").split("e")
    return rf"{mantissa}\times 10^{{{int(exp)}}}"


def integer(x):
    return format(x, ",d").replace(",", "{,}")


def fraction(x):
    f = Fraction(x)
    return str(f.numerator) if f.denominator == 1 else rf"{f.numerator}/{f.denominator}"


def rate(x):
    return f"{x:g}"


def math(x):
    return rf"\ensuremath{{{x}}}"


def row(*cells):
    return " & ".join(map(str, cells)) + r" \\"


def table(cols, headers, rows):
    return "\n".join([rf"\begin{{tabular}}{{{cols}}}", r"\toprule", row(*headers), r"\midrule",
                      *(row(*r) if isinstance(r, tuple) else r for r in rows), r"\bottomrule", r"\end{tabular}"])


class Build:
    def __init__(self):
        self.macros = {}
        self.outputs = {}

    def add(self, name, value, source, selector, *, fmt=str, text=False):
        check(name.isalpha() and name.isascii() and name not in self.macros, f"invalid or duplicate macro: {name}")
        value = fmt(value)
        rendered = value if text else math(value)
        self.macros[name] = (rendered, source, selector)
        return rendered

    def add_table(self, name, tex, sources):
        header = "\n".join(f"% Source: {s}" for s in sources)
        self.outputs[PAPER / "tables" / f"{name}.tex"] = f"% Generated by scripts/build_tables.py\n{header}\n{tex}\n"

    def write(self):
        lines = [r"% Generated by scripts/build_tables.py. Regenerate from scripts/source_pins.json."]
        for name, (value, source, selector) in self.macros.items():
            lines.append(f"% {source} :: {selector}")
            lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
        self.outputs[PAPER / "macros_verified.tex"] = "\n".join(lines) + "\n"
        for path, content in self.outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")


def pvalue(x):
    return "1.000" if x == 1 else decimal(x, 4)


def add_many(build, source, entries):
    for name, value, selector, fmt in entries:
        build.add(name, value, source, selector, fmt=fmt)


def phase_one(build):
    audit = load(AUDIT1)
    cuda = audit["backends"]["cuda"]
    dev = "backends.cuda."
    contrast = cuda["fisher_BWT_vs"]
    neural = cuda["neural_rollback"]
    poison_alarms = cuda["glide_alarms_zero_based"]["poison"]
    check(all(x == 0 for x in poison_alarms), "Phase 1 alarm timing no longer matches the manuscript")
    add_many(build, AUDIT1, (
        ("PoneSeeds", len(audit["seeds"]), "len(seeds)", str),
        ("PoneNeuralQ", neural["q_l2"], dev + "neural_rollback.q_l2", sci),
        ("PoneNeuralP", neural["p_l2"], dev + "neural_rollback.p_l2", sci),
        ("PoneNeuralSteps", neural["steps"], dev + "neural_rollback.steps", str),
        ("PonePoisonsDetected", len(poison_alarms),
         dev + "glide_alarms_zero_based.poison (all alarmed at index zero)", str),
        ("PoneCleanAlarms", sum(x is not None for x in cuda["glide_alarms_zero_based"]["clean"]),
         dev + "glide_alarms_zero_based.clean (count non-null)", str),
        ("PoneAdamGap", cuda["adamw_followup_gap_pp"], dev + "adamw_followup_gap_pp", decimal),
        ("PoneFisherSGDDelta", contrast["sgdm"]["mean_advantage_pp"],
         dev + "fisher_BWT_vs.sgdm.mean_advantage_pp", lambda x: decimal(x, 3)),
        ("PoneFisherAdamDelta", contrast["adamw"]["mean_advantage_pp"],
         dev + "fisher_BWT_vs.adamw.mean_advantage_pp", lambda x: decimal(x, 3)),
        ("PonePrimaryP", contrast["sgdm"]["p_one_sided"], dev + "fisher_BWT_vs.{sgdm,adamw}.p_one_sided", pvalue),
        ("PoneFisherStartPct", cuda["fisher_pre_pct"], dev + "fisher_pre_pct", decimal),
        ("PoneFisherEndPct", cuda["fisher_final_pct"], dev + "fisher_final_pct", decimal),
        ("PoneFisherLivePct", cuda["fisher_live_pct"], dev + "fisher_live_pct", decimal),
        ("PoneDimensions", cuda["total_dimensions"], dev + "total_dimensions", str),
    ))
    check(contrast["sgdm"]["p_one_sided"] == contrast["adamw"]["p_one_sided"], "Phase 1 primary p-values differ")
    build.add("PoneFollowup", int(next(k.rsplit("_", 1)[-1] for k in cuda["B_components"]
                                     if k.startswith("adamw_matched_residual_after_"))),
              AUDIT1, dev + "B_components.adamw_matched_residual_after_<followup>")
    analytic = load(f"{P1}/analytic.json")
    orders = analytic["orders_physical_shadow"]
    rows = [
        (r"Neural f64 position L2", r"\PoneNeuralQ"),
        (r"Neural f64 momentum L2", r"\PoneNeuralP"),
        (r"Physical-energy orders", math(r",\;".join(decimal(q, 4) for q, _ in orders))),
        (r"Shadow-energy orders", math(r",\;".join(decimal(p, 4) for _, p in orders))),
        (r"Glide poison detections", r"\PonePoisonsDetected/\PoneSeeds"),
        (r"Glide clean-window alarms", r"\PoneCleanAlarms/\PoneSeeds"),
        (r"AdamW-continue follow-up gap (pp)", r"\PoneAdamGap"),
        r"\midrule",
        (r"Fisher minus momentum SGD (pp; $p$)", r"\PoneFisherSGDDelta; \PonePrimaryP"),
        (r"Fisher minus AdamW (pp; $p$)", r"\PoneFisherAdamDelta; \PonePrimaryP"),
        (r"Fisher minus isotropic (pp; $p$)",
         f"{math(decimal(contrast['strang']['mean_advantage_pp'], 3))}; {math(pvalue(contrast['strang']['p_one_sided']))}"),
    ]
    build.add_table("phaseone", table("@{}lr@{}", ("Outcome or paired contrast", "CUDA result"), rows),
                    [AUDIT1 + " backends.cuda", f"{P1}/analytic.json orders_physical_shadow"])


def phase_two(build):
    run = f"{P2}/test_cuda.json"
    raw = load(run)
    dec = raw["decisions"]
    audit = load(AUDIT2)
    b = audit["backends"]["cuda"]["B"]
    c = audit["backends"]["cuda"]["C"]
    post = audit["backends"]["cuda"]["exploratory_B2b"]["effects"]["dirty"]
    cohorts = {arm: {window: [e[arm][window]["policies"] for e in raw["B"]]
                     for window in ("dirty", "clean")} for arm in ("a", "b")}
    check(all(len(cohort) == len(raw["seeds"]) for arm in cohorts.values() for cohort in arm.values()),
          "Phase 2 alarm cohorts differ from the held-out roster")

    def alarms(arm, window, policy):
        return sum(row[policy]["alarm_after_commits"] is not None for row in cohorts[arm][window])

    damage_key = next(key for key in b["b"]["components"] if key.startswith("outcome_verified_damage_at_least_"))
    damage_pp = int(damage_key.split("_at_least_")[1].split("pp")[0])
    per_seed_cases, remainder = divmod(post["denominator"], len(raw["seeds"]))
    check(remainder == 0 and damage_pp * per_seed_cases % 100 == 0, "Phase 2 damage threshold does not map to integer counts")
    damage_correct = damage_pp * per_seed_cases // 100
    for arm in cohorts:
        check(b[arm]["clean_rewind_alarms"] == alarms(arm, "clean", "rewind"), f"Phase 2 {arm} clean rewind alarm mismatch")
        damaged = sum(r["rewind"]["raw_acc_at_alarm"] is not None and
                      round(per_seed_cases * (r["rewind"]["target_acc"] - r["rewind"]["raw_acc_at_alarm"])) >= damage_correct
                      for r in cohorts[arm]["dirty"])
        check(damaged == b[arm]["material_damage_count"], f"Phase 2 {arm} dirty rewind damage mismatch")
    build.add("PtwoDamageThreshold", damage_pp, AUDIT2, "backends.cuda.B.b.components." + damage_key)
    for policy, tag in (("adam_skip", "Skip"), ("adam_continue", "Continue")):
        build.add(f"PtwoTerminalAdam{tag}CleanAlarms", alarms("b", "clean", policy), run,
                  f"count(B[*].b.clean.policies.{policy}.alarm_after_commits is non-null)")
    pref = "backends.cuda."
    arm_a = pref + "B.a."
    arm_b = pref + "B.b."
    add_many(build, AUDIT2, (
        ("PtwoRampRecovered", b["a"]["seeds_with_true_origin_recovery"], arm_a + "seeds_with_true_origin_recovery", str),
        ("PtwoRampDamaged", b["a"]["material_damage_count"], arm_a + "material_damage_count", str),
        ("PtwoRampEarly", len(b["a"]["passive_warning_commits"]), arm_a + "len(passive_warning_commits)", str),
        ("PtwoTerminalRecovered", b["b"]["seeds_with_true_origin_recovery"], arm_b + "seeds_with_true_origin_recovery", str),
        ("PtwoTerminalDamaged", b["b"]["material_damage_count"], arm_b + "material_damage_count", str),
        ("PtwoTerminalAdamImmediate", b["b"]["rewind_minus_adam_skip_endpoint_pp"], arm_b + "rewind_minus_adam_skip_endpoint_pp", decimal),
        ("PtwoTerminalAdamBudget", b["b"]["rewind_minus_adam_skip_equal_budget_pp"], arm_b + "rewind_minus_adam_skip_equal_budget_pp", decimal),
        ("PtwoCaps", c["energy"]["capped"], pref + "C.energy.capped", integer),
        ("PtwoEnergySteps", c["energy"]["rows"], pref + "C.energy.rows", integer),
        ("PtwoGenericMassDelta", c["generic"]["contrasts"]["mass"]["mean_pp"], pref + "C.generic.contrasts.mass.mean_pp", decimal),
        ("PtwoGenericMassP", c["generic"]["contrasts"]["mass"]["p"], pref + "C.generic.contrasts.mass.p", pvalue),
    ))
    post_fields = {
        "PtwoEarlySkipImmediateCorrect": "early_skip_endpoint_correct",
        "PtwoRewindImmediateCorrect": "rewind_endpoint_correct",
        "PtwoEarlySkipBudgetCorrect": "early_skip_equal_budget_correct",
        "PtwoRewindBudgetCorrect": "rewind_equal_budget_correct",
        "PtwoTerminalTotal": "denominator",
        "PtwoEarlyImmediateDelta": "rewind_minus_early_skip_endpoint_pp",
        "PtwoEarlyBudgetDelta": "rewind_minus_early_skip_equal_budget_pp",
    }
    for name, key in post_fields.items():
        fmt = integer if isinstance(post[key], int) else lambda x: decimal(x, 3)
        build.add(name, post[key], AUDIT2, pref + "exploratory_B2b.effects.dirty." + key, fmt=fmt)
    check(post == audit["backends"]["cpu"]["exploratory_B2b"]["effects"]["dirty"], "Phase 2 post-hoc counts differ by backend")
    early_rows = audit["backends"]["cuda"]["exploratory_B2b"]["rows"]
    dirty_observations = [r for r in early_rows if r["window"] == "dirty"]
    warnings = {window: [r["first_warning_after_updates"] for r in early_rows
                         if r["window"] == window and r["first_warning_after_updates"] is not None]
                for window in ("dirty", "clean")}
    check(all(sum(r["window"] == window for r in early_rows) == len(raw["seeds"])
              for window in warnings), "Phase 2 early-skip control omitted a window")
    for window in warnings:
        build.add("PtwoEarlyWarn" + window.title(), len(warnings[window]), AUDIT2,
                  pref + f"exploratory_B2b.rows count {window} windows with a warning")
    for edge, stat in (("First", min), ("Last", max)):
        build.add("PtwoEarlyWarn" + edge, stat(warnings["dirty"]), AUDIT2,
                  pref + f"exploratory_B2b.rows {stat.__name__} dirty first_warning_after_updates")
    for post_name, early_field, rewind_field in (("endpoint", "endpoint_task1_correct", "endpoint_acc"),
                                                  ("equal_budget", "equal_budget_task1_correct", "equal_budget_acc")):
        early = sum(r[early_field] for r in dirty_observations)
        denominator = post["denominator"]
        n_per_seed, remainder = divmod(denominator, len(dirty_observations))
        rewind = sum(round(r["b"]["dirty"]["policies"]["rewind"][rewind_field] * n_per_seed) for r in raw["B"])
        check(remainder == 0 and early == post[f"early_skip_{post_name}_correct"] and
              rewind == post[f"rewind_{post_name}_correct"] and
              (rewind - early) * 100 / denominator == post[f"rewind_minus_early_skip_{post_name}_pp"],
              f"Phase 2 early-skip control count mismatch: {post_name}")
    for backend in ("CPU", "CUDA"):
        source = f"{P2}/dense_{backend.lower()}.json"
        build.add("PtwoDense" + backend, load(source)["first_failed_horizon"], source, "first_failed_horizon")
    build.add("PtwoPrecisionThreshold", load(f"{P2}/dense_cuda.json")["absolute_l2_limit"],
              f"{P2}/dense_cuda.json", "absolute_l2_limit", fmt=sci)
    horizon = dec["horizons"]
    live = dec["C"]["live_means"]
    add_many(build, run, (
        ("PtwoSeeds", len(raw["seeds"]), "len(seeds)", str),
        ("PtwoGenericLivePct", 100 * live["generic"], "100*decisions.C.live_means.generic", decimal),
        ("PtwoAdamLivePct", 100 * live["adamw"], "100*decisions.C.live_means.adamw", decimal),
        ("PtwoGridLast", horizon["glide"]["T_max_grid_prefix"], "decisions.horizons.glide.T_max_grid_prefix", str),
        ("PtwoGridFail", horizon["glide"]["first_failed_grid_horizon"], "decisions.horizons.glide.first_failed_grid_horizon", str),
        ("PtwoIntegerLast", horizon["lattice"]["T_max_grid_prefix"], "decisions.horizons.lattice.T_max_grid_prefix", str),
        ("PtwoTerminalBudget", raw["B"][0]["b"]["dirty"]["policies"]["rewind"]["calls"]["total"], "B[0].b.dirty.policies.rewind.calls.total", str),
        ("PtwoTerminalUpdates", raw["B"][0]["b"]["dirty"]["policies"]["rewind"]["calls"]["forward"], "B[0].b.dirty.policies.rewind.calls.forward", str),
        ("PtwoRampMinChanges", min(x["a"]["dirty"]["policies"]["rewind"]["changed_post_onset_before_alarm"] for x in raw["B"]),
         "min(B[*].a.dirty.policies.rewind.changed_post_onset_before_alarm)", str),
    ))
    # The alpha and tolerance are frozen protocol constants, not observed results.
    n_contrasts = 1 + sum(len(c[arm]["contrasts"]) for arm in ("mass", "generic"))
    build.add("PtwoCutoff", f"0.05/{n_contrasts}", "theory/phase2_math_spec.md", "familywise alpha / registered primary contrasts")
    build.add("PtwoAlarmTargetTol", 1e-11, "theory/phase2_math_spec.md", "registered f64 recovery threshold", fmt=sci)
    def alarm_row(label, window, policy):
        return label, *(math(f"{alarms(arm, window, policy)}/{len(cohorts[arm][window])}") for arm in cohorts)

    rows = [
        ("True-origin recoveries", math(f"{b['a']['seeds_with_true_origin_recovery']}/{len(raw['seeds'])}"),
         r"\PtwoTerminalRecovered/\PtwoSeeds"),
        (r"Dirty DKD rewind damage at alarm ($\geq\PtwoDamageThreshold$ pp)",
         *(math(f"{b[arm]['material_damage_count']}/{len(cohorts[arm]['dirty'])}") for arm in cohorts)),
        alarm_row("Dirty DKD rewind alarms", "dirty", "rewind"),
        alarm_row("Clean DKD rewind alarms", "clean", "rewind"),
        alarm_row("Clean AdamW-skip alarms", "clean", "adam_skip"),
        alarm_row("Clean AdamW-continue alarms", "clean", "adam_continue"),
        ("Rewind minus AdamW-skip, immediate (pp)", math(decimal(b["a"]["rewind_minus_adam_skip_endpoint_pp"])),
         r"\PtwoTerminalAdamImmediate"),
        ("Rewind minus AdamW-skip, equal budget (pp)", math(decimal(b["a"]["rewind_minus_adam_skip_equal_budget_pp"])),
         r"\PtwoTerminalAdamBudget"),
        ("Pre-specified verdict", *("Pass" if b[arm]["pass"] else "Fail" for arm in ("a", "b"))),
        r"\midrule",
        r"\multicolumn{3}{@{}l@{}}{\textit{Exploratory control on terminal dirty windows}} \\",
        ("Correct over " + r"\PtwoTerminalTotal", "Immediate", "Equal budget"),
        ("Early-skip AdamW", r"\PtwoEarlySkipImmediateCorrect", r"\PtwoEarlySkipBudgetCorrect"),
        ("DKD rewind", r"\PtwoRewindImmediateCorrect", r"\PtwoRewindBudgetCorrect"),
        ("Rewind minus early skip (pp)", r"$+$\PtwoEarlyImmediateDelta", r"$+$\PtwoEarlyBudgetDelta"),
    ]
    build.add_table("recovery", table("@{}lrr@{}", ("Outcome", "Ramped", "Terminal"), rows),
                    [AUDIT2 + " backends.cuda.B and exploratory_B2b.effects.dirty", run + " B[*].{a,b}.{dirty,clean}.policies"])

    names = {"sgdm": "Momentum SGD", "adamw": "AdamW", "ewc": "EWC", "glide": "unit mass", "mass": "Fisher mass", "low": "low Fisher"}
    c_rows = []
    for arm, label in (("mass", "Fisher mass"), ("generic", "GENERIC"), ("high", "High Fisher")):
        for comp, contrast in c[arm]["contrasts"].items():
            c_rows.append((label, names[comp], math(decimal(contrast["mean_pp"])),
                           math(f"{contrast['wins']}/{contrast['non_ties']}"), math(pvalue(contrast["p"]))))
    check(len(c_rows) == n_contrasts, "Phase 2 contrast roster changed")
    build.add_table("retention", table("@{}llrrr@{}", ("Method", "Comparator", r"$\Delta\mathrm{BWT}$ (pp)", "Wins / non-ties", r"One-sided $p$"), c_rows),
                    [AUDIT2 + " backends.cuda.C.{mass,generic,high}.contrasts"])


def search_summary(rows):
    """Pool model-rule pairs and private draws, using exhaustion on failed draws."""
    exact_solved = exact_cases = sampled_solved = 0
    stop_times = []
    for run in rows:
        times = [next((leaf["elapsed"] for leaf in run["leaves"] if leaf["terminal_pass"][rule]), None)
                 for rule in range(run["D"] + 1)]
        exact_solved += sum(t is not None for t in times)
        exact_cases += len(times)
        for rule in run["private_draws"]:
            t = times[rule]
            sampled_solved += t is not None
            stop_times.append(run["elapsed"] if t is None else t)
    return (exact_solved, exact_cases), (sampled_solved, len(stop_times)), mean(stop_times)


def phase_three(build):
    test_key = f"{P3}/test.json"
    physics_key = f"{P3}/test_parts/physics.json"
    prefix_key = f"{P3}/prefix.json"
    test, report, prefix, audit = (load(key) for key in (test_key, physics_key, prefix_key, AUDIT3))
    physics = report["physics"]
    depth = max(map(int, test["decisions"]["depths"]))
    deep = test["decisions"]["depths"][str(depth)]
    peers = deep["methods"]
    # Sum byte-equal audited positions/momenta directly from each worker.
    counts = {}
    for name in ("native", "int32", "int64"):
        pairs = [bits for d in test["decisions"]["depths"]
                 for result in load(f"{P3}/test_parts/{name}_{d}.json")["cell"]["audits"]
                 for bits in result["bit_equal"]]
        counts[name] = sum(all(p) for p in pairs)
        if name != "native":
            check(counts[name] == len(pairs), f"inexact tree parent in {name}")
    check(counts == audit["P3_A"]["exact_tree_parents"] and counts["int32"] == counts["int64"],
          "Phase 3 exact tree parent counts differ")
    exact = counts["int32"]
    build.add("PthreeParents", exact, AUDIT3, "P3_A.exact_tree_parents by lattice (checked against worker bit_equal)")
    build.add("PthreeDepth", depth, test_key, "max(decisions.depths)")
    build.add("PthreeSeeds", len(test["seeds"]), test_key, "len(seeds)")
    build.add("PthreeWorkers", len(test["parts"]), test_key, "len(parts)")
    build.add("PthreeParams", report["parameters"], physics_key, "parameters", fmt=integer)
    build.add("PthreeHighStep", max(r["h"] for r in physics["native_runs"]), physics_key,
              "max(physics.native_runs[*].h)", fmt=fraction)
    high_step = max(r["h"] for r in physics["native_runs"])
    fit_upper = 0.1
    high_runs = [native for native in physics["native_runs"] if native["h"] == high_step]
    high_fits = [[sample["T"] for sample in native["samples"] if 0 < sample["relative_error"] < fit_upper]
                 for native in high_runs]
    check(all(steps == high_fits[0] for steps in high_fits), "Phase 3 high-step fits use different grid points")
    check(all(len(grid) == native["fit"]["usable"] for grid, native in zip(high_fits, high_runs)),
          "Phase 3 high-step usable counts disagree with the time grid")
    build.add("PthreeHighFitN", len(high_fits[0]), physics_key, "physics.native_runs high step fit.usable")
    build.add("PthreeHighFitSteps", r",\,".join(map(str, high_fits[0])), physics_key,
              "physics.native_runs high step samples[*].T with 0<relative_error<0.1")
    build.add("PthreeFitErrorUpper", fit_upper, "theory/phase3_math_spec.md", "registered fit sample relative error ceiling")
    final_horizon = max(native["samples"][-1]["T"] for native in physics["native_runs"])
    build.add("PthreeRateHorizon", final_horizon, physics_key, "max(physics.native_runs[*].samples[-1].T)")
    build.add("PthreeNativeThreshold", 1e-5, "theory/phase3_math_spec.md", "registered relative f32 error tolerance", fmt=sci)
    monitors = (m for depth_row in test["decisions"]["depths"].values() for m in depth_row["monitors"].values())
    gated_fields = (("sampled", "max_orientation_free"), ("sampled", "first_leaf_max"),
                    ("exact_prior", "max_orientation_free"), ("conditional_on_rank", "max_orientation_free"))
    max_gate = max(m[key][field] for m in monitors for key, field in gated_fields)
    build.add("PthreeAuc", max_gate, test_key, "max gated AUC across all depths and formats, excluding visit rank alone",
              fmt=lambda x: decimal(x, 3))
    build.add("PthreeNativeRatio", physics["speed"]["ratio"]["native"], physics_key,
              "physics.speed.ratio.native", fmt=lambda x: decimal(x, 3))
    build.add("PthreeSgdRoundTrip", 1000 * median(sum(v) for v in physics["copy"]["transfers"]["sgd"]["D2H_H2D_seconds"]),
              physics_key, "1000*median(physics.copy.transfers.sgd.D2H_H2D_seconds[*] summed)", fmt=decimal)
    build.add("PthreeIntLowRewind", 1000 * median(physics["copy"]["reverse_seconds"]["int32"]["1"]),
              physics_key, "1000*median(physics.copy.reverse_seconds.int32.1)", fmt=decimal)
    build.add("PthreeSingleSolved", round(peers["single"]["exact_prior"] * (depth + 1)), test_key,
              "D8 single exact prior times rule count (per public seed)")
    build.add("PthreeDfsSolved", round(peers["int32"]["exact_prior"] * (depth + 1)), test_key,
              "D8 int32 exact prior times rule count (per public seed)")
    long_runs = [row["last_completed_prefix"] for row in prefix["formats"].values()]
    check(len({r["seed"] for r in long_runs}) == len({r["h"] for r in long_runs}) == 1,
          "Phase 3 long-tape seed or step size differs by integer format")
    build.add("PthreeLongSeed", long_runs[0]["seed"], prefix_key, "formats.*.last_completed_prefix.seed")
    build.add("PthreeLongStep", long_runs[0]["h"], prefix_key, "formats.*.last_completed_prefix.h", fmt=fraction)
    for name, tag in (("int32", "IntLow"), ("int64", "IntHigh")):
        details = prefix["formats"][name]
        prefix_run = details["last_completed_prefix"]
        pos = f"formats.{name}."
        build.add(f"Pthree{tag}Guard", details["first_rejected_step"], prefix_key, pos + "first_rejected_step")
        build.add(f"Pthree{tag}Prefix", details["prefix_steps"], prefix_key, pos + "prefix_steps")
        build.add(f"Pthree{tag}Kick", prefix_run["forward_rounding"]["first_whole_shear_lost"][0][2],
                  prefix_key, pos + "last_completed_prefix.forward_rounding.first_whole_shear_lost[0][2]")
        shifts = prefix["chosen"]["shifts"][name]
        check(shifts == test["chosen"]["shifts"][name], f"Phase 3 {name} shifts differ from the lock")
        build.add(f"Pthree{tag}Shifts", shifts, prefix_key, f"chosen.shifts.{name}", fmt=lambda pair: f"({pair[0]},{pair[1]})")
    for name, tag in (("int32", "IntLow"), ("sgd_gpu", "GpuSgd"), ("sgd_cpu", "PinnedSgd")):
        build.add(f"Pthree{tag}MiB", peers[name]["peak_gpu"] / 2**20, test_key,
                  f"decisions.depths.{depth}.methods.{name}.peak_gpu / 2^20", fmt=lambda x: decimal(x, 1))
        build.add(f"Pthree{tag}Seconds", peers[name]["median_exhaust_s"], test_key,
                  f"decisions.depths.{depth}.methods.{name}.median_exhaust_s", fmt=lambda x: decimal(x, 3))
    build.add("PthreePinnedSpeedup", peers["int32"]["median_exhaust_s"] / peers["sgd_cpu"]["median_exhaust_s"],
              test_key, "D8 int32 / pinned SGD median exhaustive seconds", fmt=lambda x: decimal(x, 2))
    native_key = f"{P3}/test_parts/native_{depth}.json"
    sgd_key = f"{P3}/test_parts/sgd_gpu_{depth}.json"
    adam_key = f"{P3}/test_parts/adam_gpu_{depth}.json"
    native_run = load(native_key)["cell"]["rows"][0]
    sgd_run = load(sgd_key)["cell"]["rows"][0]
    adam_run = load(adam_key)["cell"]["rows"][0]
    steps = native_run["K"]
    id_bytes, id_rem = divmod(native_run["peak_tape_payload"], steps * depth)
    scalar_bytes, scalar_rem = divmod(sgd_run["active_state_bytes"], 2 * sgd_run["n"])
    check(id_rem == scalar_rem == 0 and sgd_run["n"] == adam_run["n"], "Phase 3 state size accounting mismatch")
    counter_bytes = adam_run["active_state_bytes"] - 3 * adam_run["n"] * scalar_bytes
    check(counter_bytes > 0, "Phase 3 Adam checkpoint counter is missing")
    add_many(build, native_key, (
        ("PthreeStepsPerEdge", steps, "cell.rows[0].K", str),
        ("PthreeEventIDBytes", id_bytes, "cell.rows[0].peak_tape_payload/(K*D)", str),
        ("PthreeSingleForkTapeBytes", id_bytes * steps, "cell.rows[0].peak_tape_payload/D", str),
        ("PthreeDrawsPerSeed", len(native_run["private_draws"]), "len(cell.rows[0].private_draws)", str),
    ))
    build.add("PthreeAdamCounterBytes", counter_bytes, adam_key,
              f"cell.rows[0].active_state_bytes minus three n-scalar buffers; scalar width from {sgd_key}")
    build.add("PthreeIntHighFirstSgdWin", min(test["decisions"]["P3_C"]["depths_with_lower_peak_allocated"]["int64_vs_sgd_gpu"]),
              test_key, "min(decisions.P3_C.depths_with_lower_peak_allocated.int64_vs_sgd_gpu)")
    head = audit["head_only_control"]
    head_exact, head_sampled, head_stop = search_summary(load(HEAD)["cell"]["rows"])
    check(head_exact == (sum(head["exact_solved_out_of_9"]), (depth + 1) * len(head["exact_solved_out_of_9"])) and
          head_sampled[0] == sum(head["sampled_solved_out_of_128"]), "Phase 3 head-only coverage differs from the audit")
    add_many(build, AUDIT3, (
        ("PthreeHeadParams", head["trainable_parameters"], "head_only_control.trainable_parameters", integer),
        ("PthreeHeadSolved", sum(head["exact_solved_out_of_9"]), "sum(head_only_control.exact_solved_out_of_9)", str),
        ("PthreeHeadCases", (depth + 1) * len(head["exact_solved_out_of_9"]), "head_only_control cases across public seeds", str),
        ("PthreeHeadMiB", head["peak_allocated_gpu_mib"], "head_only_control.peak_allocated_gpu_mib", lambda x: decimal(x, 1)),
        ("PthreeHeadKiB", head["live_checkpoint_bytes"] // 1024, "head_only_control.live_checkpoint_bytes / 1024", str),
        ("PthreeHeadSeconds", head["median_exhaust_s"], "head_only_control.median_exhaust_s", lambda x: decimal(x, 3)),
    ))
    build.add("PthreeHeadSampledSolved", head_sampled[0], HEAD, "cell.rows[*].private_draws covered by first passing leaf")
    build.add("PthreeHeadSampledCases", head_sampled[1], HEAD, "sum(len(cell.rows[*].private_draws))")
    build.add("PthreeHeadStopSeconds", head_stop, HEAD,
              "mean first passing leaf elapsed across all private_draws, or run elapsed if none", fmt=lambda x: decimal(x, 3))
    headers = ("Seed", r"$h$", "First miss", r"Fit $n$", rf"$G_{{{final_horizon}}}/({final_horizon}h)$", r"Exp. $R^2$", r"Power $R^2$")
    rows = []
    for native in physics["native_runs"]:
        fit = native["fit"]
        rows.append((math(native["seed"]), math(fraction(native["h"])), math(fit["first_grid_miss"]), math(fit["usable"]),
                     math(decimal(native["samples"][-1]["finite_time_rate"], 3)),
                     math(decimal(fit["r2_exponential"], 3)), math(decimal(fit["r2_power"], 3))))
    build.add_table("transformer", table("@{}rrrrrrr@{}", headers, rows),
                    [physics_key + " physics.native_runs[*].{fit,samples[-1]}"])

    rows = []
    cohort_summaries = {}
    for d, result in sorted(test["decisions"]["depths"].items(), key=lambda item: int(item[0])):
        methods, monitors = result["methods"], result["monitors"].values()
        summaries = {name: search_summary(load(f"{P3}/test_parts/{name}_{d}.json")["cell"]["rows"]) for name in methods}
        cohort_summaries[d] = summaries
        for name, (exact, sampled, stop) in summaries.items():
            check(abs(exact[0] / exact[1] - methods[name]["exact_prior"]) < 1e-12 and
                  abs(sampled[0] / sampled[1] - methods[name]["sampled"]) < 1e-12 and
                  abs(stop - methods[name]["mean_time_to_stop"]) < 1e-12,
                  f"Phase 3 {name} coverage or stopping time disagrees with the manifest at depth {d}")
        coverage = [summaries[name][:2]
                    for name in ("native", "int32", "int64")]
        check(all(pair == coverage[0] for pair in coverage), f"unequal DKD coverages at depth {d}")
        exact, sampled = coverage[0]
        exact_single, sampled_single, _ = summaries["single"]
        gated = max(m[bank][field] for m in monitors for bank, field in gated_fields)
        rank = max(m["conditional_on_rank"]["visit_rank_alone_auc"] for m in monitors)
        rows.append((math(d), *(math(f"{n}/{total}") for n, total in (exact, exact_single, sampled, sampled_single)),
                     math(decimal(gated, 3)), math(decimal(rank, 3))))
    headers = ("Depth", "DFS exact", "Single exact", "DFS sampled", "Single sampled", "Gated AUC", "Rank AUC")
    build.add_table("branching", table("@{}rrrrrrr@{}", headers, rows),
                    [test_key + " decisions.depths[*].{methods,monitors}",
                     f"{P3}/test_parts/{{native,int32,int64,single}}_*.json cell.rows[*].{{leaves,private_draws}}"])

    single_key = f"{P3}/test_parts/single_{depth}.json"
    exact_single, sampled_single, _ = cohort_summaries[str(depth)]["single"]
    for cohort, (solved, cases) in (("Exact", exact_single), ("Sampled", sampled_single)):
        build.add(f"PthreeSingle{cohort}Solved", solved, single_key,
                  f"covered {cohort.lower()} cases from cell.rows[*].leaves terminal_pass")
        build.add(f"PthreeSingle{cohort}Cases", cases, single_key,
                  f"total {cohort.lower()} cases from cell.rows[*].D or private_draws")

    contrasts = [("native", "Native DKD"), ("int32", "Int32 DKD"), ("int64", "Int64 DKD"),
                 ("sgd_gpu", "Momentum SGD, GPU"), ("adam_gpu", "AdamW, GPU"),
                 ("sgd_cpu", "Momentum SGD, pinned"), ("adam_cpu", "AdamW, pinned"), ("single", "Single path")]
    cells = audit["P3_C"]["cells"]
    rows = []
    for name, label in contrasts:
        peer = peers[name]
        fallback = cells[f"{name}_{depth}"]
        exact, sampled, stop = cohort_summaries[str(depth)][name]
        rows.append((label, *(math(f"{n}/{total}") for n, total in (exact, sampled)),
                     math(decimal(peer["peak_gpu"] / 2**20, 1)),
                     math(decimal(fallback["os_lifetime_peak_host_rss"] / 2**30, 2)),
                     math(decimal(peer["median_exhaust_s"], 3)), math(decimal(stop, 3))))
    rows += [r"\midrule", r"\multicolumn{7}{@{}l@{}}{\textit{Exploratory head-only control on reused seeds}} \\",
             ("Momentum SGD, GPU", r"\PthreeHeadSolved/\PthreeHeadCases", r"\PthreeHeadSampledSolved/\PthreeHeadSampledCases",
              r"\PthreeHeadMiB", math(decimal(head["os_peak_host_gib"], 2)), r"\PthreeHeadSeconds", r"\PthreeHeadStopSeconds")]
    labels = ("Method", "Exact", "Sampled", "GPU MiB", "Host GiB", "Exhaust s", "Stop s")
    build.add_table("systems", table("@{}lrrrrrr@{}", labels, rows),
                    [test_key + f" decisions.depths.{depth}.methods", AUDIT3 + " P3_C.cells and head_only_control",
                     f"{P3}/test_parts/*_{depth}.json and {HEAD} cell.rows[*].{{leaves,private_draws,elapsed}}"])


def provenance(build):
    p1 = load(f"{P1}/continual.json")
    p2 = load(f"{P2}/test_cuda.json")
    p3 = load(f"{P3}/test.json")
    entries = (("PoneEvalHash", p1["eval_sha256"], f"{P1}/continual.json", "eval_sha256"),
               ("PtwoEvalHash", p2["eval_v2_sha256"], f"{P2}/test_cuda.json", "eval_v2_sha256"),
               ("PthreeEvalHash", p3["eval_sha256"], f"{P3}/test.json", "eval_sha256"),
               ("PoneAdapterHash", p1["factory_module_sha256"], f"{P1}/continual.json", "factory_module_sha256"),
               ("PtwoAdapterHash", p2["adapter_python_tree_sha256"], f"{P2}/test_cuda.json", "adapter_python_tree_sha256"),
               ("PthreeAdapterHash", p3["adapter_tree_sha256"], f"{P3}/test.json", "adapter_tree_sha256"),
               ("CudaHardware", p3["hardware"], f"{P3}/test.json", "hardware"),
               ("TorchVersion", p3["torch"], f"{P3}/test.json", "torch"),
               ("PythonVersion", p3["python"], f"{P3}/test.json", "python"),
               ("CudaBuild", p3["cuda_build"], f"{P3}/test.json", "cuda_build"))
    for name, value, path, selector in entries:
        build.add(name, value, path, selector, text=True)
    paths = (
        ("PoneAnalyticHash", f"{P1}/analytic.json"), ("PoneContinualHash", f"{P1}/continual.json"),
        ("PonePhysicsHash", f"{P1}/physics.json"), ("PoneAuditHash", AUDIT1),
        ("PtwoValidationHash", f"{P2}/validation.json"), ("PtwoTestCudaHash", f"{P2}/test_cuda.json"),
        ("PtwoTestCpuHash", f"{P2}/test_cpu.json"), ("PtwoPhysicsCpuHash", f"{P2}/physics_cpu.json"),
        ("PtwoDenseCudaHash", f"{P2}/dense_cuda.json"), ("PtwoDenseCpuHash", f"{P2}/dense_cpu.json"),
        ("PtwoAuditHash", AUDIT2), ("PthreeValidationHash", f"{P3}/validation.json"),
        ("PthreeSmokeHash", f"{P3}/smoke.json"), ("PthreeTestHash", f"{P3}/test.json"),
        ("PthreePhysicsHash", f"{P3}/test_parts/physics.json"), ("PthreePrefixHash", f"{P3}/prefix.json"),
        ("PthreeAuditHash", AUDIT3), ("PthreeHeadHash", HEAD),
    )
    for name, path in paths:
        build.add(name, pinned_digest(path), path, "verified raw byte SHA-256", text=True)
    build.add("PthreeMathHash", load(AUDIT3)["evaluator_and_spec_pins"]["theory/phase3_math_spec.md"],
              AUDIT3, "evaluator_and_spec_pins.theory/phase3_math_spec.md", text=True)
    build.add("PtwoMathHash", load(AUDIT2)["math_spec_sha256"], AUDIT2, "math_spec_sha256", text=True)
    build.add("PoneUnitCount", len(load(AUDIT1)["unit_tests_passed"]), AUDIT1, "len(unit_tests_passed)")
    build.add("PtwoUnitCount", len(load(AUDIT2)["unit_tests_passed"]), AUDIT2, "len(unit_tests_passed)")
    build.add("PthreeUnitCount", len(load(AUDIT3)["unit_methods_passed"]), AUDIT3, "len(unit_methods_passed)")


def main():
    verify()
    build = Build()
    for stage in (phase_one, phase_two, phase_three, provenance):
        stage(build)
    build.write()
    print(f"Verified {len(PINS)} source artifacts. Wrote {len(build.macros)} macros and {len(build.outputs) - 1} tables.")


if __name__ == "__main__":
    main()
