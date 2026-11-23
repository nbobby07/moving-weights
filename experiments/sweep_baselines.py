"""Tune continual-learning baselines on disjoint seeds and score the frozen seeds."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
from statistics import mean, stdev
import sys
import traceback

import torch

from eval.bench_continual import STEPS, accuracy, gradient, prepare
from eval.common import H, SEEDS, tick, verify_hash
from mttt.baselines import AdamW, EWC, SGDM

from .run_phase1 import FROZEN, ROOT, new_run, write_json


VALIDATION_SEEDS = (2027, 4099, 6029)
LR_MULTIPLIERS = (0.5, 1.0, 2.0)
EWC_STRENGTHS = (0.0, 10.0, 100.0, 1000.0)
LIVE_TOLERANCE = 0.01
METRICS = ("task1_bwt", "live_acc", "arrival_to_live_gain", "ms_per_step")


@dataclass(frozen=True)
class Config:
    method: str
    h: float
    strength: float = 0.0

    @property
    def key(self):
        return f"{self.method}-h{self.h:g}-lambda{self.strength:g}"

    def optimizer(self, q, fisher):
        if self.method == "ewc":
            return EWC(self.h, fisher=fisher, anchor=q, strength=self.strength)
        return (AdamW if self.method == "adamw" else SGDM)(self.h)


def finite(q, state):
    return bool(torch.stack([torch.isfinite(v).all() for v in (q, *state)
                             if isinstance(v, torch.Tensor)]).all())


def run_trial(path, seed, split, config, prepared):
    row = {"seed": seed, "split": split, "config": asdict(config), "status": "started"}
    write_json(path, row)
    task, step = None, None
    try:
        q, fisher, tests, streams, *_ = prepared
        opt = config.optimizer(q, fisher)
        state = opt.init(q, torch.zeros_like(q))
        if not finite(q, state):
            raise FloatingPointError("initial parameters or state are nonfinite")
        matrix = [[accuracy(q, batch) for batch in tests]]
        seconds, curves = [], []
        for task, stream in enumerate(streams, start=2):
            curve = [matrix[-1][task - 1]]
            elapsed = 0.0
            for offset in range(0, STEPS, 12):
                block = stream[offset:offset + 12]
                before_q, before_state = q, state
                start = tick(q.device)
                for step, batch in enumerate(block, start=offset + 1):
                    q, state = opt.step(q, state, gradient(batch), gamma=0.0)
                elapsed += tick(q.device) - start
                # Keep validity checks out of wall-clock timings to match the locked protocol.
                if not finite(q, state):
                    for step, batch in enumerate(block, start=offset + 1):
                        before_q, before_state = opt.step(before_q, before_state, gradient(batch), gamma=0.0)
                        if not finite(before_q, before_state):
                            row["first_nonfinite_step"] = step
                            break
                    raise FloatingPointError("nonfinite parameters or state at block end")
                curve.append(accuracy(q, tests[task - 1]))
            seconds.append(elapsed)
            curves.append(curve)
            matrix.append([accuracy(q, batch) for batch in tests])
        row.update(status="passed", test_accuracy_matrix=matrix,
                   arrival_to_live_12_step_curves=curves, optimizer_wall_seconds_by_task=seconds,
                   task1_bwt=matrix[-1][0] - matrix[0][0], live_acc=mean(c[-1] for c in curves),
                   arrival_to_live_gain=mean(c[-1] - c[0] for c in curves),
                   ms_per_step=1000 * sum(seconds) / (len(streams) * STEPS))
    except (Exception, KeyboardInterrupt) as error:
        row.update(status="failed", failed_task=task, failed_step_or_block_end=step,
                   error=repr(error), traceback=traceback.format_exc())
        if isinstance(error, KeyboardInterrupt):
            raise
    finally:
        write_json(path, row)
    return row


def summarize(rows):
    clean = [row for row in rows if row["status"] == "passed"]
    metrics = {}
    for name in METRICS:
        values = [row[name] for row in clean]
        metrics[name] = {"mean": mean(values) if values else None,
                         "sample_std": stdev(values) if len(values) >= 2 else None}
    return {"passed": len(clean), "failed": len(rows) - len(clean), **metrics}


def run_split(directory, name, seeds, configs, device):
    results = {config: [] for config in configs}
    for i, seed in enumerate(seeds):
        print(f"{name} seed {seed}", flush=True)
        try:
            prepared = prepare(seed, device)
        except Exception as error:
            for config in configs:
                row = {"seed": seed, "split": name, "config": asdict(config), "status": "failed",
                       "error": repr(error), "traceback": traceback.format_exc()}
                write_json(directory / f"{name}-{seed}-{config.key}.json", row)
                results[config].append(row)
            continue
        # Rotate the order to distribute startup and timing effects across configurations.
        start = i % len(configs)
        for config in configs[start:] + configs[:start]:
            path = directory / f"{name}-{seed}-{config.key}.json"
            results[config].append(run_trial(path, seed, name, config, prepared))
    return {config: summarize(rows) for config, rows in results.items()}


def select(validation, live_floor):
    selected = {}
    for method in ("sgdm", "adamw", "ewc"):
        eligible = [config for config, stats in validation.items()
                    if config.method == method and stats["failed"] == 0
                    and (method != "ewc" or config.strength > 0)
                    and stats["live_acc"]["mean"] >= live_floor]
        if eligible:
            selected[method] = max(eligible, key=lambda c: (validation[c]["task1_bwt"]["mean"],
                                                            validation[c]["live_acc"]["mean"]))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    directory = new_run(f"baseline-sweep-{args.device}")
    print(directory.relative_to(ROOT), flush=True)
    nominal = (Config("sgdm", H["sgdm"]), Config("adamw", H["adamw"]))
    configs = tuple(Config(method, H[method] * multiplier) for method in ("sgdm", "adamw")
                    for multiplier in LR_MULTIPLIERS) + tuple(
        Config("ewc", H["sgdm"] * multiplier, strength) for multiplier in LR_MULTIPLIERS
        for strength in EWC_STRENGTHS)
    path = directory / "sweep.json"
    baselines = ROOT / "src" / "mttt" / "baselines.py"
    report = {"status": "started",
              "command": [sys.executable, "-B", "-m", "experiments.sweep_baselines", *sys.argv[1:]],
              "torch": str(torch.__version__), "python": sys.version.split()[0],
              "device": args.device, "cuda_build": torch.version.cuda, "dtype": "float64",
              "eval_sha256": FROZEN, "deterministic": True, "threads": 1,
              "source_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                                for source in (Path(__file__), baselines)},
              "validation_seeds": VALIDATION_SEEDS, "heldout_seeds": SEEDS,
              "grid": [asdict(config) for config in configs], "live_tolerance": LIVE_TOLERANCE,
              "selection_rule": "Require all validation runs to be finite and mean live accuracy within "
                                "1pp of the lower nominal SGDM/AdamW mean. Select max mean Task 1 BWT, "
                                "breaking ties by mean live accuracy, then grid order. Require positive EWC "
                                "strength; zero is a matched SGD control."}
    write_json(path, report)
    try:
        assert set(SEEDS).isdisjoint(VALIDATION_SEEDS)
        verify_hash(FROZEN)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(VALIDATION_SEEDS[0])
        device = torch.device(args.device)
        report["hardware"] = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        validation = run_split(directory, "validation", VALIDATION_SEEDS, configs, device)
        report["validation"] = {config.key: stats for config, stats in validation.items()}
        if any(validation[config]["failed"] for config in nominal):
            raise RuntimeError("nominal validation run failed so the live accuracy floor is undefined")
        live_floor = min(validation[c]["live_acc"]["mean"] for c in nominal) - LIVE_TOLERANCE
        selected = select(validation, live_floor)
        report.update(live_floor=live_floor, selected={method: asdict(c) for method, c in selected.items()})
        write_json(path, report)
        finalist_configs = tuple(dict.fromkeys((*nominal, *selected.values())))
        heldout = run_split(directory, "heldout", SEEDS, finalist_configs, device)
        report["heldout"] = {config.key: stats for config, stats in heldout.items()}
        verify_hash(FROZEN)
        incomplete = len(selected) != 3 or any(stats["failed"] for stats in heldout.values())
        report["status"] = "incomplete" if incomplete else "passed"
        return int(incomplete)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(path, report)


if __name__ == "__main__":
    raise SystemExit(main())
