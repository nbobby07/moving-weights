"""Replay each last safe Phase 3 lattice prefix with the frozen CUDA receipt."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
import sys

from .run_phase3 import ADAPTER, OUTPUTS, PIN3, pins


FORWARD = re.compile(r"\['long_forward',\s*(?P<seed>\d+),\s*(?P<step>\d+)\]")
DETAIL = re.compile(
    r"(?P<shear>drift1|kick|drift2) index=(?P<coordinate>\d+) "
    r"increment=(?P<increment>\S+) attempted=(?P<attempted>\S+) limit=(?P<limit>\d+)"
)


def payload(run, name):
    path = (run / name).resolve()
    if not path.is_relative_to(run):
        raise ValueError(f"{name} points outside the locked run")
    blob = path.read_bytes()
    return blob, {"path": name, "sha256": sha256(blob).hexdigest()}


def require_fields(report, expected, name):
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{name} has inconsistent {key}")


def audit_prefix(factory, model, method, locked_row, cfg, seed):
    import torch
    from eval_v3.physics import trajectory

    fault = locked_row.get("failure")
    if fault is None:
        return {"status": "frozen_horizon_completed", "frozen_horizon": locked_row}
    match = FORWARD.search(fault)
    details = DETAIL.search(fault)
    result = {"status": "forward_fault_unresolved", "first_rejected": fault}
    if not match or not details or int(match["seed"]) != seed or not 1 <= int(match["step"]) <= 500:
        return result
    step = int(match["step"])
    h, shifts = cfg["rates"]["native"], cfg["shifts"][method]
    result.update(first_rejected_step=step, prefix_steps=step - 1, shear=details["shear"],
                  coordinate=int(details["coordinate"]), increment=details["increment"],
                  attempted=details["attempted"], limit=int(details["limit"]))
    try:
        if step == 1:
            # A rejection on the first shear leaves an empty replay interval.
            q = model.initial(seed)
            state = factory(method, h=h, shifts=shifts).init(q, q.sign() * 2**-14)
            restored = (state[0], -(-state[1]))
            q_l2, p_l2 = [torch.linalg.vector_norm(x.double() - y.double()).item() / 2**s
                          for x, y, s in zip(state, restored, shifts)]
            exact = all(torch.equal(x, y) for x, y in zip(state, restored))
            row = {"seed": seed, "method": method, "h": h, "forward_rounding": {},
                   "samples": [{"T": 0, "q_l2": q_l2, "p_l2": p_l2, "exact": exact}]}
        else:
            row = trajectory(factory, model, seed, h, shifts, method, (step - 1,))
    except (ArithmeticError, torch.OutOfMemoryError) as exc:
        return {**result, "status": "prefix_failed", "prefix_failure": repr(exc)}
    sample = row["samples"][0]
    exact = sample["exact"] and sample["q_l2"] == sample["p_l2"] == 0
    return {**result, "status": "prefix_exact" if exact else "prefix_inexact", "last_completed_prefix": row}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--lock-sha", required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = run / "prefix.json"
    if run.parent != OUTPUTS or not run.is_dir() or output.exists():
        parser.error("choose a locked run directly under artifacts/sol/p3 with a new prefix.json")
    frozen = pins()

    import torch
    import mttt
    from eval_v3.run import choose, configure, write
    from eval_v3.task import DEPTHS, METHODS, TEST, VAL, Transformer, tree_hash
    from mttt.phase3 import make_engine

    source = Path(mttt.__file__).resolve().parent
    roots = {"eval_sha256": PIN3, "factory": ADAPTER, "adapter_tree_sha256": tree_hash(source, True)}
    runtime = {"torch": str(torch.__version__), "python": sys.version.split()[0],
               "platform": platform.platform(), "cuda_build": torch.version.cuda}
    common = {**roots, **runtime}
    links = {}
    for name in ("smoke", "validation", "test"):
        blob, links[name] = payload(run, f"{name}.json")
        if name == "validation" and links[name]["sha256"] != args.lock_sha:
            raise ValueError("validation lock digest mismatch")
        report = json.loads(blob)
        if name == "smoke":
            require_fields(report, {**common, "stage": "smoke", "device": "cpu"}, name)
            if not report["physics"]["long_integer_pass"] or not report["full_size_cpu_probe"]["repeat"]["equal"]:
                raise ValueError("CPU mechanics smoke did not pass")
        elif name == "validation":
            require_fields(report, {**common, "stage": "validate", "device": "cuda", "seeds": list(VAL)}, name)
            cfg = choose(report["raw"])
            require_fields(report, {"chosen": cfg}, name)
        else:
            manifest = report
    locked = {**common, "device": "cuda", "seeds": list(TEST), "validation_sha256": args.lock_sha}
    require_fields(manifest, {**locked, "stage": "test", "chosen": cfg}, "test manifest")
    keys = {"physics", *(f"{method}_{depth}" for depth in DEPTHS for method in METHODS)}
    if set(manifest["parts"]) != keys:
        raise ValueError("test manifest is missing frozen workers")
    for key in sorted(keys):
        blob, link = payload(run, f"test_parts/{key}.json")
        if link != manifest["parts"][key]:
            raise ValueError(f"test manifest digest mismatch for {key}")
        if key == "physics":
            links[key], physics = link, json.loads(blob)
    device = configure("cuda")
    hardware = torch.cuda.get_device_name(device)
    require_fields(manifest, {"hardware": hardware}, "test manifest")
    require_fields(physics, {**locked, "stage": "physics", "hardware": hardware}, "physics worker")
    model = Transformer(device)
    require_fields(physics, {"parameters": model.n}, "physics worker")

    try:
        integer = physics["physics"]["long_integer"]
        rows = {m: audit_prefix(make_engine, model, m, integer[m], cfg, TEST[0]) for m in ("int32", "int64")}
    finally:
        pins()
        if tree_hash(source, True) != roots["adapter_tree_sha256"]:
            raise ValueError("adapter changed during prefix replay")
        for link in links.values():
            if payload(run, link["path"])[1] != link:
                raise ValueError(f"{link['path']} changed during prefix replay")
    report = {**locked, "stage": "supplemental_lattice_prefix", "hardware": hardware,
              "parameters": model.n, "pins": frozen, "inputs": links, "chosen": cfg, "formats": rows}
    print(f"prefix {output} {write(output, report)}", flush=True)
    return int(any(r["status"] not in ("frozen_horizon_completed", "prefix_exact") for r in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
