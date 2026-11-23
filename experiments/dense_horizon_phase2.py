"""Measure native f32 rollback prefixes through the first failure on any seed."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import sys

from .run_phase2 import ADAPTER, PIN1, PIN2, ROOT, pins


def prefixes(factory, seed, device, limit):
    import torch
    from eval_v2.protocol import ForceCounter, accuracy, l2, make, prepare

    pretrained, _, tests, streams, *_ = prepare(seed, device)
    q0 = pretrained.float()
    p0 = (0.02 * torch.cos(torch.arange(q0.numel(), device=device, dtype=torch.float64))).float()
    tape = [(x.float(), y) for stream in streams for x, y in stream]
    test = (tests[0][0].float(), tests[0][1])
    clean_count = round(300 * accuracy(q0, test))
    parameters = {"h": 0.12}
    forward = make(factory, "glide", parameters, q0)
    q, state = q0, forward.init(q0, p0)
    forward_counter, replay_counter = ForceCounter(), ForceCounter()

    for length in range(1, limit + 1):
        q, state = forward_counter.step(forward, q, state, tape[(length - 1) % len(tape)])
        back_q, back_state = q.clone(), (-state[0].clone(),)
        reverse = make(factory, "glide", parameters, back_q)
        before = replay_counter.calls
        for i in range(length - 1, -1, -1):
            back_q, back_state = replay_counter.step(reverse, back_q, back_state, tape[i % len(tape)])
        back_p = -back_state[0]
        q_error, p_error = l2(back_q, q0), l2(back_p, p0)
        back_correct = round(300 * accuracy(back_q, test))
        count_change = back_correct - clean_count
        finite = math.isfinite(q_error) and math.isfinite(p_error)
        yield {
            "seed": seed, "T": length, "q_l2": q_error if math.isfinite(q_error) else str(q_error),
            "p_l2": p_error if math.isfinite(p_error) else str(p_error),
            "starting_correct": clean_count, "recovered_correct": back_correct,
            "change_in_correct": count_change, "finite": finite,
            "within_tolerance": finite and max(q_error, p_error) < 1e-5 and abs(count_change) <= 1,
            "state_exact": torch.equal(back_q, q0) and torch.equal(back_p, p0),
            "forward_acc": accuracy(q, test), "forward_gradients": forward_counter.calls,
            "replay_gradients": replay_counter.calls - before,
        }


def measure(factory, device, seeds, limit, grid):
    iterators = [prefixes(factory, seed, device, limit) for seed in seeds]
    rows, disagreements = [], []
    first_failed = None
    for length in range(1, limit + 1):
        current = [next(trace) for trace in iterators]
        rows.extend(current)
        for row in current:
            frozen = grid.get((row["seed"], length))
            if frozen is None:
                continue
            for key in ("q_l2", "p_l2", "change_in_correct", "within_tolerance", "state_exact", "forward_acc"):
                if row[key] != frozen[key]:
                    disagreements.append({"seed": row["seed"], "T": length, "field": key,
                                          "dense": row[key], "frozen": frozen[key]})
        if not all(row["within_tolerance"] for row in current):
            first_failed = length
            break

    return {
        "first_failed_horizon": first_failed, "right_censored": first_failed is None,
        "largest_tested_horizon": length,
        "first_failed_by_seed": {row["seed"]: length if not row["within_tolerance"] else None for row in current},
        "test_grid_disagreements": disagreements, "total_forward_gradients": len(seeds) * length,
        "total_replay_gradients": sum(row["replay_gradients"] for row in rows), "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--lock-sha", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(ROOT / tree) for tree in ("eval", "eval_v2", "theory")):
        parser.error("choose a new output path outside the frozen trees")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    pins()

    from eval_v2.run import locked
    from eval_v2.protocol import HORIZONS, TEST, tree_hash
    from eval_v2.recovery import horizon_verdict
    import torch
    import mttt
    from mttt.phase2 import make_optimizer

    source = Path(mttt.__file__).resolve().parent
    roots = {"eval_v1_sha256": PIN1, "eval_v2_sha256": PIN2, "factory": ADAPTER,
             "adapter_python_tree_sha256": tree_hash(source, python_only=True)}
    lock = args.artifact_dir / "validation.json"
    chosen = locked(argparse.Namespace(lock=lock, lock_sha=args.lock_sha), roots)
    seeds = TEST[:3]
    tests, test_digests = {}, {}
    for backend in ("cpu", "cuda"):
        path = args.artifact_dir / f"test_{backend}.json"
        payload = path.read_bytes()
        report = json.loads(payload)
        expected = {**roots, "stage": "test", "device": backend, "seeds": list(TEST),
                    "validation_sha256": args.lock_sha, "chosen": chosen, "horizon_seeds": list(seeds)}
        for key, value in expected.items():
            if report[key] != value:
                raise ValueError(f"{path.name} disagrees with the validation receipt or current adapter on {key}")
        if len(report["horizons"]) != len(seeds) or report["decisions"]["horizons"] != horizon_verdict(report["horizons"]):
            raise ValueError(f"{path.name} has inconsistent horizon evidence")
        tests[backend] = report
        test_digests[path] = sha256(payload).hexdigest()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(11173)
    device = torch.device(args.device)
    metadata = {"device": device.type, "torch": str(torch.__version__), "python": sys.version.split()[0],
                "cuda_build": torch.version.cuda, "deterministic": True,
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()}
    frozen = tests[device.type]
    if any(frozen[key] != value for key, value in metadata.items()):
        raise ValueError("runtime differs from the locked test on this backend")
    grid = {(seed, row["T"]): row for seed, rows in zip(seeds, frozen["horizons"])
            for row in rows if row["method"] == "glide"}
    if set(grid) != {(seed, t) for seed in seeds for t in HORIZONS}:
        raise ValueError("locked test is missing native horizon rows")
    cap = frozen["decisions"]["horizons"]["glide"]["first_failed_grid_horizon"] or HORIZONS[-1]

    try:
        result = measure(make_optimizer, device, seeds, cap, grid)
    finally:
        pins()
        if tree_hash(source, python_only=True) != roots["adapter_python_tree_sha256"]:
            raise ValueError("adapter changed during dense measurement")
        for path, expected in {lock: args.lock_sha, **test_digests}.items():
            if sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"{path.name} changed during dense measurement")
    report = {**roots, **metadata, "stage": "supplemental_dense_horizon", "seeds": seeds,
              "validation_sha256": args.lock_sha, "chosen": chosen,
              "test_sha256": {path.name: digest for path, digest in test_digests.items()},
              "dtype": "float32", "h": 0.12, "absolute_l2_limit": 1e-5, "max_correct_count_change": 1,
              "locked_grid_cap": cap, **result}
    payload = (json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n").encode()
    with output.open("xb") as file:
        file.write(payload)
    print(device.type, output, sha256(payload).hexdigest(), flush=True)
    return int(bool(result["test_grid_disagreements"]))


if __name__ == "__main__":
    raise SystemExit(main())
