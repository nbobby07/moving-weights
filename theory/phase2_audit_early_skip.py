"""Post-hoc B2b policy audit. From the repo root, set PYTHONPATH=src and use -B -m."""
import argparse
import hashlib
import json
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

from eval.bench_continual import logits
from eval_v2.protocol import (ROOT, TEST, ForceCounter, accuracy, batches, check, data,
                              episode_tape, make, prepare, tree_hash)
from mttt.phase2 import make_optimizer


SUBMISSION = ROOT.parent / "artifacts/sol/p2/sol-20261122-cpu-lock"
TEST_SHA = {"cpu": "ac8f6352a54d718fbd745b8de50dc08b40d4356614566f1e3779c3cc95ab6126",
            "cuda": "b946d42816f0ca600fe0f6bcbff9db20354a3cfc7a6f97b366d4dacd86c89a7d"}


def early_skip(q, tape, tests, continuation, cfg):
    opt = make(make_optimizer, "adamw", cfg, q)
    state = opt.init(q, torch.zeros_like(q))
    meter = ForceCounter()
    disagreements = []
    for event in tape.arrivals:
        x, observed = tape.fetch(event)
        wrong = int((logits(q, x).argmax(-1) != observed).sum())
        disagreements.append(wrong)
        if 2 * wrong > len(observed):
            break
        q, state = meter.step(opt, q, state, (x, observed))

    used = meter.calls
    endpoint = round(300 * accuracy(q, tests[0]))
    live = round(300 * accuracy(q, tests[1]))
    for batch in continuation[:64 - used]:
        q, state = meter.step(opt, q, state, batch)
    check(meter.calls == 64, "gradient budget mismatch")
    return {"first_warning_after_updates": used if used < len(tape.arrivals) else None,
            "preupdate_disagreements_out_of_32": disagreements, "monitor_forwards": len(disagreements),
            "window_gradients": used, "clean_repair_gradients": 64 - used, "total_gradients": meter.calls,
            "endpoint_task1_correct": endpoint, "endpoint_task2_correct": live,
            "equal_budget_task1_correct": round(300 * accuracy(q, tests[0]))}


def measure(backend):
    payload = (SUBMISSION / f"test_{backend}.json").read_bytes()
    check(hashlib.sha256(payload).hexdigest() == TEST_SHA[backend], "unexpected submitted test artifact")
    test = json.loads(payload)
    check(tree_hash(ROOT) == test["eval_v2_sha256"], "Phase 2 source mismatch")
    check(tree_hash(ROOT.parent / "eval") == test["eval_v1_sha256"], "Phase 1 source mismatch")
    check(tree_hash(ROOT.parent / "src/mttt", python_only=True) == test["adapter_python_tree_sha256"],
          "adapter source mismatch")
    device = torch.device(backend)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(11173)
    cfg = {"h": test["chosen"]["B"]["adamw"]}
    rows = []
    for seed in TEST:
        q, _, tests, *_ = prepare(seed, device)
        continuation = batches(data(seed * 100 + 62, 0, 384, device), 96, seed * 100 + 63)
        for tag in ("clean", "dirty"):
            tape = episode_tape(seed, "b", tag == "dirty", device)
            rows.append({"seed": seed, "window": tag,
                         **early_skip(q.clone(), tape, tests, continuation, cfg)})
    return {"device": backend, "test_sha256": TEST_SHA[backend], "cfg": cfg,
            "scope": "Post-hoc early-skip sensitivity on previously scored Phase 2 B2b seeds", "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    print(json.dumps(measure(parser.parse_args().device), allow_nan=False))
