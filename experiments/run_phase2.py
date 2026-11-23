"""Run Phase 2 through physics and CPU selection, then score a recorded lock."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from .run_phase1 import ROOT, new_run, run_command, write_json


PIN1 = "a2c2f60d9fe7508f0f5c62701016d488b365d8d68710e72bac41cec7ebdcc89d"
PIN2 = "19aef02de8768e14dec3f1557130b838f217e0684c6acd8f926e6a87f20270f0"
SPEC = "f0d22f7eff029c9aeb14d97c752c9d908534b545630d82bda877382dd5442721"
ADAPTER = "mttt.phase2:make_optimizer"
OUTPUTS = ROOT / "artifacts" / "sol" / "p2"
HANDOFF = ROOT / "comms" / "sol_to_astra_p2.md"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def pins():
    if not sys.dont_write_bytecode:
        raise RuntimeError("launch with python -B to protect the frozen trees")
    from eval_v2.protocol import tree_hash

    expected = {"eval": PIN1, "eval_v2": PIN2, "theory/phase2_math_spec.md": SPEC}
    for file, pinned in expected.items():
        path = ROOT / file
        actual = tree_hash(path) if path.is_dir() else digest(path)
        if actual != pinned:
            raise ValueError(f"integrity failure for {file}: expected {pinned}, got {actual}")
    return expected


def run_checked(logs, name, args, *, artifact=None):
    before = after = None
    failure = None
    try:
        before = pins()
        try:
            return run_command(logs, name, args, artifact=artifact)
        finally:
            after = pins()
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        write_json(logs / f"{name}_integrity.json", {"before": before, "after": after, "error": failure,
                   "artifact_sha256": digest(artifact) if artifact is not None and artifact.is_file() else None})


def benchmark(logs, out, stage, device, stem, lock_args=()):
    destination = out / f"{stem}.json"
    return run_checked(logs, stem, ["-m", "eval_v2.run", stage, "--factory", ADAPTER,
                       "--expected-root", PIN2, "--device", device, "--output", str(destination), *lock_args],
                       artifact=destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "test"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock-sha")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != OUTPUTS.resolve():
        parser.error("--out must name a direct subdirectory of artifacts/sol/p2")
    logs = new_run(f"phase2-{args.stage}")
    print(f"Artifacts {out}\nLogs {logs}", flush=True)
    if args.stage == "prepare":
        out.mkdir(parents=True, exist_ok=False)
        if not run_checked(logs, "unit", ["-m", "unittest", "-v", "experiments.test_phase2"]):
            return 1
        if not benchmark(logs, out, "physics", "cpu", "physics_cpu"):
            return 1
        if not benchmark(logs, out, "validate", "cpu", "validation"):
            return 1
        validation = out / "validation.json"
        record = json.loads(validation.read_text(encoding="utf-8"))
        receipt = {"validation": str(validation.relative_to(ROOT)), "validation_sha256": digest(validation),
                   "adapter_python_tree_sha256": record["adapter_python_tree_sha256"], "chosen": record["chosen"]}
        write_json(logs / "receipt.json", receipt)
        print(json.dumps(receipt, indent=2), flush=True)
        return 0

    if not args.lock_sha or not HANDOFF.is_file() or args.lock_sha not in HANDOFF.read_text(encoding="utf-8"):
        parser.error("record the validation SHA-256 in the handoff before testing")
    options = ["--lock", str(out / "validation.json"), "--lock-sha", args.lock_sha]
    cpu = benchmark(logs, out, "test", "cpu", "test_cpu", options)
    cuda = benchmark(logs, out, "test", "cuda", "test_cuda", options)
    return 0 if cpu and cuda else 1


if __name__ == "__main__":
    raise SystemExit(main())
