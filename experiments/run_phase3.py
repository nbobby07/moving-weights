"""Run Phase 3 smoke and CUDA validation, then evaluate the recorded lock."""

import argparse
import json
from pathlib import Path
import sys

from .run_phase1 import ROOT, new_run, run_command, write_json
from .run_phase2 import PIN1, PIN2, digest


PIN3 = "4cf1e4d5208bba25b42f2a34f92205044e3ccf4804b1512e2c49d5fc2903195d"
SPEC = "4e2c227be7ed734e3f6c7c5390494369e87e6d2293f60ce9f9fc547012fb518c"
ADAPTER = "mttt.phase3:make_engine"
OUTPUTS = ROOT / "artifacts" / "sol" / "p3"
HANDOFF = ROOT / "comms" / "sol_to_astra_p3.md"


def pins():
    if not sys.dont_write_bytecode:
        raise RuntimeError("launch with python -B to protect the frozen trees")
    from eval_v3.task import tree_hash

    expected = {"eval": PIN1, "eval_v2": PIN2, "eval_v3": PIN3, "theory/phase3_math_spec.md": SPEC}
    for file, pin in expected.items():
        path = ROOT / file
        actual = tree_hash(path) if path.is_dir() else digest(path)
        if actual != pin:
            raise ValueError(f"integrity failure for {file}: expected {pin}, got {actual}")
    return expected


def run_checked(logs, name, args, *, artifact=None):
    before = after = error = None
    try:
        before = pins()
        try:
            return run_command(logs, name, args, artifact=artifact)
        finally:
            after = pins()
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        write_json(logs / f"{name}_integrity.json", {"before": before, "after": after, "error": error,
                   "artifact_sha256": digest(artifact) if artifact is not None and artifact.is_file() else None})


def benchmark(logs, out, stage, device, filename, options=()):
    output = out / filename
    return run_checked(logs, stage, ["-m", "eval_v3.run", stage, "--device", device, "--factory", ADAPTER,
                                    "--expected-root", PIN3, "--output", str(output), *options], artifact=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "test"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock-sha")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != OUTPUTS.resolve():
        parser.error("--out must be a new or locked run directly under artifacts/sol/p3")
    if args.stage == "test" and (not args.lock_sha or not HANDOFF.is_file()
                                 or args.lock_sha not in HANDOFF.read_text(encoding="utf-8")):
        parser.error("record the validation receipt digest in the handoff before testing")
    logs = new_run(f"phase3-{args.stage}")
    print(f"Artifacts {out}\nLogs {logs}", flush=True)
    if args.stage == "test":
        options = ["--lock", str(out / "validation.json"), "--lock-sha", args.lock_sha]
        return int(not benchmark(logs, out, "test", "cuda", "test.json", options))

    out.mkdir(parents=True, exist_ok=False)
    if not run_checked(logs, "unit", ["-m", "unittest", "-v", "experiments.test_phase3"]):
        return 1
    if not benchmark(logs, out, "smoke", "cpu", "smoke.json"):
        return 1
    if not benchmark(logs, out, "validate", "cuda", "validation.json"):
        return 1
    validation = out / "validation.json"
    data = json.loads(validation.read_bytes())
    receipt = {"validation": str(validation.relative_to(ROOT)), "validation_sha256": digest(validation),
               "adapter_tree_sha256": data["adapter_tree_sha256"],
               "rates": data["chosen"]["rates"], "shifts": data["chosen"]["shifts"]}
    write_json(logs / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
