"""Run the three frozen benchmarks in fresh processes and keep every attempt."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
ARTIFACTS = ROOT / "artifacts" / "sol"
FROZEN = "a2c2f60d9fe7508f0f5c62701016d488b365d8d68710e72bac41cec7ebdcc89d"
ADAPTER = "mttt.optimizers:make_optimizer"
BENCHMARKS = {"physics": "eval.verify_physics", "analytic": "eval.bench_analytic",
              "continual": "eval.bench_continual"}


def new_run(kind):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = RESULTS / f"{kind}-{stamp}-{os.getpid()}"
    directory.mkdir(parents=True)
    return directory


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_command(run_dir, name, args, *, artifact=None):
    command = [sys.executable, "-B", *args]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), existing)))
    record = {"command": command, "cwd": str(ROOT), "pythonpath": env["PYTHONPATH"],
              "status": "started"}
    log = run_dir / f"{name}_run.json"
    write_json(log, record)
    print(f"Running {name}", flush=True)
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                encoding="utf-8", errors="backslashreplace", check=False)
        has_artifact = artifact is not None and artifact.is_file()
        passed = result.returncode == 0 and (artifact is None or has_artifact)
        record.update(status="passed" if passed else "failed", exit_code=result.returncode,
                      stdout=result.stdout, stderr=result.stderr,
                      artifact=artifact.name if has_artifact else None)
        return passed
    except (OSError, KeyboardInterrupt) as error:
        record.update(status="failed", error=repr(error))
        if isinstance(error, KeyboardInterrupt):
            raise
        return False
    finally:
        write_json(log, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--acceptance", action="store_true", help="copy raw JSON into artifacts/sol")
    parser.add_argument("--run-tests", action="store_true", help="run the independent unittest suite first")
    parser.add_argument("--no-report", action="store_true", help="skip plotting and tabulation")
    args = parser.parse_args()
    run_dir = new_run(f"frozen-{args.device}")
    print(run_dir.relative_to(ROOT), flush=True)
    source_digests = {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
                      for path in sorted((ROOT / "src" / "mttt").glob("*.py"))}
    statuses = {}
    if args.run_tests:
        statuses["tests"] = run_command(run_dir, "tests", ["-m", "unittest", "discover", "-s", "tests", "-v"])
    for name, module in BENCHMARKS.items():
        artifact = run_dir / f"{name}.json"
        statuses[name] = run_command(
            run_dir, name, ["-m", module, "--factory", ADAPTER, "--expected-root", FROZEN,
                            "--device", args.device, "--output", str(artifact.relative_to(ROOT))],
            artifact=artifact,
        )
    publish = args.acceptance and all(statuses[name] for name in BENCHMARKS)
    if publish:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        for name in BENCHMARKS:
            artifact = run_dir / f"{name}.json"
            temporary = ARTIFACTS / f".{artifact.name}.{run_dir.name}.tmp"
            shutil.copyfile(artifact, temporary)
            temporary.replace(ARTIFACTS / artifact.name)
    if not args.no_report and all((run_dir / f"{name}.json").is_file() for name in ("analytic", "continual")):
        statuses["report"] = run_command(
            run_dir, "report", ["-m", "experiments.phase1_report", str(run_dir.relative_to(ROOT))],
            artifact=run_dir / "summary.csv",
        )
    write_json(run_dir / "run.json", {"device_requested": args.device, "copy_to_acceptance": args.acceptance,
                                    "acceptance_published": publish, "source_sha256": source_digests,
                                    "checks": statuses})
    return 0 if all(statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
