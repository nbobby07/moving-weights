"""Shared run contract, frozen protocol, metrics and exact-byte integrity check."""
import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
PIN = ROOT.parent / "theory" / "phase1_eval.sha256"
SEEDS = (1729, 2718, 3141, 4211, 5501, 6673, 7919, 8861)
METHODS = ("glide", "strang", "fisher", "sgdm", "adamw")
PHASE = METHODS[:3]
H = {"glide": 0.12, "strang": 0.12, "fisher": 0.12, "sgdm": 0.0144, "adamw": 0.01}
GAMMA = 1.0
ORACLE = "eval.reference:make_optimizer"


def fingerprint():
    # Include new files, names and bytes. The pin lives outside this tree.
    digest = hashlib.sha256()
    for path in sorted(ROOT.rglob("*"), key=lambda p: p.relative_to(ROOT).as_posix()):
        if path.is_symlink():
            raise ValueError(f"unexpected symlink in eval: {path}")
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def verify_hash(expected):
    actual = fingerprint()
    if actual != expected:
        raise RuntimeError(f"eval integrity failed: expected {expected}, got {actual}")
    return actual


def l2(a, b):
    # Accumulate the error in f64 even when the trajectory itself is f32.
    return torch.linalg.vector_norm(a.double() - b.double()).item()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tick(device):
    sync(device)
    return time.perf_counter()


def unwind(opt, q, state, steps, gradient):
    # Each call to gradient(i) regenerates/selects the original forward batch.
    p, = state
    state = (-p,)
    for i in reversed(range(steps)):
        q, state = opt.step(q, state, gradient(i), gamma=0.0)
    return q, (-state[0],)


def run(benchmark, evaluate):
    parser = argparse.ArgumentParser(description=benchmark)
    parser.add_argument("--factory", default=ORACLE, metavar="MODULE:CALLABLE")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--expected-root", help="independently supplied SHA-256 pin")
    parser.add_argument("--output", type=Path, help="JSON artifact outside eval/ (default stdout)")
    args = parser.parse_args()
    if args.output and args.output.resolve().is_relative_to(ROOT):
        parser.error("artifact must be outside eval/")
    # This must precede the first CUDA context. The launch command disables .pyc.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    expected = args.expected_root or PIN.read_text(encoding="ascii").strip()
    verify_hash(expected)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEEDS[0])
    name = args.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    module, attr = args.factory.split(":")
    imported = importlib.import_module(module)
    factory = getattr(imported, attr)
    origin = getattr(imported, "__file__", None)
    report = {
        "benchmark": benchmark, "eval_sha256": expected, "factory": args.factory,
        "factory_module_sha256": hashlib.sha256(Path(origin).read_bytes()).hexdigest() if origin else None,
        "torch": str(torch.__version__), "python": sys.version.split()[0],
        "platform": platform.platform(), "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if name == "cuda" else platform.processor(),
        "cuda_build": torch.version.cuda, "deterministic": True, "threads": 1,
    }
    try:
        report.update(evaluate(factory, device))
    finally:
        verify_hash(expected)
    data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(data, encoding="utf-8")
    else:
        print(data, end="")
    if report.get("checks_passed") is False:
        raise SystemExit(1)
