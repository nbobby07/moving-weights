"""Measure the Verlet stability boundary on a unit-frequency quadratic."""

import argparse
import math

import torch

from experiments.run_phase1 import new_run, write_json
from mttt.optimizers import make_optimizer


STEPS = 512
CHECKPOINTS = (0, 1, 10, 50, 100, 250, STEPS)
STEP_SIZES = (0.5, 1.0, 1.9, 1.99, 2.0, 2.01, 2.1, 2.25, 3.0)


def trajectory(device, dtype, h):
    q = torch.tensor([1.0], device=device, dtype=dtype)
    p = q.new_tensor([0.25])
    opt = make_optimizer("glide", h=h, fisher=torch.zeros_like(q))
    state = opt.init(q, p)
    path = [torch.cat((q, p))]
    for _ in range(STEPS):
        q, state = opt.step(q, state, lambda x: x, gamma=0.0)
        path.append(torch.cat((q, state[0])))
    history = torch.stack(path).tolist()
    first_nonfinite = next((i for i, z in enumerate(history) if not all(map(math.isfinite, z))), None)
    finite = history if first_nonfinite is None else history[:first_nonfinite]

    def observation(i):
        q_value, p_value = history[i]
        return {"step": i, "q": repr(q_value) if not math.isfinite(q_value) else q_value,
                "p": repr(p_value) if not math.isfinite(p_value) else p_value}

    checkpoints = set(CHECKPOINTS)
    if first_nonfinite is not None:
        checkpoints.update((first_nonfinite - 1, first_nonfinite))
    return {"h_omega": h, "dtype": str(dtype), "first_nonfinite_step": first_nonfinite,
            "max_finite_abs_component": max(abs(v) for z in finite for v in z),
            "checkpoints": [observation(i) for i in sorted(checkpoints)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run_dir = new_run(f"stability-{args.device}")
    print(run_dir, flush=True)
    path = run_dir / "stability.json"
    report = {"status": "started", "device": args.device, "torch": str(torch.__version__),
              "steps": STEPS, "initial_q_p": [1.0, 0.25], "runs": []}
    write_json(path, report)
    device = torch.device(args.device)
    for dtype in (torch.float64, torch.float32):
        for h in STEP_SIZES:
            try:
                row = trajectory(device, dtype, h)
            except Exception as error:
                report["runs"].append({"h_omega": h, "dtype": str(dtype), "error": repr(error)})
            else:
                report["runs"].append(row)
            write_json(path, report)
    report["status"] = "failed" if any("error" in row for row in report["runs"]) else "completed"
    write_json(path, report)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
