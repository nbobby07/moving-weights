# Sol | Phase 1 implementation contract

Implement `mttt.optimizers:make_optimizer` in your own package. Astra owns `theory/`, `eval/` and this file. Treat every byte in `eval/` as read-only. Put candidate outputs in `artifacts/sol/` and your reply in `comms/sol_to_astra.md`. Implement the equations independently. The audit oracle is available for diagnosis. [Math and thresholds](../theory/phase1_math_spec.md)

**Frozen SHA-256** `a2c2f60d9fe7508f0f5c62701016d488b365d8d68710e72bac41cec7ebdcc89d`

## Python interface

```python
from collections.abc import Callable
from typing import Protocol
from torch import Tensor

Grad = Callable[[Tensor], Tensor]
State = tuple[Tensor] | tuple[Tensor, Tensor, int]

class Optimizer(Protocol):
    def init(self, q: Tensor, p: Tensor) -> State: ...
    def step(self, q: Tensor, state: State, grad: Grad, *,
             gamma: float) -> tuple[Tensor, State]: ...
    def brake_diag(self, q: Tensor) -> Tensor: ...  # queried on phase methods

def make_optimizer(name: str, *, h: float, fisher: Tensor) -> Optimizer: ...
```

`q`, `p` and `fisher` are flat equal-length tensors on one device, f64 or f32. `fisher` is a finite nonnegative diagonal. `h` is positive. `grad(x)` returns the positive gradient of the current batch loss, matching `x`. The evaluator owns autograd, batch selection, alarms and momentum flips. Return detached finite tensors with the input shape, dtype and device. Treat all inputs as immutable. An optimizer can cache frozen coefficients. Each call must depend only on its arguments and those coefficients. The evaluator can replace the instance between any two steps.

| `name` | Initial state | Required update |
|---|---|---|
| `glide` | `(p.clone(),)` | Kick-drift-kick, unit mass, two force sites, zero gate. `brake_diag` returns ones as the ungated coefficient. |
| `strang` | `(p.clone(),)` | Same leapfrog between two half-brakes. `brake_diag` returns ones. |
| `fisher` | `(p.clone(),)` | Same split. `brake_diag` returns `(fisher <= fisher.median()).to(q)`. Include lower-median ties. |
| `sgdm` | `(zeros_like(q),)` | `v1 = 0.9*v + grad(q)`, then `q1 = q - h*v1`. One force evaluation, zero dampening and weight decay, Nesterov off. |
| `adamw` | `(zeros_like(q), zeros_like(q), 0)` | Standard bias-corrected AdamW with betas `(0.9, 0.999)`, epsilon `1e-8` outside the square root, decoupled weight decay `0.01`. Return `(m1, v1, n+1)`. One force evaluation. |

For both splits multiply momentum elementwise by `exp(-0.5*h*gamma*brake_diag(q))` before the first kick and at the new position after the second kick. The harness passes `gamma=0` to both baselines. No worker may cache old parameters, momenta, gradients, closures, minibatches or RNG history in the optimizer. Review the implementation for this constraint. A hash check verifies evaluator files and a fresh-instance replay tests state sufficiency. Neither audits hidden external storage by a candidate.

## Acceptance run

From the repository root in a fresh Python process per command, use Python 3.10+ and PyTorch with `torch.func`. The commands below are PowerShell. `-B` prevents bytecode writes to the frozen tree. Select `--device cpu` for CPU fallback. Astra verified both backends using PyTorch 2.14.0+cu130. SymPy is used only for the separate symbolic check.

```powershell
$frozen = 'a2c2f60d9fe7508f0f5c62701016d488b365d8d68710e72bac41cec7ebdcc89d'
$adapter = 'mttt.optimizers:make_optimizer'
New-Item -ItemType Directory -Force artifacts/sol | Out-Null
python -B -m eval.verify_physics --factory $adapter --expected-root $frozen --device auto --output artifacts/sol/physics.json
python -B -m eval.bench_analytic --factory $adapter --expected-root $frozen --device auto --output artifacts/sol/analytic.json
python -B -m eval.bench_continual --factory $adapter --expected-root $frozen --device auto --output artifacts/sol/continual.json
```

1. Return all three JSON paths, adapter source revision or digest, invocation, device and library version. Report any exception or nonzero exit with its failing check. The pinned digest covers sorted relative file names, a NUL byte after each UTF-8 name, and each file's raw SHA-256 digest. Keep the pin supplied here independent of the checkout.
2. Require passing physics checks, passing energy-order checks, and absolute f64 rollback errors below `1e-11` in both phase components. Include the f32 curve. A physics mismatch goes back to implementation review.
3. Return `verdict.B_components` and `verdict.C_components` even when false. Include poison and clean alarm counts, pre/post/follow-up accuracies, paired BWT contrasts and sign-test p-values, live accuracy and synchronized step times. A negative learning result is a valid deliverable.

The reference run passed A and finite-horizon B and failed C. The Fisher brake averaged 2.125 percentage points worse Task 1 BWT than momentum SGD and 3.00 points worse than the isotropic brake. Preserve the locked protocol for replication. Any proposed change belongs in a separately identified follow-up. [Reference measurements](../theory/evidence/continual_cuda.json)

## Astra's replication decision | 2026-11-22

**Accepted on the frozen protocol. A passed. B passed at 32 clean follow-up steps. C failed retention.** I verified the evaluator pin and all four adapter source hashes, checked the phase implementation for saved trajectory state, and reran the six unit tests and all three benchmarks on the supplied implementation. All seven physics probes passed on each backend. On each device every recorded field outside wall-clock timing reproduced exactly. I recomputed the energy orders from the traces, alarms from the loss thresholds, and the paired tests from integer correct counts in the task matrices. [Audit record](../theory/evidence/phase1_replication_audit.json) · [Sol's handoff](sol_to_astra.md)

| Claim | Audit finding |
|---|---|
| A | The 500-step CUDA neural replay recovered weights to L2 error `2.73e-15` and momentum to `3.83e-16`. Reproduction required the ordered batch stream. |
| B | Glide restored the original correct count in 8/8 poisoned windows, with one clean alarm. Unrewound AdamW scored 116 fewer correct across 2,400 held-out predictions than its matched clean run after 32 steps, a 4.83-point deficit. Every poisoned window crossed the supervised threshold before its first update. Given the known boundaries, skipping each flagged window at that point would also have preserved the starting weights. The frozen comparison measures recovery after applying the updates. |
| C | Fisher reached 97.99% live accuracy. Its Task 1 accuracy fell from 98.96% to 3.58% after Task 5. Its retention differences were −2.125 points versus momentum SGD, +0.292 versus AdamW and −3.000 versus isotropic damping. No seed exceeded the required +2-point margin against either primary baseline. Both one-sided p-values were 1. |

Sol, report the comparator as AdamW that continues updating after the alarm. Unknown onset and longer recovery remain open. The projected brake acted on 90 of 179 coordinates at the same nominal gate as the isotropic brake. This comparison leaves the effect of orientation at matched energy dissipation unresolved. The supplemental sweeps are outside the frozen decision.
