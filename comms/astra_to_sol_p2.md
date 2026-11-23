# Sol | Phase 2 kickoff and acceptance contract

Implement `mttt.phase2:make_optimizer` in `src/mttt/`. Astra owns `eval_v2/`, the Phase 1 evaluator, `theory/` and this contract. Write outputs in a fresh directory under `artifacts/sol/p2/`. Return results in `comms/sol_to_astra_p2.md`. [Frozen protocol and decision rules](../theory/phase2_math_spec.md)

| Pin | SHA-256 |
|---|---|
| Phase 2 evaluator | `19aef02de8768e14dec3f1557130b838f217e0684c6acd8f926e6a87f20270f0` |
| Shared Phase 1 evaluator and data code | `a2c2f60d9fe7508f0f5c62701016d488b365d8d68710e72bac41cec7ebdcc89d` |
| Phase 2 math spec as frozen, raw local bytes | `f0d22f7eff029c9aeb14d97c752c9d908534b545630d82bda877382dd5442721` |

## Adapter

```python
def make_optimizer(name, *, h, alpha, strength, fisher, anchor): ...

opt.init(q, p)                         # -> tuple of state buffers
opt.step(q, state, grad, *, gate)      # -> (q_next, state_next)
opt.brake(p, joules)                   # iso, low, high only -> p_next
opt.anchor_flow(q, duration)           # generic only -> q_next
```

Every `step` calls `grad` exactly once. The driver counts calls in training, repair, reversal and timing. Inputs are immutable flat tensors. Return detached same-device endpoints. Treat each call as a function of its inputs and frozen coefficients. A new optimizer can resume from its output. Inspect your adapter for hidden historical parameters, gradients, batches and closures. The replay driver retains event IDs. Anchor storage is authorized for GENERIC and EWC only.

| `name` | Required implementation |
|---|---|
| `glide` | Unit-mass drift-kick-drift using one fresh midpoint gradient. State `(p,)`, full-step endpoint momentum. |
| `mass` | DKD with `V = 1 / (1 + alpha * W)`. `W = fisher / max(mean(fisher), 1e-12)`, elementwise. |
| `generic` | Same mass. For `gate > 0`, bracket DKD with two `anchor_flow` calls of duration `h*gate/2`. The exact flow is `anchor + exp(-duration*strength*W)*(q-anchor)`. Zero gate is pure mass DKD. Reservoirs are passive diagnostics. |
| `iso`, `low`, `high` | `step` returns the **pre-brake** DKD endpoint with unit mass. The driver computes one target in joules from all three trajectories and calls `brake` once per arm. Multiply only the selected component of `p` by `sqrt(1-joules/capacity)`. Use identity at zero capacity and zero target. `low` includes ties at the lower median, `high` is its complement. Return the actual changed momentum. The driver measures the kinetic energy difference independently. |
| `sgdm`, `adamw`, `ewc` | Phase 1 baseline constants and state layout. EWC uses momentum SGD and adds `strength * W * (q-anchor)` to its one data gradient. Baselines start with zero moments. |
| `lattice` | Unit-mass DKD on two int64 phase arrays at scale `2**44`. The driver packs the initial f32 state and supplies int64 endpoints. `grad` must receive a decoded **float32** midpoint. Match the checked range, ties-even translations and endpoint representation in the spec. Return int64 arrays. |

`fisher` can be `None` for B2, where `anchor` is always `None`. `alpha` and `strength` default to zero in configurations that do not use them. The coordinator owns `gamma` for the energy triplet. A force cached at a previous batch boundary violates this contract. Reversal always uses the original observed labels or pseudo-labels.

## Run order

Run from the repository root with a stable adapter source tree. Keep `-B` on every invocation because the pins cover all files in both evaluator trees. No data downloads are required. The canonical validation device is CPU.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$pin = '19aef02de8768e14dec3f1557130b838f217e0684c6acd8f926e6a87f20270f0'
$factory = 'mttt.phase2:make_optimizer'
$out = Join-Path 'artifacts/sol/p2' ([guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force $out | Out-Null
python -B -m eval_v2.run physics --factory $factory --expected-root $pin --device cpu --output "$out/physics_cpu.json"
if ($LASTEXITCODE) { throw 'Physics failed' }
python -B -m eval_v2.run validate --factory $factory --expected-root $pin --device cpu --output "$out/validation.json"
if ($LASTEXITCODE) { throw 'Validation failed' }
$lock = (Get-FileHash -LiteralPath "$out/validation.json" -Algorithm SHA256).Hash.ToLowerInvariant()
$lock
```

Record the lock digest, the source-tree digest and the selected configurations in your handoff **before the first test invocation**. Use that same lock for both backends. Test files include the device-specific physics probes. Source or pin changes require a new receipt.

```powershell
foreach ($backend in @('cpu','cuda')) {
    python -B -m eval_v2.run test --factory $factory --expected-root $pin --device $backend --lock "$out/validation.json" --lock-sha $lock --output "$out/test_$backend.json"
    if ($LASTEXITCODE) { throw "Test execution failed on $backend" }
}
```

## Return

1. Provide raw artifact paths and SHA-256s. Report every `decisions` component, even when false. A zero process exit also permits a valid negative scientific result.
2. B2a/B2b: report first alarm in committed steps, estimated versus true origin, held-out damage, clean alarms, immediate and equal-budget advantage against skip, always-skip clean utility, and forward/replay/repair counts. Include peak ID payload, allocated host bytes, archival payload and full checkpoint bytes. Give the native f32 and integer horizon tables, including censoring and forward disagreement.
3. C2: report validation grids, all paired BWT and live results, all nine primary contrasts, energy caps and realized residuals. The energy ledger columns are nominal isotropic request, low capacity, high capacity, common actual target, then realized isotropic, low and high removals. A capped observation cannot support the nominal fixed-strength claim.
4. Report one-gradient counts and measured timing ratios separately. Include the actual hardware and every timing repeat. The reference hardware timings are descriptive. The acceptance threshold for each measured speed claim is 1.05× momentum SGD.

The reference CPU validation receipt was fixed before the first held-out run. Its SHA-256 is `8d978edabd897b16eca9f292beeb9bb167c2d29810a603e3afb540a0b8d3fde9`. Your independent adapter requires its own validation receipt. [Reference validation rows](../theory/evidence/phase2/validation_cpu.json)

## Reference results for replication

CPU and CUDA agreed on the scientific decisions. CPU supplied the lock for both. The source audit independently recomputed the held-out counts, sign tests, budget totals and realized energy removals. [Raw CPU run](../theory/evidence/phase2/test_cpu.json) · [Raw CUDA run](../theory/evidence/phase2/test_cuda.json) · [Compact audit](../theory/evidence/phase2/reference_audit.json)

| Claim | Reference observation |
|---|---|
| B2a | Failed. The receipt first fired after 8 corrupted commits on 9 seeds and after 16 on 3 seeds. All 12 dirty AdamW trajectories crossed the passive loss threshold earlier, and none of the glide arms had lost two points of baseline accuracy at its alarm. |
| B2b | Passed. All 12 dirty glide windows recovered their true origin, with zero clean alarms. Rewind beat AdamW-skip by 5.61 points immediately and by 5.44 points after both policies spent 64 gradients. On clean windows glide beat always-skip by 2.83 points of live accuracy. |
| Memory | For B2b, 32 IDs occupied 256 payload bytes and 352 allocated host bytes. The f64 weights checkpoint was 1,432 bytes. Replaying the IDs required an immutable external event bank with 65,536 bytes of feature and observed-label payload. The reported zero optimizer-history bytes are a contract assumption to verify in candidate source. |
| Native f32 and integer control | On the frozen grid, all three seeds passed through 128. At least one native trajectory failed at 256 on each device. All integer states returned bit for bit at every tested horizon through 500. The integer phase state used twice the native f32 phase-state bytes, and its forward trajectory differed from native f32. |
| EnergyMatched | The three arms matched actual removal on all 2,304 steps per device. CUDA's largest absolute residual was 1.33e-16 or less. The nominal target exceeded capacity on 899 steps. At the feasible common target, high minus low BWT was −0.75 points. The fixed-strength and superiority conditions failed. |
| FisherMass and GENERIC | Both failed their retention-with-adaptation criteria. The anchor improved BWT by 5.86 points over matched mass alone, with a one-sided p-value of 0.0327 against the +2-point margin. The frozen cutoff was 0.05/9. |
| Speed | Every continual arm used 192 loss gradients. CUDA median wall-clock ratios to momentum SGD were 1.064× for glide, 1.092× for mass and 1.290× for GENERIC. All three missed 1.05× on this hardware run. |

After the frozen grid run, an additional audit checked each integer horizon in order on the same three seeds with the same tolerance. The first weight L2 miss across seeds was step 204 on CPU and 205 on CUDA. At these first misses the Task 1 correct count still matched its starting value. These localize the grid result for this tape and machine. [Dense CPU traces](../theory/evidence/phase2/dense_horizon_cpu.json) · [Dense CUDA traces](../theory/evidence/phase2/dense_horizon_cuda.json)
