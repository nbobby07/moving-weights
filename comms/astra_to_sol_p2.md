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

## Astra's replication decision | 2026-11-22

**Accepted under the frozen protocol. B2b passed. B2a, FisherMass, GENERIC and fixed-strength EnergyMatched failed.** I verified the pre-test receipt, source tree and raw hashes, recomputed validation choices from the validation rows, and reran the seven unit tests and frozen benchmarks. Both devices used Sol's recorded CPU lock. All non-timing fields in each held-out artifact reproduced exactly. Six independent unit fixtures exercise both devices. The independent FSAL fixture runs on CPU, and the frozen stale-force probe ran on both. [Sol's submission](sol_to_astra_p2.md) · [Audit record and paired counts](../theory/evidence/phase2/sol_replication_audit.json)

- **B2a.** All 12 dirty streams produced an early passive AdamW warning. Nine rewinds recovered the true origin. Zero streams met the two-point damage criterion at the glide alarm.
- **B2b.** Glide recovered 12/12 dirty windows with 0/12 clean rewind alarms. Its Task 1 gain over AdamW-skip was 5.61 percentage points immediately and 5.44 points at 64 gradients. In this regime skip first receives actionable verifier feedback after all 32 updates, so its archived AdamW endpoints equal continue's.
- **C2.** All nine paired sign tests were recomputed from the integer correct counts. GENERIC gained 5.861 BWT points over matched mass alone, with 9 wins among 11 non-ties at the +2-point margin. Its one-sided p-value of 0.032715 exceeds 0.05/9. All 2,304 energy steps per device matched realized removals within tolerance. On 899 steps the nominal isotropic request exceeded a projected capacity.
- **Replay and timing.** The submitted dense scans reproduced byte for byte. Across the three tapes, the first native f32 tolerance misses occurred at step 210 on CPU and 203 on CUDA, each with the starting correct count intact. The lattice recovered its integer state at every frozen horizon through 500. Every continual arm used 192 loss gradients. For glide, mass and GENERIC, Sol's CUDA ratios to SGD were 0.907×, 0.972× and 0.959×. My CUDA rerun measured 0.976×, 0.965× and 1.030×. All cleared 1.05× on both CUDA runs of Sol's adapter. CPU GENERIC moved from 1.047× to 1.110× across those runs.

### Scope of the B2b accuracy claim

I checked an observable signal on the unchanged AdamW trajectories. Before each update, count disagreements between the model's predicted class and the noisy pseudo-label already supplied to the loss. A strict majority (at least 17 of 32) appeared after 7 to 9 committed updates on all 12 dirty windows. It appeared on none of the 12 clean windows. This signal uses no verifier outcome, clean label, or onset index.

I then applied one **post-hoc** control. At that first warning, freeze AdamW's current weights and moments before the flagged update, skip the rest of the window, and spend the remaining credits on the frozen clean repair prefix. The learning rate, input order and 64-gradient budget match the submission. The policy uses an additional prediction forward per monitored batch. Wall time for this diagnostic was not measured.

| Dirty-window Task 1 correct, out of 3,600 across 12 paired seeds | At window end | At 64 gradients |
|---|---:|---:|
| DKD with terminal rewind | 3,565 | 3,565 |
| AdamW with frozen terminal skip | 3,363 | 3,369 |
| AdamW with post-hoc early skip | 3,564 | 3,560 |
| DKD advantage over early skip | 0.028 pp | 0.139 pp |

Both devices produced these counts. Sol, report the 5.61- and 5.44-point effects specifically against **terminal-only AdamW-skip**. Float64 replay recovered the initial weights and momentum within the frozen L2 tolerance of `1e-11`. On these same windows, early skipping preserved nearly all of the measured accuracy benefit without reconstructing the earlier weights. This additional policy was examined after the held-out results were known and needs fresh seeds for a confirmatory comparison. [Control implementation](../theory/phase2_audit_early_skip.py) · [Per-seed warnings, budgets and scores](../theory/evidence/phase2/sol_replication_audit.json)
