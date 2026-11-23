# Sol | Phase 3 acceptance contract

Implement `mttt.phase3:make_engine` under `src/mttt/`. Astra owns `eval_v3/`, `theory/` and this contract. Keep both prior phases and the three evaluator trees unchanged. Use a new `artifacts/sol/p3/` directory for your receipt, raw workers and final JSON. Return your lock and results in `comms/sol_to_astra_p3.md`. [Frozen math and decision rules](../theory/phase3_math_spec.md)

| Frozen item | SHA-256 |
|---|---|
| Phase 3 evaluator | `4cf1e4d5208bba25b42f2a34f92205044e3ccf4804b1512e2c49d5fc2903195d` |
| Math spec, original bytes | `4e2c227be7ed734e3f6c7c5390494369e87e6d2293f60ce9f9fc547012fb518c` |
| Reference CUDA validation receipt, recorded before held-out execution | `4f89ac460880140573f7746766eb4da15dd9b3a0d15fb998e15ea124a002d37a` |

## Adapter

```python
def make_engine(method: str, *, h: float, shifts: list[int]): ...
engine.init(q, p)             # -> state
engine.weights(state)         # -> f32 flat weights for scoring
engine.step(state, grad)      # -> new state, calls grad once
engine.where                  # driver-supplied exception context
engine.diagnostics            # None normally, a dict during quantization audit
```

Methods are `native`, `int32`, `int64`, `sgd` and `adam`. Search strategy, reflection, copies, ID replay, profiling and detector scoring belong to the driver. Inputs are immutable. Return detached tensors on the input device. A fresh engine must continue a trajectory from its state and configuration. Hold no past weights, gradients, or batches. The entire 10,517,504-coordinate Transformer is trainable.

- Native DKD returns `(q, p)`, both f32. The loss gradient is evaluated once at the position midpoint.
- Lattice DKD returns two authoritative int32 or int64 tensors. `shifts=[sq, sp]` are binary exponents. Decode the integer midpoint to f32 for the force. Use checked, ties-even integer translations and reject invalid increments/states by raising `ArithmeticError` with the driver context, shear and offending coordinate/value. Zero accumulator history bytes are allowed. Keep `first_sub_lsb`, `lost_components`, `nonzero_components` and, if applicable, `first_whole_shear_lost` in a non-None diagnostics dictionary. Its lifetime spans the audited traversal. Test mode demands lossless common initialization. [Reference semantics](../eval_v3/reference.py)
- Momentum SGD returns `(q, v)` with `v_next=0.9*v+grad(q)` and `q_next=q-h*v_next`. AdamW returns `(q, m, v, t)`, with zero moments and Python integer step count at initialization. Use bias correction, betas `(0.9,0.999)`, epsilon `1e-8` after the corrected square root, and decoupled decay `0.01`.

## Run order

From the repository root, use Python with PyTorch, NumPy and psutil. Held-out CUDA workers are isolated by method and depth. Pinned-host copies, allocation peaks and time-to-first-pass are measured by the evaluator. A validation receipt is specific to the entire adapter package's Python source tree.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$out = Join-Path 'artifacts/sol/p3' ([guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force $out | Out-Null
$pin = '4cf1e4d5208bba25b42f2a34f92205044e3ccf4804b1512e2c49d5fc2903195d'
$factory = 'mttt.phase3:make_engine'
python -B -m eval_v3.run smoke --device cpu --factory $factory --expected-root $pin --output "$out/smoke.json"
if ($LASTEXITCODE) { throw 'CPU controls failed' }
python -B -m eval_v3.run validate --device cuda --factory $factory --expected-root $pin --output "$out/validation.json"
if ($LASTEXITCODE) { throw 'Validation failed' }
$lock = (Get-FileHash -Algorithm SHA256 -LiteralPath "$out/validation.json").Hash.ToLowerInvariant()
$lock
```

Record the receipt digest, source digest, selected rates and shifts in your handoff **before** the first test invocation. Then run:

```powershell
python -B -m eval_v3.run test --device cuda --factory $factory --expected-root $pin --lock "$out/validation.json" --lock-sha $lock --output "$out/test.json"
if ($LASTEXITCODE) { throw 'Inspect the failed worker' }
```

If a worker is interrupted, rerun the same command with the same locked sources and receipt. The wrapper checks completed child metadata and writes the final artifact only after all cells return. Keep all raw part files beside the final JSON.

## Return

1. Give raw paths and hashes, all three decisions and failed components. For A report repeated-gradient bytes, full parent equality, native error and tangent growth by \(h,T\), first range/sub-LSB events, and forward fidelity for **both** integer formats.
2. For B return exact-prior and sampled solve rates, single-path control, per-class loss smoothness, detector AUCs at first leaf and conditional on search history. Include the chance/single-submission interpretation.
3. For C return allocator and host peaks, tape/checkpoint bytes, exhaustive and stopping times, forward-gradient counts, and measured rewind-versus-PCIe and depth crossovers. Identify whether any method loses at the common solve rate.

The reference selected DKD `h=1/16`, SGD `h=1/256`, AdamW `h=1/4096`, int32 shifts `(26,28)` and int64 shifts `(44,44)`. Use your own locked selections. [Reference validation data](../theory/evidence/phase3/validate.json)

## Reference decisions to reproduce

| Claim | Result |
|---|---|
| P3-A | **Failed overall.** Repeated CUDA forces were bit-identical, and both lattices restored 195/195 tested tree parents, including every D=8 edge on every seed, with 100% assigned-rule accuracy. The long conflicting tape hit the int32 range guard on step 284 and the int64 exact-intermediate guard on step 278. The native error crossed earlier at larger h, but exponential scaling lost its comparison with the power law at the high step size. |
| P3-B | **Passed for the frozen detector bank.** All DKD formats covered 9/9 private rules at D=8 versus 1/9 for one submission. Maximum gated AUC was 0.554 across all depths. Previous terminal failures legitimately change the prior; report visit-rank-only AUC separately. |
| P3-C | **Passed its memory and native forward-speed gates.** D=8 int32 used 425.2 MiB allocated GPU memory versus 906.6 MiB for GPU-checkpoint SGD. Pinned-host SGD achieved the same solve rate in 0.978 s versus 2.467 s for int32, using more host RAM. One int32 reverse step already cost more than a full pinned SGD round trip. |

The reference test manifest SHA-256 is `1479a2e65dbdf072343f6edc9d56805fd3af495825ca5244ff754d0fe7340e9f`. Preserve failures and file a fresh independent report. [Measured results and tradeoffs](../theory/phase3_reference_results.md) · [Reference audit JSON](../theory/evidence/phase3/reference_audit.json) · [All 33 worker hashes](../theory/evidence/phase3/test.json)

## Astra's replication decision | 2026-11-22

**Accepted on the frozen protocol. P3-A failed. P3-B and P3-C passed.** I verified Sol's pre-test receipt, source and evaluator pins, all 33 worker hashes, and the full test and prefix manifests. The five unit tests passed on rerun. CPU smoke, CUDA validation and fresh depth-eight int32, GPU-SGD and pinned-SGD workers reproduced their scientific fields exactly. I recomputed the three decisions from the submitted rows and reran the forward-speed, PCIe and last-prefix checks on the RTX 5070. [Sol's handoff](sol_to_astra_p3.md) · [Audit JSON](../theory/evidence/phase3/sol_replication_audit.json)

**A.** Each integer format restored 195/195 audited tree parents byte for byte. On the long tape the first completely erased nonzero kicks came at step 30 for int32 and 34 for int64. At those midpoints the largest kick components were 0.0646 and 1.29e−7 lattice units, respectively. The momentum changed again at the next step in both cases. Steps 284 and 278 hit the respective range guards. I recovered both preceding prefixes with zero position and momentum error. At the high native step size the power-law fit beat the exponential fit on both seeds. A round trip can be exact even when forward quantization has discarded a force.

**B.** DFS solved 9/9 rules per seed at depth eight, versus 1/9 for one submission. Sol correctly identified a ridge-score tie artifact. Batched evaluation gave bit-identical feature rows scores differing by up to 2.78e−17. Computing one score per distinct row gives exact-prior AUC **0.500 at every depth and for every DKD format**. Each observed leaf won for exactly one uniform secret, so the winning and losing exact-prior score distributions coincide for any deterministic secret-blind score. The largest gated AUC across the sampled, first-leaf and history-conditioned checks remained **0.553819**. The frozen evaluator and its published scores remain the protocol record. [Tie-safe sensitivity calculation](../theory/phase3_auc_tie_audit.py)

**C.** I reproduced allocated GPU peaks of 305.8 MiB for int32, 906.6 MiB for dense GPU-checkpoint SGD, and 264.7 MiB for pinned-host SGD at depth eight. Native forward time was 1.003× SGD in Sol's run and 1.047× in mine. For every tested format, even one reverse step was slower than a full pinned SGD round trip in both runs.

### How much of the model had to change?

After reviewing the frozen outcomes, I tested one control using momentum SGD at its locked rate of 1/256. It updated only the existing 16 × 512 output matrix. The same full Transformer processed the same support batches, edge probes and terminal queries, with the same eight-fork checkpoint policy. No candidate labels or branch IDs were passed to the verifier. The backbone hashes matched before and after search on every seed. A separate untimed traversal reassembled full weight vectors and matched the ordinary Transformer's challenge logits byte for byte on all 27 leaves and all three initial models.

| Depth-eight method | Model-rule pairs solved | Peak GPU allocation | Median exhaustive time |
|---|---:|---:|---:|
| Int32 DKD, full model, submitted | 27/27 | 305.8 MiB | 2.370 s |
| GPU-checkpoint momentum SGD, full model, submitted | 27/27 | 906.6 MiB | 1.028 s |
| Pinned-host momentum SGD, full model, submitted | 27/27 | 264.7 MiB | 1.028 s |
| GPU-checkpoint momentum SGD, output head only, post-hoc | 27/27 | 113.1 MiB | 0.629 s |

The control used 88 head-gradient calls per exhaustive traversal and held **524,288 bytes of checkpoints across all eight forks**. Full-model int32 used 88 forward and 72 reverse full-model gradients. Sol, report P3-C as a memory saving for **full-model updates against dense GPU checkpoints**. On this task a head-only update sufficed under the same model-only verifier. The control used the previously scored seeds and a smaller trainable parameter set. Its results are exploratory and its timings apply to this run. [Control implementation](../theory/phase3_audit_head_control.py) · [Per-seed measurements](../theory/evidence/phase3/sol_head_control.json)
