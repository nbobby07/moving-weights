# Sol to Astra | Manuscript handoff

This handoff records the manuscript version that passed Astra's final claim-to-artifact audit before the editorial polish. The audited PDF is 16 Letter pages and can be rebuilt from the archived source package below. The current manuscript has its own build receipt. [Audited source package](../paper/manuscript-source.zip) · [Current PDF](../paper/main.pdf) · [Current build receipt](../paper/verification/polished-build.json) · [Build instructions](../paper/README.md)

## Independent signoff

Astra approved the release identified below after rebuilding from the source ZIP in two further fresh detached checkouts. Each build verified all 58 original pins after staging and reproduced all 16 delivered outputs byte for byte. Astra checked the rendered Table 2 caption and terminal counts against both backends, reproduced the exploratory early-skip outcomes on CPU and CUDA and the head-only CUDA leaf votes and correct counts, and confirmed Figure 4's modeled and measured crossovers. All release findings are closed.

## Release identity and clean builds

| Item | Identifier |
| --- | --- |
| Evidence Git revision | `834128d6870fab5dcd2a4c6a39c91d3750632602` |
| Source ZIP SHA-256 | `54f07da13159e6f57d93e6b870dfdb3b2a8e5e6b14de39cd4b07d690b196adb0` |
| Original manifest SHA-256 | `d52b5df5f903361eb12e85d039da2f93083d5d58ff5525294f8ae085859834d2` |
| Final PDF SHA-256 | `dc200185ff058d1a68ea735bd315bf94cb2c60fb0fb291621ca5909b125f9b3c` |

Each test used a separate clone with `--no-checkout --no-hardlinks`, set its local `core.autocrlf` before the detached checkout, and unpacked the exact source ZIP listed above. No generated files or evidence were copied from the working tree into either build. Git blobs supplied evidence and the shared compiler cache supplied TeX resources. Each unstaged build correctly stopped without writing output. The restorer then hashed all 58 files under a new staging root and recovered 44 directly from Git blobs and 14 through exact CRLF reconstruction. The original 58 paths and SHA-256 pins are unchanged.

| Local checkout mode | Original pins matched before staging | Original pins matched after staging | Tables, figures, and PDF |
| --- | ---: | ---: | --- |
| `core.autocrlf=true` | 15/58 | 58/58 | Rebuilt, zero TeX warnings |
| `core.autocrlf=false` | 44/58 | 58/58 | Rebuilt, zero TeX warnings |

All 16 generated files were byte-identical across modes. The delivered assets are those tested bytes. Both detailed logs include command lines, exit codes, per-path restoration records, and output digests. [CRLF-on build](../paper/verification/clean-autocrlf-true.json) · [CRLF-off build](../paper/verification/clean-autocrlf-false.json) · [CRLF-on TeX log](../paper/verification/clean-autocrlf-true.tex-log.txt) · [CRLF-off TeX log](../paper/verification/clean-autocrlf-false.tex-log.txt)

The first CRLF-on staging attempt received Windows error 5 during the final directory rename. The same published restorer succeeded on retry with a fresh destination in that scratch directory. The CRLF-on log records both attempts.

Appendix A also gives commands for staging the archived controls from Git bytes and verified receipts. I checked both historical bundles under `core.autocrlf=true`. All five historical evaluator and adapter tree digests matched, and the three control entry points passed startup checks. No study or control was rerun. [Archived control staging log](../paper/verification/archive-controls.json)

## Audit corrections

- Table 2 reports clean alarms separately for DKD-rewind, AdamW-skip, and AdamW-continue over 12 clean windows per policy and regime. The B2b counts are 0/12, 4/12, and 4/12 on both CPU and CUDA. The damage rows use dirty DKD-rewind windows.
- Section 6.1 identifies the long lattice tape by seed, locked step, and format-specific shifts. The native growth table gives the fit rule and counts. Both native high-step fits use only the four eligible horizons. Directional rates remain at step 500.
- Tables 5 and 6 separate exact-prior model-rule pairs from sampled draws. Stopping time charges exhaustion on failed draws. The head-only value is 0.320 s over all 384 draws, and the text states the trainable head size, changed backward work, and reused seeds.
- Figure 4 states the event count per edge, ID allocation assumption, and logical AdamW counter width. The model's first strict int64 storage win is at two forks. The allocator first wins at tested depth four, with depth three unmeasured.


## Claim-to-artifact map

| Manuscript claim | Evidence and scope |
| --- | --- |
| Phase 1 replay, finite-horizon poison recovery, and retention failure | `artifacts/sol/analytic.json`, `continual.json`, and `physics.json` plus `theory/evidence/phase1_replication_audit.json`. The poison alarms all precede the first update. The measured comparator continues updating. Both primary Fisher retention tests have (p=1.000). |
| Phase 2 B2a and B2b | CPU and CUDA `test_*.json` under `artifacts/sol/p2/sol-20261122-cpu-lock/`, and the independent `theory/evidence/phase2/sol_replication_audit.json`. B2a fails its registered conjunction. B2b passes against terminal-only AdamW-skip. |
| Phase 2 exploratory early skip | `theory/evidence/phase2/sol_replication_audit.json`, `backends.{cpu,cuda}.exploratory_B2b`. Early skip gets 3,564 versus 3,565 correct out of 3,600 at the immediate endpoint. Rewind's gains against this policy are 0.028 and 0.139 pp at the two endpoints. The policy was selected after seeing the same outcomes, adds monitoring forwards, and has no measured timing. |
| Phase 2 retention and precision | The locked tests and independent audit support all primary C2 comparisons. GENERIC versus mass is +5.86 pp with (p=0.0327>0.05/9). The energy cap binds on 899 of 2,304 steps. Dense replay scans come from the two `dense_*.json` files and are distinguished from the frozen horizon grid. |
| Phase 3 lattice exactness and long-tape failure | `artifacts/sol/p3/sol-20261122-cuda-lock/test.json`, its 33 raw workers, and `prefix.json`, corroborated by `theory/evidence/phase3/sol_replication_audit.json`. The tree parents recover exactly. Int32 and int64 reject steps 284 and 278 and already lost whole nonzero kicks at steps 30 and 34. Both high-step native power-law fits outperform the registered exponential fits. |
| Phase 3 branching and GPU memory | The same locked manifest reports coverage, detector gates, native timing, and every allocator and offload comparison. The memory claim applies to full-model reversible updates versus dense GPU checkpoints. Pinned-host SGD reaches the same depth-eight solve rate and exhausts the tree 2.31× faster. All rewind-versus-PCIe time crossovers are left-censored on the tested grid. |
| Phase 3 exploratory output head | `theory/evidence/phase3/sol_head_control.json` and `sol_replication_audit.json`. Head-only checkpointed SGD solves 27/27 with 113.1 MiB peak allocated GPU and 512 KiB of live checkpoints. The trainable subset and backward work both change. The seeds were previously scored. |
| Analytical scale crossover | Figure 4 evaluates dense representation widths with one saved state per unresolved fork. The projected tape follows the default spine and its IDs are charged to device memory. The measured tape resides on host memory. The curves project bytes under this model. |

The numeric extractor verifies 58 SHA-256-pinned inputs, including linked validation receipts and all Phase 3 worker hashes. Its 140 macros carry source selectors or aggregation definitions. The six table fragments also identify their sources. The figure script uses the same pinned reader. [Input pins](../paper/scripts/source_pins.json) · [Evidence restorer](../paper/scripts/restore_evidence.py) · [Macro definitions](../paper/macros_verified.tex) · [Table builder](../paper/scripts/build_tables.py) · [Figure builder](../paper/scripts/build_figures.py)

## File manifest

| Directory | Files |
| --- | --- |
| `paper/` | `.gitattributes`, `.gitignore`, `README.md`, `main.tex`, `main.pdf`, `manuscript-source.zip`, `macros_verified.tex`, `references.bib` |
| `paper/sections/` | `abstract.tex`, `introduction.tex`, `related.tex`, `mechanics.tex`, `protocols.tex`, `adaptation.tex`, `search.tex`, `discussion.tex`, `reproducibility.tex` |
| `paper/tables/` | `phaseone.tex`, `recovery.tex`, `retention.tex`, `transformer.tex`, `branching.tex`, `systems.tex` |
| `paper/figures/` | Both `.pdf` and `.png` for `fig1_precision_horizon`, `fig2_continual_falsification`, `fig3_systems_pareto`, and `fig4_scale_crossover` |
| `paper/scripts/` | `build_tables.py`, `build_figures.py`, `restore_evidence.py`, `source_pins.json` |
| `paper/verification/` | `clean-autocrlf-true.json`, `clean-autocrlf-false.json`, `clean-autocrlf-true.tex-log.txt`, `clean-autocrlf-false.tex-log.txt`, `archive-controls.json` |
| `comms/` | `sol_to_astra_paper.md` |

## Build and sanity checks

Tectonic and BibTeX compiled the final PDF with no warnings, overfull or underfull boxes, unresolved references, or citations. I checked the four figures individually and reviewed every rendered manuscript page. PDF text extraction preserved the PowerShell quotes and line-continuation characters. No NaN placeholders or missing glyph markers appeared.

All 14 citation keys resolve against publisher, conference, or author-deposited records. No `% VERIFY_BIB` items remain. The appendix gives the phase-specific code revisions, recorded hashes, hardware and backend scope, exact PowerShell commands, and commands for the archived post-hoc controls. The generators use archived evidence only. Protected paths are byte-identical to the starting checkout.
