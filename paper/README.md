# Manuscript build

Use `manuscript-source-polished.zip` to build the current manuscript. The earlier `manuscript-source.zip` and `verification/clean-autocrlf-*` receipts identify the independently audited release before the editorial pass. Both packages use the original Git history through `834128d6870fab5dcd2a4c6a39c91d3750632602`. Use Python, Matplotlib, Git, and Tectonic or a conventional LaTeX toolchain.

## Clean checkout and evidence staging

In PowerShell, set `$repo` to a repository that contains the original revision and `$package` to the absolute path of the source ZIP. Use a fresh checkout and a separate staging directory. Either `core.autocrlf` setting is supported.

```powershell
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('mttt-' + [guid]::NewGuid())
$checkout = Join-Path $scratch 'checkout'
$staged = Join-Path $scratch 'evidence'
New-Item -ItemType Directory -Path $scratch | Out-Null
git clone --no-checkout --no-hardlinks $repo $checkout
if ($LASTEXITCODE) { throw 'Clone failed' }
git -C $checkout config --local core.autocrlf true  # or false, before checkout
if ($LASTEXITCODE) { throw 'Git config failed' }
git -C $checkout checkout --detach 834128d6870fab5dcd2a4c6a39c91d3750632602
if ($LASTEXITCODE) { throw 'Checkout failed' }
Expand-Archive -LiteralPath $package -DestinationPath $checkout
Set-Location $checkout
python -B paper/scripts/restore_evidence.py --output $staged
if ($LASTEXITCODE) { throw 'Evidence staging failed' }
$env:MTTT_EVIDENCE_ROOT = (Resolve-Path -LiteralPath $staged).Path
```

The original manifest is `scripts/source_pins.json`, with SHA-256 `d52b5df5f903361eb12e85d039da2f93083d5d58ff5525294f8ae085859834d2`. The restorer reads raw Git blobs from the specified revision. A blob or its CRLF reconstruction is accepted only if its SHA-256 equals the original path's pin. The staged directory stays separate from the checkout. The JSON report lists the selected representation and digest for every input. The generators recheck the bytes and audit-to-run links before reducing them.

## Generate and compile

Run from the checkout root with `MTTT_EVIDENCE_ROOT` set as above.

```powershell
python -B paper/scripts/build_tables.py
if ($LASTEXITCODE) { throw 'Table generation failed' }
python -B paper/scripts/build_figures.py
if ($LASTEXITCODE) { throw 'Figure generation failed' }
$env:SOURCE_DATE_EPOCH = (git show -s --format=%ct 834128d).Trim()
Push-Location paper
tectonic --keep-logs main.tex
$texExit = $LASTEXITCODE
Pop-Location
if ($texExit) { throw 'LaTeX build failed' }
```

With a conventional TeX installation, `latexmk -pdf main.tex` from `paper/` builds the same source. `macros_verified.tex` identifies the raw field or aggregation for each numeric macro. Table fragments carry matching source comments. Figure 4 evaluates closed-form byte counts under its stated assumptions. The generators use archived data and require no model runs.

## Historical source for archived controls

The post-hoc controls expect the archived receipts at their original relative paths and import the historical evaluator and adapter. Appendix A gives commands that restore the original pinned data to two separate scratch roots. Overlay only the listed source paths with `git --no-replace-objects -c core.autocrlf=false -c core.eol=lf archive`. These command-local settings preserve the historical source tree hashes even when the current checkout uses CRLF. For the Phase II control, use revision `e4635c638bd53e3d36ffa5db9603040195ca924e`. For the Phase III controls, use `834128d6870fab5dcd2a4c6a39c91d3750632602`. Run each control from its own staged root with `PYTHONPATH` set to that root's `src`.
