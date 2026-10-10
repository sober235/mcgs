# SDD ledger — plan: docs/superpowers/plans/2026-10-09-sg2x2-shared-geometry-ablation.md

Worktree /data0/congcong/code/GS/mcgs_sg2x2_wt, branch exp/shared-geometry-2x2-ablation-20261009, start 9df60e7.
Spec: docs/experiments/2026-10-09-mcgs-shared-geometry-2x2-ablation.md (reachable; the binding authority).
Execution method: subagent-driven (user, 2026-10-09).

## Pre-flight scan (2026-10-09)

| pair / task | produces vs consumes | found |
|---|---|---|
| T1 cells.py ↔ T4 run_sg2x2.sh | CLI prints one arg per line; launcher passes --init/--iters/--warmup; cells accepts them | ok; launcher's `mapfile < <(...)` could not see a failing exit code → Ruling R2 |
| T1 ↔ T5 manifest | `resolved_switches`, `CELLS[c].reg/config/trainer/contrast` | ok |
| T1 ↔ T3 single trainer | cells set `--trainer single`, zero-valued cross terms via --set; trainer refuses only > 0 / True | ok |
| T1 C/D ↔ topic yaml `edge_tv: true` | `--set loss.edge_tv=false` applied last in cycle.py build_cfg | ok |
| T2 fixture ↔ T3 tests | `base_cfg(pair, tmp, exp, reg, single, init_mode, npts, seed, iters)`; T3 calls match | ok; fixture test created subdirs after writing → Ruling R3 |
| T2 init branch ↔ T4 make_init_support | npz key `support`, bool, volume shape | ok |
| T3 cycle.py single volumes/json ↔ T6 table | keys rec/gt/zf; json plain only; joint keys recT1/gtT1/recFL/gtFL | ok |
| T3 single log line ↔ T5 audit | single logs R_ptv, joint does not; audit uses `.get(k, 0)`; start-up prefixes identical in both trainers | ok |
| T4 launcher ↔ T5 manifest args | `--seed --npts --init --support` | ok |
| T4 smoke expectation ↔ T2 dilation | expected `in_own_zf_support=1.0000` but the support is dilated 3 voxels | wrong expectation → Ruling R4 |
| T0 ↔ T7 regress compare | iter lines and score lines present in cycle.py stdout log | ok (verified on the finished reference run) |
| T3 equivalence test (self-consistency) | asserted parameter equality after one Adam step; Adam's eps makes parameters with gradients ~1e-8 differ by up to lr | fragile → Ruling R1 |
| T8 queue ↔ T4 launcher | queue line `EXP CELL ITERS SEED` → `run_sg2x2.sh EXP card CELL ITERS SEED` | ok |
| workspace | `.superpowers/` not ignored on this branch (no tracked files under it) | Ruling R5 |

Rulings (all applied to the plan text before Task 1 was dispatched):
- Ruling R1: the single-vs-duplicate-slot equivalence is tested on the gradients after the one backward pass (joint gradient = one positive multiple of the single gradient per tensor, geometry multiple = 2 × density multiple, residual < 1e-4), not on parameters after an Adam step — Adam's eps makes near-zero-gradient parameters differ by up to lr; the gradient statement is exact. Cost if wrong: a looser test than the parameter check (none in practice; the log-component checks remain).
- Ruling R2: run_sg2x2.sh validates CELL with a case guard and refuses when cells.py produced no arguments — otherwise an unknown cell would launch the default winner_v2 config. Cost if wrong: none.
- Ruling R3: the data-independence test creates its two subdirectories before writing the fixture pairs. Cost: none.
- Ruling R4: on the development subject `in_own_zf_support` is expected ≥ 0.85, not 1.0, because the frozen support is the own support dilated by 3 voxels. Cost if wrong: a wrong acceptance line in the smoke step only.
- Ruling R5: Task 1 appends `.superpowers/` to `.gitignore` (one additive line; the knee line kept its ledger ignored the same way) so manifests do not list the ledger as dirty. Cost if wrong: one extra ignored path.

Minor plan-vs-spec note: the spec's §4 equal-budget per-arm λ search is deliberately out of scope (plan Global Constraints); the main table uses the identical frozen recipe for A and B (spec §3).

## Task 0: baseline and reference smoke — complete (controller, no code)

- Baseline tests in this worktree (2026-10-09, card 4): `223 passed, 1 warning in 445.81s` (results_arc/baseline_tests_20261009.log).
- Reference smoke `sg2x2_regress_before` (card 3, unmodified tree 9df60e7, results_arc/sg2x2_regress_before.log), the four lines Task 7 compares against:
```
[iter     1/50] loss=1.3397e-01 dc_p=7.8237e-02 dc_c=5.4249e-02 R_geo=6.1685e-03 R_dtv=0.0000e+00 R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 R_wav=7.8157e-03 R_nl=0.0000e+00 balance_violation=False
[iter    50/50] loss=4.5198e-02 dc_p=2.0174e-02 dc_c=2.0549e-02 R_geo=1.1108e-02 R_dtv=0.0000e+00 R_edge=0.0000e+00 R_vtv=0.0000e+00 R_dtvp=0.0000e+00 R_wav=2.8066e-02 R_nl=0.0000e+00 balance_violation=False
T1    psnr=37.579 ssim=0.9078 nmse=0.0227 hfen=0.7701 e_unobs=0.6379 e_unobs_xz=0.6379 head_psnr=33.807 head_ssim=0.9201 head_hfen=0.1491 N=1.099 kz_energy=0.000e+00 sharp=0.803 grain=0.0104 head_psnr_cplx=30.631 cos_phase=0.9888 comp=29.696 beat=True
FLAIR psnr=36.660 ssim=0.8922 nmse=0.0253 hfen=0.8195 e_unobs=0.7606 e_unobs_xz=0.7605 head_psnr=32.772 head_ssim=0.8968 head_hfen=0.1420 N=0.945 kz_energy=0.000e+00 sharp=0.703 grain=0.0108 head_psnr_cplx=30.002 cos_phase=0.9896 comp=22.924 beat=True
```
- Dev-subject support probe (CPU, scratch): union-ZF support 0.4273 of the box, bbox = whole volume, inscribed ellipsoid covers 0.8954 (margin 1.0) — the reason the plan uses a support file, not an ellipsoid.

Task 0: complete (no commits; artefacts under results_arc/, gitignored)

## Task 1: cell table
- implementer (haiku) DONE_WITH_CONCERNS, commit a3d50df; concerns 1,2,3,5 are the content of Tasks 2–4 (expected); 6 is plan-mandated (minor, deferred).
- Ruling R6: concern 4 — the audit of Task 5 must also require `lambda_nl`, `lambda_nl_plain`, `lambda_nl_contrast` to resolve to 0 in every cell (fixed expectation inside `_resolved_from_cfg` / `audit`, so `resolved_switches` and Task 1's tests stay as they are). Cost if wrong: one extra zero check.
- note: the implementer could not write under .superpowers/ (its Write was refused); the controller saved its report verbatim to task-1-report.md.
- review (sonnet): spec ✅, quality Approved, 0 Critical / 0 Important / 6 Minor; ⚠️ items (kspace_freq_adaptive not emitted by the table; support_file and --trainer not yet consumed) are sequenced to Tasks 2–5.
- Ruling R6 (extended after the review): Task 5's audit requires, as fixed expectations in every cell, `lambda_nl = lambda_nl_plain = lambda_nl_contrast = 0`, `guide_path_plain` / `guide_path_contrast` unset, `kspace_freq_adaptive = true` and `kspace_high_weight = 2.2` in the resolved config, in addition to `resolved_switches`. Cost if wrong: a few extra checks.
- Task 1: minor (deferred): docstring says the shared knobs live in "COMMON" but the block is inline in cell_args (scripts/sg2x2_cells.py:14-15).
- Task 1: minor (deferred): the plan's path label was copied as line 1 of both new files, pushing the shebang to line 2 (dead) — later tasks are told to keep the shebang first; fix in the final-review wave.
- Task 1: minor (deferred): recipe constants exist twice (argv strings in cell_args, floats in resolved_switches) with no cross-check test; Task 5's audit cross-checks them at run time.
- Task 1: minor (deferred): tests never pass non-default iters/warmup; num_private_points / adaptive_control / lambda_nl / bad init unasserted; `len(seen)==1` tautological; `_sets` keeps the last duplicate key.
- Task 1: minor (deferred, plan-mandated): `--support` required with `--init head`; `resolved_switches` annotated as bare dict.
Task 1: complete (commits 0c5ff2b..a3d50df, review clean)

## Task 2: frozen init support
- implementer (sonnet) DONE_WITH_CONCERNS, commit 12ff5f5 (+299/-0; joint_trainer.py insertion-only, 0 removed lines). Step 6: 6 passed, 1 deselected (card 4); existing test_init_cloud + test_tile_voxelizer_invariance 10 passed.
- Ruling R7: the frozen file `configs/sg2x2/init_support_af16_jyz.npz` is made with `--dilate 0` (the implementer measured dilate 1/2/3 → support 0.4811/0.5431/0.6035 vs 0.4273 undilated; a 3-voxel ring would park 29 % of the cloud outside the own support and move the operating point). With dilate 0 the support equals what init_mode=head draws from on this subject, so cell A's cloud is the registered recipe's own cloud and in_own_zf_support reads 1.0 for every cell. Plan Task 4 Step 4 / Step 7 and Task 8 Step 3 updated; R4's "≥ 0.85" is superseded by "= 1.0000". Cost if wrong: a less robust support for a future same-grid subject (none planned; the 16-pair set is another grid and needs its own file).
- Task 2: minor (deferred): scripts/sg2x2_make_init_support.py reads the plain slot's mask for both contrasts; correct for this topic (one mask), should assert both mask_path values are equal before it is reused on an asymmetric config.
- note: the implementer once ran the two existing test files on a card another session had just taken; it re-ran on idle card 4 and recorded that run.
- review (sonnet): spec ✅, quality Approved, 0 Critical / 1 Important (plan-mandated) / 7 Minor; both named risks hold (downstream of the branch identical to head; support_file draw == head draw on the same support, verified on CPU).
- Ruling R8: Important 1 (init_cloud.py:127-141 repeats sample_positions' decode/jitter/clamp block, no test ties them) — the duplication stands (the additive-only rule forbids refactoring `sample_positions`, which every registered run goes through); the fix is the reviewer's CPU parity test (head draw == support_file draw on head_support(mag), N below and above the support size, jitter on and off). Cost if wrong: a 10-line test too few; none for the registered lines.
- Task 2: minor (deferred): the log test pins only the format of `in_own_zf_support` (a mutant computing it from the file's support would pass).
- Task 2: minor (deferred): maker's overwrite guard checks `out` before savez appends `.npz`; an `--out x` would overwrite `x.npz` then crash on read_bytes; guard runs after the 19 s compute. Fix: require the `.npz` suffix and check right after argparse.
- Task 2: minor (deferred): maker applies the plain slot's mask to both contrasts (correct for this topic); assert equality or load each; `--config` resolves against ROOT, `--out` against CWD; unclosed file handle.
- Task 2: minor (deferred): same-cloud identity rests on the CUDA device generator (same as head mode); a positions checksum in the start-up line would let the manifest assert cell-to-cell identity.
- Task 2: minor (deferred): `match="shape"` is loose; no test of the replacement path / jitter=False / empty-path ValueError; `support_frac` means the file's fraction in the new line but the subject's own in the head line.
- Task 2: fix round 1/5 dispatched (1 open: parity test)
- Task 2: fix round 1/5 (1 addressed by the implementer's account, 0 open — parity test; commits 12ff5f5..5c13c9d); plan edit for R7 committed by the controller as the next commit (docs only).
- re-review (haiku): the parity-test finding ADDRESSED (tests/test_sg2x2_init_support.py:39-49, three cases, torch.equal on vox and pos); no new breakage.
Task 2: complete (commits a3d50df..5c13c9d, review clean; plus the controller's plan-edit commit 0cbf3ff)

## Task 3: SingleContrastTrainer + arc/cycle.py --trainer
- BASE 0cbf3ff; implementer sonnet dispatched. Expected count in the brief's Step 5 is now 26 (18 new + 8 in test_sg2x2_init_support.py after the parity test), not 25.
- implementer (sonnet) DONE_WITH_CONCERNS, commit bfdfcc5 (3 files); 26 passed on card 3; gradient-ratio residuals ~1e-7, ratios exactly 2:1; cycle.py joint path A/B-identical on the toy pair.
- Concern 1 is a real pre-existing bug, verified by the controller: joint_trainer.py:991/993 read `self._edge_kappa_c` / `_p` every iteration but they are only set inside `if self._edge_tv:` (924-928), so any joint run with `loss.edge_tv=false` dies with AttributeError at iteration 1 since f413c26 (2026-09-26). Cell C needs that path. Side effect for the user: the iso-TV reproduction commands in the project memory (configs without edge_tv) have been broken since 09-26.
- Ruling R9: fix it additively inside Task 3's fix round — two lines `self._edge_kappa_p = None` / `self._edge_kappa_c = None` inserted right after `_kappa_refresh = 0` in train(), before `if self._edge_tv:` (the block overwrites them when edge-TV is on, so every registered run is unchanged; None is never dereferenced because every kappa consumer is gated on edge_tv / lambda_ptv). Plan Global Constraints updated. Cost if wrong: the regression gate (Task 7) catches any change to the default path.
- Ruling R10: the brief's test workaround (edge_tv forced True on the duplicate-slot proxy for reg=False) is removed so the reg=False parity test runs the real edge_tv=false joint path and is the regression test of the bug; Task 4's smoke covers all four cell kinds (A, C, B_T1, D_FLAIR). Plan Task 4 Step 7 updated. Cost: ~8 extra GPU-minutes of smoke.
- also in this fix round: the fixture's loss key `tv_weight` → `lambda_geo` (test-only) to silence the 5 DeprecationWarnings the implementer reported.
- Task 3: minor (deferred): `--trainer single --prewhiten_only contrast` raises KeyError (no cell uses it).
- Task 3: fix round 1/5 dispatched (pre-review: R9 fix, workaround removal, fixture key)
- Task 3: fix round 1/5 (3 addressed — R9 kappa init (+5 lines, 0 removed), workaround removed, fixture lambda_geo; 0 open; commits bfdfcc5..bef4d4a); CPU evidence: pre-fix trainer raises at :991 with the final fixture, fixed trainer completes; edge_tv=true logs and tensors bit-identical before/after.
- Task 3: minor (deferred): no test exercises the `tv_weight` compatibility branch any more; a base yaml carrying `loss.lambda_geo` would shadow `--lambda_geo` (the sg2x2 configs copy the topic yaml, which carries `tv_weight`, so the real cells are fine — Task 4's config test pins that).
- controller committed the plan edits for R9/R10 (docs only). Task review (opus) dispatched over 0cbf3ff..bef4d4a.
- review (opus): spec ✅, quality Approved, 0 Critical / 1 Important (plan-mandated duplication of the joint's per-channel logic; fix = broaden the parity tripwire) / 7 Minor; named risks (a)-(d) all hold: single arm reproduces the joint's per-channel terms, cycle.py joint path unchanged, None kappas never dereferenced, inherited methods channel-agnostic.
- Ruling R11: the fixture edit in bef4d4a (tests/sg2x2_fixture.py tv_weight→lambda_geo) is in scope — the fixture is this plan's own test code (Task 2), not a pre-existing file. Cost: none.
- Ruling R12: in single cells `in_own_zf_support` measures the union file against that one contrast's own ZF support, so it reads < 1.0 there (A/C read 1.0000); Task 4 Step 7 and Task 8 Step 3 expectations amended to record the values as diagnostics. Cost if wrong: a wrong acceptance line only.
- Ruling R13: the fix round for the Important finding also adds a parity variant in which lambda_ptv dominates (reviewer's Minor 1), because it is the same parametrised test and the phase-TV term is one of the three regularisers the A/B comparison relies on. Cost: one extra parametrisation.
- Task 3: minor (deferred): `loss.get(k) not in (None, "", False)` lets `kspace_high_weight_plain: 0` through (0 == False); `lambda_wav_plain > 0` with `lambda_wav == 0` silently disables the wavelet; `R_wav` is logged unweighted; refuse `lambda_wav_plain != lambda_wav`.
- Task 3: minor (deferred): refusal tests carry an unnecessary @cuda; cycle test asserts no "no TSV row"; it writes into results_arc/ (checkpoints) every run; falls back to card 0 when CUDA_VISIBLE_DEVICES is unset; aliasing test misses csm_fit_c / kspace_dc_c / _dc_norm_c.
- Task 3: minor (deferred): `--trainer single --prewhiten_only contrast` KeyError (should be ap.error); inherited export_results() raises under the single trainer (cycle.py never calls it).
- Task 3: minor (deferred, process): a diagnostic script used tempfile.TemporaryDirectory under .superpowers/ and removed its own scratch dir; future implementers use persistent scratch dirs.
- Task 3: fix round 2/5 dispatched (1 open: broaden the parity test — kappa_source render, refresh_iter 1, curriculum, tv_weight-only, lambda_ptv-dominant)
- Task 3: fix round 2/5 (1 addressed by the implementer's account — parity test over 7 variants, each drift probe trips an assertion; 0 open; commits bef4d4a..b257d3e; 31 passed on card 0). Plan edit for R12 committed by the controller (docs only).
- re-review (sonnet, re-dispatched after the session restart): parity-test finding ADDRESSED (7 variants, both configs, assertions unchanged, 31 passed); no new breakage.
Task 3: complete (commits 0cbf3ff..b257d3e, review clean; plus the controller's plan-edit commits 59b0ac0, 80c03cd)

## Task 4: configs, init support, launcher
- BASE 80c03cd; briefs 4-8 regenerated from the amended plan (R2, R7, R10, R12); implementer sonnet dispatched.
- implementer (sonnet) DONE, commit 8c13b44; support md5 310a6520631a330e59cb5deb390e9035, support_frac 0.4273; 4 tests passed; smokes A/C/B_T1/D_FLAIR DONE on card 1 (4m55 / 4m44 / 2m31 / 2m27); in_own_zf_support 1.0000 (A, C), 0.9697 (T1 single), 0.9206 (FLAIR single).
- note: the delete-guard hook blocked heredoc writes containing the launcher test's marker-removal call; the implementer used the Write tool. That test still removes its own marker file at run time — flagged to the reviewer through the constraints block (user rule: nothing is deleted, scratch included).
- review (sonnet) dispatched over 80c03cd..8c13b44.
- review (sonnet): spec ✅, quality Approved; 1 Important (plan-mandated: the launcher test removes the marker file it created, against the user's no-deletion rule) / 2 Minor.
- Ruling R14: the launcher test keeps one permanent marker `results_arc/sg2x2_refuse_test_score.json` (created when absent, never removed; results_arc is gitignored) instead of removing it after the assertion; the `tmp_path` fixture is dropped from that test. Cost if wrong: one 2-byte file in a gitignored directory.
- Task 4: minor (deferred): `run_sg2x2.sh` positional handling silently treats `--set` as ITERS when ITERS/SEED are omitted before extra args; fail early when `$4`/`$5` start with `--`.
- Task 4: minor (deferred): the `mapfile` guard cannot see a non-zero exit of sg2x2_cells.py (only empty output); capture with `out=$(...) || exit` instead.
- Task 4: fix round 1/5 dispatched (1 open: R14)
- Task 4: fix round 1/5 (1 addressed by the implementer's account — permanent marker, nothing removed; 0 open; commits 8c13b44..5275f90; 4 passed twice).
- re-review (haiku): R14 finding ADDRESSED; no new breakage.
Task 4: complete (commits 80c03cd..5275f90, review clean)

## Task 5: manifest + regulariser audit
- BASE 5275f90; implementer sonnet dispatched with R6 (extra fixed expectations: lambda_nl* 0, guide_path_* unset, kspace_freq_adaptive true, kspace_high_weight 2.2) and R10 (audit all four smokes).
- implementer (sonnet) DONE, commit c8447c8; 7 tests passed; the four smoke manifests all audit ok (A, C, B_T1, D_FLAIR). Concern (guide_path keys under `loss`?) verified by the controller: the trainer reads `loss.guide_path_<channel>` (joint_trainer.py, the `_loss_cfg.get(f"guide_path_{_ch}")` line) — correct.
- review (sonnet): spec ✅, Approved, 0 Critical / 0 Important / 4 Minor; mutant reads confirm last-iteration prior and wrong FLAIR wavelet weight both fail the audit.
- Task 5: minor (deferred, fix in the final wave): the `_KV` regex does not match `nan`/`inf`, and a missing R_* key reads as 0, so a NaN prior term would pass `loss_ok` (Task 8's separate NaN grep is the backstop); only first/last logged iterations are checked; no test for the [nl]/[guide]/[dtv-proj] start-up rejection.
Task 5: complete (commits 5275f90..c8447c8, review clean)

## Task 6: result table
- BASE c8447c8; implementer haiku dispatched (complete code in the brief).
- implementer (haiku) DONE, commit 53ed3a5; 2 tests passed; empty-prefix run prints dashes. Concern: the head-NMSE path through real volumes is untested until a `_s<seed>_` run exists (Task 8's pilot exercises it).
- review (sonnet): Needs fixes; 2 Important (plan-mandated code: `_mean_sd` silently drops missing seeds with no per-cell count; SD prints `± 0.000` at n=1) / 4 Minor.
- Ruling R15: both Important findings are fixed (the brief's formatting is superseded): a cell with fewer valid seeds than requested carries `(n=k/N)`, and a cell with one valid value prints `± n/a`; the existing assertions stay valid because a complete two-seed cell keeps the old form.
- Task 6: minor (deferred): np.load handles never closed; `row["contrast"]` would raise if a joint-named run were single; no test for the volumes/NMSE path or the C−D / FLAIR interaction cells.
- Task 6: fix round 1/5 dispatched (2 open)
- Task 6: fix round 1/5 (2 addressed by the implementer's account; 0 open; commits 53ed3a5..98f728a; 4 passed)
- re-review (haiku): both findings ADDRESSED; no new breakage. Out-of-scope note: an all-missing cell renders a bare `—` (no `(n=0/n)`), kept as is (the dash already says it).
Task 6: complete (commits c8447c8..98f728a, review clean)

## Task 7: regression gate
- BASE 98f728a; implementer sonnet dispatched. Expected suite count now 278 (223 + 9 + 8 + 23 + 4 + 7 + 4) — the implementer reports the measured count.
- implementer (sonnet) DONE, commit 911804e; REGRESSION PASS — after-smoke (card 7) bit-identical to the 9df60e7 reference on iter1/iter50 components and both score lines; suite 277 passed + 1 OOM flake (another session took card 7 mid-run; the test alone passes on card 5) = 278; joint_trainer.py removed lines 0; diff stat 3 files +102/-7 (the 7 deletions are cycle.py's permitted guard/move lines).
- review (sonnet): spec ✅ (compare re-run by the reviewer: REGRESSION PASS, all 12 metrics identical), Approved; one fragility noted (compare script KeyError-crashes with exit 1 on an incomplete log instead of a clear FAIL/4) → deferred minor; ⚠️ the 7 cycle.py deletions verified by the controller (listed below), the OOM flake re-run rests on the implementer's report.
- Task 7: minor (deferred): regress compare script should fail loudly (exit 4 with a message) on a missing iter/score line; several `[iter 50]` lines → last wins silently; file handle never closed.
Task 7: complete (commits 98f728a..911804e, review clean)

## Task 8: queue, record, STATUS/CLAUDE; pilot waits for the user
- BASE 911804e; implementer sonnet dispatched for the docs/queue part; the six pilot runs are NOT launched by the implementer.
- implementer (sonnet) DONE for steps 1/4/5, commit 53becc8, tag handoff/2026-10-09-sg2x2-pilot (= code complete, pilot pending); queue runner `bash -n` ok; no GPU job launched.
- review (sonnet): spec ✅, Approved, 0 Critical / 0 Important / 3 Minor (stale header comment in sg2x2_queue.sh documents the old wd line format; header lines 2/3/7 changed beyond the listed substitutions; STATUS quotes the ledger's test line without re-running).
- Task 8: minor (deferred): scripts/sg2x2_queue.sh line 3 header still shows `EXP ITERS AF ARM LAMBDA_EDGE`; the real format is `EXP CELL ITERS SEED`.
Task 8: complete for steps 1/4/5 (commits 911804e..53becc8, review clean); steps 2/3 (pilot launch + acceptance) wait for the user's go.

## Final whole-branch review
- MERGE_BASE 9df60e7 → HEAD 53becc8; package built; reviewer on the most capable model (fable).
- final review (fable): "With fixes" — 0 Critical / 4 Important (audit does not pin the frozen knobs nor argv vs resolved config; NaN/inf passes the audit; record/plan overstate raw-render metrics; core-test init source under-specified vs spec §4) / 8 Minor; triage: fix before handover = Important 1-4, NaN (Task 5 minor), maker `.npz` suffix guard (Task 2 minor), positions checksum (Task 2 minor), vram/npts back to the original point in cycle.py, stale comments, launcher→manifest end-to-end once; the rest stays deferred.
- Ruling R16 (final fix wave, one dispatch): fix Important 1, 2, 4 and Minors 6, 7, 8, 12 plus the `.npz` guard in code; Important 3 is DESCOPED (raw-render metrics are not produced: the record states it and the plan's self-review sentence is corrected) — saving a second volume per run and a table row is not needed for the pilot's question; Minor 9/10 are record notes (warmup inert; resource row via a jq line); Minor 5 is a record sentence (the "equals head init" claim rests on the smoke's identical iter-1 numbers, not on a pinned positions equality); Minor 11 (ledger copy into docs/superpowers/records/) is done by the controller at the end. Cost if wrong: the pilot's question is answered without raw-render metrics; they can be rendered post hoc from the checkpoints.
- final fix wave (sonnet): commits 1f6d00d (code/tests) + dfbdce4 (docs); 94 sg2x2 tests passed on card 0; five audits ok incl. the first launcher→manifest end-to-end run (sg2x2_smoke2_D_T1, 146 s on card 0); joint_trainer.py 0 removed vs 9df60e7; cycle.py exactly the 5 guard deletions. Scoped re-review dispatched.
- scoped re-review of the fix wave (opus): all 11 findings ADDRESSED; checks: joint_trainer.py 0 removed lines vs 9df60e7, cycle.py exactly the 5 guard lines, 46 manifest tests pass, five manifests audit ok. Out-of-scope: the end-to-end smoke ran on the then-uncommitted fix tree (its manifest lists the 12 dirty files; the pilot's manifests will show a clean head).
Final review: clean after one fix wave (commits 53becc8..dfbdce4). Workspace NOT deleted (user rule: nothing is deleted); the ledger is copied into docs/superpowers/records/ as the durable record.

## Pilot (launched by the user 2026-10-09 12:39; branch + tags pushed)
- 12:39-12:42 A/C/B_T1 launched on idle cards 0/1/3; 16:46 B_FLAIR on card 2 (done 18:34, 108 min); 18:34 D_T1 on card 2.
- The CE line's queue (ar_queue.py, fixed lanes, no occupancy check; the user's other session) later placed 44 GB jobs on cards 0/1/3. Both processes on each shared card then made zero progress for 2.5-3 h (our A/C/B_T1 logs last written 16:11/18:31/16:46; the CE logs 15:26/16:07/16:11), GPU at "100 %" util but 1 % memory util and 85 W — the mutual stall recorded earlier today as "共享卡会饿死邻居".
- Ruling R17 (18:44): stopped OUR three stalled runs only (rc=143 in sg2x2_runs.log; cycle.py cannot resume from latest.pth), per the house rule "撞卡只停自己的运行,换名重跑"; the CE jobs resumed at full speed immediately (182-229 W, 42-77 % memory util). Re-queued A/C/B_T1 as `sg2x2_pilot2_af16_s42_*` (appended to scripts/sg2x2_pilot_s42.txt; the runner re-reads it). For the table, the finished `sg2x2_pilot_af16_s42_{B_FLAIR,D_T1,D_FLAIR}` outputs will be symlinked under the `sg2x2_pilot2_af16` prefix. Cost if wrong: ~40 GPU-minutes of real compute lost per stopped run; nothing else.
- 21:13 queue drained; six cells done rc=0 (B_FLAIR/D_T1/D_FLAIR under sg2x2_pilot_af16, A/C/B_T1 under sg2x2_pilot2_af16, symlinked under the latter for the table). Acceptance: 6/6 audits ok, pos_sum 319918.032643 in all six, in_own 1.0000/0.9697/0.9206, no NaN/OOM, git_head 53d38db (git_dirty = the appended queue file on five runs).
- Sanity: cell A 35.644773 / 34.875728 vs the registered J0 (phase-3 recipe, lambda_edge 0, head init) 35.644656 / 34.875808 → +0.0001 / −0.0001 dB: the pipeline reproduces the registered joint arm and the support_file init reproduces the head init on this subject.
- PILOT result (n=1, NOT EVIDENCE): A−B +0.006 / −0.022 dB (T1 / FLAIR); C−D +1.563 / +1.049; interaction −1.557 / −1.071 → spec §5.3 row "A ≈ B 但 C > D": the per-channel priors replace the whole shared-geometry benefit on this subject. Table results_arc/sg2x2_pilot2_af16_table.md.

## After the pilot (2026-10-10, user: "continue")
- Ruling R18: "continue" = run the spec's preset next steps that need no new design: seeds 43/44 of the six cells on the development subject (spec §5.1.4) and, at the lowest queue priority, the §4 capacity supplement (B/D at 1.8M points per contrast, A/C reused at 3.6M); start the read-only core-test data inventory. Not started, because each needs a user decision or a new design: B's equal-budget own-λ search (needs a recipe-variant mechanism in cells.py and the audit), the multi-subject core test (mask rule, development subject on that grid, sample size). Cost if wrong: ~12 GPU-h on otherwise idle cards; pausable with `touch results_arc/.sg2x2_queue_pause`.
- Ruling R19: `scripts/sg2x2_queue.sh` now skips any card claimed by a running `*_queue.py --lanes ...` of another line (the CE line's fixed-lane queues launch without checking occupancy; yesterday's three-hour mutual stall). At 02:52 the CE queues claimed cards 0,1,2,4,5,6,7, so only card 3 is usable; others become usable as those queues end. Commit 03482b3. Cost if wrong: our queue waits longer for a card.
- New prefixes: `sg2x2_dev_af16_s{42,43,44}_<CELL>` (seed 42 = symlinks to the pilot runs) and `sg2x2_dev_af16_bd18_s{seed}_<CELL>` (B/D at 1.8M; A/C symlinked to the 3.6M runs). Tables: `scripts/sg2x2_table.py --prefix sg2x2_dev_af16 --seeds 42 43 44` and `--prefix sg2x2_dev_af16_bd18 --seeds 42 43 44`.
- 02:52 queue launched (`results_arc/sg2x2_queue_dev_seeds.log`, ALLOWED all eight, MAXRUN 4); first job sg2x2_dev_af16_s43_A on card 3. A read-only stall/failure watcher wakes the controller (stall = a trainer's log idle > 25 min).
- Core-test data inventory dispatched (read-only, CPU) → `docs/superpowers/records/2026-10-10-sg2x2-core-test-data-inventory.md`.
- 04:19: the CE line started two more fixed-lane queues (`cmp_jobs_zs_gpu3 --lanes 3` after our s43_A took card 3 at 02:52; `cmp_jobs_dip_tv_gpu5 --lanes 5` right after our s43_C took card 5 at 03:26), so both our runs share their card; this time both sides progress (ours at about half speed: A 2850/4000, C 2000/4000), no stall. All eight cards are now claimed by CE lanes, so our queue launches nothing more until a lane ends — intended behaviour of R19.
- Core-test inventory returned (report committed with this entry): 32 files intact; grid = pilot grid + zero padding; raw axis 0 is image domain; stored CSMs are per-readout-slice maps (use fresh ESPIRiT); 14 persons / 13 test candidates; the 790 pair = second scans; pilot support covers 88-99 % of other subjects' energy; decisions for the user listed in STATUS §2.4. Scratch of the inventory: /tmp/claude-1002/sg2x2_inventory (~25 GB, not deleted — user rule).
