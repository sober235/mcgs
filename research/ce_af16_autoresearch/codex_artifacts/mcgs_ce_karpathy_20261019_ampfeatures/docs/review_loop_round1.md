Actionable findings: **0**. Unresolved findings: **0**. One verified P2 resume defect was corrected during this review and independently rechecked below.

This fresh-context review covers only `autoresearch_loop.py`, `tests/test_loop.py`, and their use of the existing public `run_gpu.py` / `evaluate.py` interfaces. I read `AGENTS.md`, `program.md`, `docs/design.md`, `docs/plan.md`, and the prior production review. The repository has no `.codegraph/` directory. The five production files were not modified and their hashes still match `docs/review_round3.md`. No GPU work or clinical scoring/GT loading was performed by this reviewer.

## Resolved finding

**P2 — Mutable campaign state could weaken the resumed keeper anchor.** In the initially reviewed driver, `execute()` read the keeper score again from `campaign_state.json` while its saved `prior_incumbent` and campaign descriptor were not checked or used to recover that score (initial revision lines 284–314). Trigger: an active loop saved the real prior score `-0.11118128532897047`, then the mutable campaign file changed that field to `-0.12`. Impact: a later `-0.115` candidate could be marked keep despite losing to the true prior incumbent. This affected keep/progress reporting; the scorer's six strict whole-head success conditions remained separate.

Evidence: a temporary stdlib fixture with numerical/review operations stubbed reached `score_run()` with the saved and current anchors disagreeing. The author corrected this without pinning the entire operational campaign file: `load_prior()` at final `autoresearch_loop.py:278` takes the original prior and `0.0001` threshold from saved loop state on resume and verifies its source descriptor. Scored journal records also bind that complete prior and threshold, and existing records must agree before recovering `best` (`autoresearch_loop.py:321`).

Independent recheck: a temporary CPU fixture exercised final `execute()` through a baseline journal append, then stopped before any numerical stage. After campaign status, prior score (`-0.12`), and threshold (`0.001`) changed, the appended baseline still recorded incumbent `-0.11118128532897047`, threshold `0.0001`, and the original prior object. A `-0.115` objective was consequently discarded. The bundled regression also rejects changed prior-source bytes.

## Scope checks

- Completed numerical outputs are reused only through `validate_observed_gate()` plus exact requested parameters and directory checks. Existing incomplete output or score directories are refused, preserving their contents.
- The queue declares matched shared/explicit two-step smokes and 500-update trials with the frozen parameters. It uses the public supervisor CLI and validates stage gates before a new numerical launch. No production scoring formula or training path changed.
- Whole-head PSNR/SSIM/NMSE and their six normalized margins determine ranking. Plane success cannot substitute for whole-head success. Candidate-array hashes must equal completed-gate artifacts and the scorer's input hashes; the reviewed evaluator hash and all recorded input bytes are checked.
- JSONL is the canonical journal. Duplicate record IDs, incomplete tails, and foreign TSV content are refused; a missing/prefix-only TSV projection is recovered without duplicate rows. Reused scored entries verify objective, completed gate, and recorded scoring-report bytes.
- Cancellation sends TERM to the owned supervisor PID and permits 120 seconds for its existing cleanup/reap path. The CPU process fixture observes the numerical child exit while an unrelated process remains alive. The fallback records unconfirmed cleanup instead of claiming it succeeded. Failures/cancellation are journaled separately from scientific discard and stop the family for root diagnosis.
- A strict whole-head win is labeled provisional. Final state keeps `success=false` and explicitly requires independent repeat plus matched GS-single before attribution; an ordinary completed family does not claim the requested superiority.

## Observed validation and binding

Independent command:

```text
/home/congcongliu/anaconda3/envs/GSTMI/bin/python -B -m pytest -q tests/test_loop.py
.................                                                        [100%]
17 passed in 0.20s
```

| File | Final SHA256 |
| --- | --- |
| `autoresearch_loop.py` | `d3a992826032b794feb34fbc1cfe2eb7ebe7ad48089a1d541fee63221f415bf2` |
| `tests/test_loop.py` | `5823c7f44580f439c0ccb82035e1453b42627a124e3dcb226950d69db3742c05` |

These checks cover CPU orchestration behavior, not the numerical quality of future trials. This review does not rerun MRI production review, score clinical arrays, or certify a complete GPU family run. There is no remaining code-review blocker for this bounded queue under the unchanged production binding and its required fresh stage evidence.
