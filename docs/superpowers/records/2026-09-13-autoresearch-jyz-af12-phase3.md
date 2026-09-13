# jiangyizhou AF 12, phase 3 — the coupling term that was missed for two phases (2026-09-13)

Branch `autoresearch-jyz-af12-phase3` (from `autoresearch-jyz-af12-phase2` at 721b80e), worktree
`/data0/congcong/code/GS/mcgs_af12_wt`. Program and round log: `autoresearch_jyz_af12_p3.md`. Ledger:
`autoresearch_jyz_af12.tsv`, `p3_` prefix. Runner: `autoresearch_jyz_af12_p3.sh`. Archive:
`Results_20260908/T1FLAIR_AF12_v3_jiangyizhou/runs_p3_loop/` (4.3 GB, self-contained).

## What the user asked, and why the phase existed

Two decisions opened it. First: **GS-single counts as a comparison method**, not as our own ablation. Under that
gauge phase 2's result read differently — T1 was first over every arm on 335 of 335 full-brain slices, but **FLAIR
was 0.037 dB behind GS-single** and behind on 310 of 332 slices. Second: after reading the parallel AF16 line's
phase-3 program, **borrow from it**.

Before that, two of the user's own hypotheses had been tested and closed: giving FLAIR less data (a fresh mask at
actual AF 10.56 against FLAIR's own support) left joint − single flat at −0.035, and the AF16 line had already
measured contrast-private residual Gaussians as null. So the gap was neither a sampling effect nor a missing
private geometry.

## Result

**P14 = P13 + `--lambda_edge 0.001 --lambda_ptv 0.01`.** Nothing else changed.

| head cell | T1 | margin over every other arm | FLAIR | margin |
|---|---|---|---|---|
| PSNR | **37.015** | **+0.150** | **36.360** | **+0.036** |
| SSIM | **0.9516** | **+0.0020** | **0.9415** | **+0.0005** |
| HFEN | **0.0928** | **+0.0041** | **0.0935** | **+0.0008** |

The binding arm in all six cells is GS-single; L1-ESPIRiT trails by +0.589 / +0.602 dB. **Joint − single: T1
+0.066 → +0.150, FLAIR −0.038 → +0.036** — the sign flip the phase existed for. Seed floor 0.0037 / 0.0026 dB.

**Held-out, pre-registered before the run**: P14 unchanged on huangyuwei gives FLAIR joint − single **+0.030**
(SSIM +0.0006, HFEN +0.0005) and **+0.670 / +0.668 dB** over that topic's strongest classical arm. The mechanism
transfers. This is the first held-out evidence this line has ever produced.

## The two levers, and why we had missed them

**Common-edge prior, λ 0.001.** A symmetric cross-contrast gradient-direction prior that already existed in this
branch's code — `pair_prior.py` and its call site are byte-identical to the AF16 branch's (verified against the
shared ancestor), so their λ transferred without rescaling. Bracketed independently here at 1500 iterations:
0.0005 +0.032/+0.027, **0.001 +0.049/+0.041**, 0.002 +0.043/+0.031, 0.005 −0.152/−0.165. Single-peaked at the same
value the AF16 line found. **Phase 1 of this line had tested only λ 0.02 — twenty times the optimum — and recorded
"loses by 1.33 dB".** That verdict was about the strength, not the term, and it closed the lever for two phases.

**Phase-only TV, λ 0.01.** Ported from `autoresearch-jyz-af16-phase2` (`paired_gs/losses/phase_tv.py`, trainer hook,
`--lambda_ptv`), 6 unit tests, default path bit-identical (first-iteration losses match a pre-port run to every
digit). Its dose curve here is **steeper than theirs** — T1 costs 0.009 / 0.040 / 0.085 dB at 0.01 / 0.02 / 0.03
against their 0.005 / 0.033 / 0.05 — because our sharpness starts 0.03 higher, so the same push buys less at the
same price. **We take 0.01 where they take 0.03: the lever is borrowed, the number is not.**

The two are complementary, which is why the combination beat either alone: the edge term buys PSNR and costs a
little sharpness, the phase term buys structure and sharpness and costs a little PSNR. Combination at 1500:
T1 +0.041 / FLAIR +0.046 with all six cells and both sharpnesses up — the best row of the phase.

## Why it works

The common-edge term is the only lever in the box that is *inert in a single-contrast run*: a single-contrast
configuration feeds the same image into both channels, so the cross-gradient misfit it penalises is identically
zero and the term adds nothing. It therefore lifts the joint arm only — which is exactly the asymmetry needed to
flip the sign of joint − single. The AF16 line saw the same thing (−0.003 → +0.042 there).

## Method notes worth keeping

- **A lever judged at one strength is not judged.** The single most consequential error of this line was recording
  a coupling term as a loser after testing it twenty times too strong. Before closing a term, bracket it.
- **Read the neighbouring line's program before proposing a structural fix.** The private-geometry idea proposed in
  chat had already been measured as null next door; reading their round log first saved a day and redirected the
  phase to the lever that worked.
- **Borrow the lever, re-measure the number.** Both borrowed λ were re-bracketed here. One agreed with theirs, one
  did not, and the one that did not would have cost 0.075 dB of T1 if copied.
- **Verify with a different tool than the one that failed.** Twice this session a command did not do what its syntax
  suggested — a `pgrep -f` pattern written in BRE against an ERE matcher, and a zsh word-splitting assumption in a
  launch loop — and both times the first check used the same broken construct and reported success. The zsh one put
  four bogus `crash` rows in the ledger; it was caught by comparing process start times, log write times and ledger
  write times, which disagreed.
- **Never edit a launched script in place** (carried over from phase 2 and respected here: the phase-TV port went
  into the trainer while four trainers ran, which is safe because Python imports at start, but the *runner* scripts
  were never touched mid-flight).

## What is open

Two subjects, not a cohort. The classical λ on both topics were selected on huangyuwei, so the comparison arms are
not disadvantaged there. Magnitude gauge throughout; the complex-gauge caveat of the earlier phases is untested for
P14. Capacity at 4.8M was a keep at 1500 (FLAIR +0.051, T1 flat) and its 4000-iteration promotion was still running
when this record was written — if it holds, FLAIR's thin margin gets thicker.
