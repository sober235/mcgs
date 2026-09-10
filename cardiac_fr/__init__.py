"""Free-running multi-echo 3D radial whole-heart data (Mackowiak et al., MRM 2023).

Phase 1 of docs/superpowers/specs/2026-09-10-cardiac-freerunning-phase1-design.md: reading the
Siemens twix files, the phyllotaxis trajectory, physiological gating, BART reconstruction and the
evaluation gauge. Nothing here touches the MC-GS code paths.
"""
from pathlib import Path

RAW_ROOT = Path("/data1/data1_congcong/data/3D/cardiac_freerunning_multiecho_5D")
DERIVED_ROOT = Path("/data2/congcong/data/cardiac_3d_cine_public/derived/megre")
TOPIC = Path("/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE")
FIGS = Path("/home/congcongliu/figs/cardiac_fr")
SUBJECTS = ("V07", "V03", "V04", "V08")

N_ECHO = 8            # free-running multi-echo GRE
N_SHOT = 1000         # interleaves
N_SEG = 22            # spokes per interleave; seg 0 is the SI self-navigation spoke
N_SPOKE = N_SHOT * N_SEG
N_SAMP = 224          # readout samples, 2x oversampled
N_COIL = 24
GRID = 112            # image matrix, FOV 220 mm
KEEP = slice(3, 221)  # readout samples kept (spec 5.2): 0-2 and 221-223 dropped
T_STEADY_MS = 10_000.0
