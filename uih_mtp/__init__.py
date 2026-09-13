"""UIH uMR raw-data reader and classical parallel-imaging tools for the pediatric MTP line.

Modules
  uih_raw   header-only index of a RAWDATA-V004 file, selective line reading, readout-oversampling removal
  grappa    3D GRAPPA for PE-only acceleration with an ACS block (torch, CPU or GPU)
  retro_mask  retrospective vd-Poisson masks restricted to the acquired support, and their exports

Design: docs/superpowers/specs/2026-09-12-peds-tsc-mtp-grappa-design.md
"""
