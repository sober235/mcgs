% prep_asym_masks.m -- two house-convention vd-Poisson draws for the asymmetric-acceleration PROBE (2026-09-26):
% the same generator and settings as the jyz AF12/16/20/24 masks, vdPoisMex(256,176,256,176, accelx, 4, 24, 0, 1),
% one draw each at accelx 16 (nominal AF 64) and accelx 12 (nominal AF 48); accelx 8 gave only 26.7x. Probe masks, not topic masks: no pool, no
% exact-point-count pick. vdPoisMex reseeds from time(0) at one-second resolution -> pause(1.1) between calls.
% Run: matlab -batch "run('<this file>')"
addpath('/data0/congcong/code/ESPIRiT/utils');
outdir = '/data0/congcong/code/GS/mcgs_widen_wt/data';
for spec = {{'af64_probe', 16}, {'af48_probe', 12}}
    tag = spec{1}{1}; accelx = spec{1}{2}; accely = 4; calib = 24; ellipse = 0; pp = 1;
    fn = fullfile(outdir, sprintf('mask_vdpois_256x176_%s.mat', tag));
    if exist(fn, 'file'), error('refusing to overwrite %s', fn); end
    pause(1.1);
    mask = vdPoisMex(256, 176, 256, 176, accelx, accely, calib, ellipse, pp);
    mask = single(mask ~= 0);
    save(fn, 'mask', 'accelx', 'accely', 'calib', 'ellipse', 'pp', '-v7');
    fprintf('%s accelx %.1f nnz %d AF_plane %.3f\n', tag, accelx, nnz(mask), numel(mask) / nnz(mask));
end
