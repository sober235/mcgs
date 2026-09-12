% AF=11 house-convention vd-Poisson mask on the (ky,kz) = (256,176) PE plane, for the FLAIR channel only
% (user request 2026-09-12: T1 keeps the AF12 mask, FLAIR gets AF11). Same generator and settings as
% prep_02_mask.m: accely=4, accelx=AF/4, calib=24, ellipse=0, pp=1. vdPoisMex draws a fresh pattern, so this
% mask is not a superset of the AF12 one.
addpath('/data0/congcong/code/ESPIRiT/utils');
outdir = '/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_v3_jiangyizhou/data';
accely = 4; calib = 24; ellipse = 0; pp = 1;
for af = [11]
    accelx = af/accely;
    mask = vdPoisMex(256, 176, 256, 176, accelx, accely, calib, ellipse, pp);
    mask = single(mask ~= 0);
    n = nnz(mask); af_eff = numel(mask)/n;
    fn = fullfile(outdir, sprintf('mask_vdpois_256x176_af%02d.mat', af));
    save(fn, 'mask', '-v7');
    fprintf('AF_target %d accelx %.2f nnz %d AF_eff %.4f samp %.4f -> %s\n', ...
            af, accelx, n, af_eff, mean(mask(:)), fn);
end
exit
